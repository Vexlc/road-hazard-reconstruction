"""Train or replay a single joint StreetGS + SC-GS + 3DGUT checkpoint."""

import argparse
from collections import Counter
from importlib import import_module
import json
import os
from pathlib import Path
import site
import sys

import numpy as np
import torch
from PIL import Image
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prepare import load_rig


def configure(root, scene, output, official):
    street = root / "work/street_crafter"
    site.addsitedir(
        os.environ.get("SCGS_SITE", str(root / "work/envs/fisheye-sc-gs/lib/python3.11/site-packages"))
    )
    sys.path[:0] = [
        str(root / "work/joint-deps"),
        str(street),
        str(root / "work/EasyVolcap"),
        str(official / "SC-GS"),
        str(official / "3dgrut"),
    ]
    # StreetCrafter reads its own CLI and config at import time.
    argv = sys.argv
    sys.argv = [
        argv[0],
        "--config",
        str(Path(__file__).with_name("streetgs_joint.yaml")),
        "source_path",
        str(root / "work/streetcrafter-rig" / scene),
        "model_path",
        str(output),
    ]
    try:
        import_module("street_gaussian.config")
    finally:
        sys.argv = argv
    from model import JointScene

    if not OmegaConf.has_resolver("int_list"):
        OmegaConf.register_new_resolver("int_list", lambda values: list(map(int, values)))
    with initialize_config_dir(version_base=None, config_dir=str(official / "3dgrut/configs")):
        config = compose(
            config_name="apps/colmap_3dgut",
            overrides=[f"path={root}/work/streetcrafter-rig/{scene}/gut_train"],
        )
    return JointScene, config


def metadata(split, seedpath):
    obj = {int(k): v for k, v in json.loads((seedpath / "objects.json").read_text()).items()}
    with np.load(seedpath / "seeds.npz") as seeds:
        xyz = seeds["background"][:, :3]
    times = {s: [0.0] * len(split["frames"]) for s in range(len(split["sensors"]))}
    for n, v in split["views"].items():
        if not v["tile"]:
            times[split["sensors"].index(v["sensor"])][v["group"]] = split["times"][n]
    return dict(
        camera_tracklets=np.load(seedpath / "tracks.npy"),
        obj_meta=obj,
        camera_timestamps=times,
        scene_center=np.median(xyz, axis=0),
        scene_radius=float(np.quantile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 0.65)),
        num_frames=len(split["frames"]),
        num_cams=len(times),
    )


def main(scene, action, steps, root):
    from threedgrut.datasets.dataset_colmap import ColmapDataset

    official = Path(os.getenv("OFFICIAL_ROOT", root.parent.parent / "01_official")).resolve()
    output = root / "runs/streetgs-joint" / scene / str(steps)
    output.mkdir(exist_ok=True, parents=True)
    model_type, config = configure(root, scene, output, official)
    bundle = root / "work/streetcrafter-rig" / scene
    split, cams = load_rig(bundle)
    checkpoint = output / "joint.pt"
    train = action == "train"
    if train and checkpoint.exists():
        print("Completed checkpoint", checkpoint)
        return
    source = (
        bundle / "gut_train"
        if train
        else (bundle / "sensor_eval" if (bundle / "sensor_eval").exists() else bundle / "eval")
    )
    seedpath = root / "work/streetgs-joint" / scene
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    if train:
        model = model_type(metadata(split, seedpath), config)
        with np.load(seedpath / "seeds.npz") as seeds:
            model.initialize(seeds)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["scene"] != scene or state["steps"] != steps:
            raise ValueError("Checkpoint mismatch")
        model = model_type(state["metadata"], OmegaConf.create(state["renderer"]))
        model.restore(state)
    model.train(train)
    ds = ColmapDataset(str(source), test_split_interval=0)
    names = [Path(p).name for p in ds.image_paths]
    expected = {
        n
        for n in (split["train"] if train else split["train"] + split["test"])
        if not split["views"][n]["tile"]
    }
    if set(names) != expected:
        raise ValueError("Rig split mismatch")
    low, high = min(split["times"].values()), max(split["times"].values())
    views = []
    for n in names:
        v = split["views"][n]
        cam = split["sensors"].index(v["sensor"])
        e = model.exposure(
            v["group"],
            cam,
            split["times"][n],
            n in split["test"],
            cams[n],
            split["projections"]["eval"][v["projection"]],
        )
        t = torch.tensor((split["times"][n] - low) / (high - low), device="cuda", dtype=torch.float32)
        views.append((e, t))

    def batch(i):
        return ds.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([ds[i]]))

    if not train:
        folder = output / "frames"
        folder.mkdir(exist_ok=True)
        effect = []
        with torch.no_grad():
            for i, n in enumerate(names):
                b = batch(i)
                rgb, delta = model(b, *views[i])
                if not torch.isfinite(rgb).all():
                    raise ValueError(n)
                if n in split["test"] and delta is not None:
                    rigid, _ = model(b, *views[i], disable_warp=True)
                    effect.append(
                        dict(
                            name=n,
                            mean_displacement_m=delta.norm(dim=1).mean().item(),
                            mean_rgb_difference=(rgb - rigid).abs().mean().item(),
                            joint_psnr=(
                                -10 * (rgb.clamp(0, 1) - b.rgb_gt).square().mean().clamp_min(1e-12).log10()
                            ).item(),
                            warp_off_psnr=(
                                -10 * (rigid.clamp(0, 1) - b.rgb_gt).square().mean().clamp_min(1e-12).log10()
                            ).item(),
                        )
                    )
                Image.fromarray(rgb[0].clamp(0, 1).mul(255).round().byte().cpu().numpy()).save(folder / n)
        if not effect or max(v["mean_rgb_difference"] for v in effect) <= 1e-7:
            raise ValueError("The learned node field had no observable effect")
        (output / "warp_effect.json").write_text(json.dumps(effect, indent=2))
        return
    if steps < len(ds):
        raise ValueError("Use at least one full pass over the camera rig")
    from fused_ssim import fused_ssim

    batches = [batch(i) for i in range(len(ds))]
    optim = torch.optim.Adam(model.parameters_for_fit(), eps=1e-12, fused=True)
    base_rates = [g["lr"] for g in optim.param_groups]
    photos = Counter()
    proof = {}
    sample_order = []
    for step in range(1, steps + 1):
        if not sample_order:
            sample_order = list(rng.permutation(len(ds)))
        i = sample_order.pop()
        b = batches[i]
        n = names[i]
        photos[n] += 1
        model.degree = min(3, step // 800)
        rgb, delta = model(b, *views[i])
        pred = rgb.permute(0, 3, 1, 2).contiguous()
        gt = b.rgb_gt.permute(0, 3, 1, 2).contiguous()
        photo = 0.8 * (rgb - b.rgb_gt).abs().mean() + 0.2 * (1.0 - fused_ssim(pred, gt))
        if not proof and delta is not None:
            actor = getattr(model.graph, model.graph.graph_obj_list[0])
            gradients = torch.autograd.grad(
                photo,
                [model.graph.background._xyz, actor._xyz, model.warp.network.gaussian_warp.weight],
                retain_graph=True,
            )
            vals = [g.norm().item() for g in gradients]
            if min(vals) > 0 and np.isfinite(vals).all():
                proof = dict(
                    image=n,
                    iteration=step,
                    streetgs_background=vals[0],
                    streetgs_object=vals[1],
                    scgs_network=vals[2],
                )
        loss = photo if delta is None else photo + 0.001 * delta.square().mean()
        loss.backward()
        if step % 100 == 0 or step == 1:
            if not torch.isfinite(loss):
                raise FloatingPointError(step)
            print(f"{scene} {step}/{steps} photo={photo.item():.5f}", flush=True)
        for g, lr in zip(optim.param_groups, base_rates):
            g["lr"] = lr * (0.18 ** (step / steps))
        optim.step()
        optim.zero_grad(set_to_none=True)
    if not proof or set(photos) != expected:
        raise ValueError("Failed end-to-end training/coverage audit")
    state = model.checkpoint()
    state.update(renderer=OmegaConf.to_container(config, resolve=True), scene=scene, steps=steps)
    tmp = checkpoint.with_suffix(".tmp")
    torch.save(state, tmp)
    (output / "training.json").write_text(
        json.dumps(
            dict(
                photo_gradient_norms=proof,
                samples=photos,
                layers={n: len(getattr(model.graph, n).get_xyz) for n in model.graph.model_name_id},
                nodes=model.warp.node_num,
            ),
            indent=2,
        )
    )
    tmp.replace(checkpoint)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene")
    p.add_argument("action", choices=["train", "render"])
    p.add_argument("steps", type=int)
    args = p.parse_args()
    main(**vars(args), root=Path(__file__).resolve().parents[2])
