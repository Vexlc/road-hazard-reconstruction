"""Install surveyed above-ground props in pairs in the KITTI 30k joint checkpoint; replay from disk."""

import argparse
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from pit_pairs import BASE, ROOT, RIG, SCENE, C0, particles, rear_mount, hd_camera
from prepare import load_rig

PAIRS = [("H234", "B232"), ("H235", "B233")]


def scenes(args):
    if args.assembly:
        if not args.site:
            raise ValueError("--assembly requires its registered --site")
        return [(args.assembly, [(args.assembly, args.site)])]
    return [("-".join(pair), list(zip(pair, ["hydrant", "barrier"]))) for pair in PAIRS]


def build(args):
    sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
    from mesh_transfer import RoadMesh
    from prop_lighting import (
        blocked_sun,
        geometry_light,
        ground_confidence,
        ground_loss,
        irradiance,
        linear,
        srgb,
        surface_radiance,
    )
    from road_geometry import read_json, save_json, sha256 as digest

    baseline = torch.load(BASE / "joint.pt", map_location="cpu", weights_only=False)
    if baseline["steps"] != 30000 or baseline["scene"] != SCENE:
        raise ValueError("Expected the KITTI 0000 30k host")
    if "roadside_props" in baseline or "pit_pair" in baseline:
        raise ValueError("Use the original road checkpoint as the pair-building input")
    source = digest(BASE / "joint.pt")
    original = baseline["layers"]["background"]
    for name, objects in scenes(args):
        background = dict(original)
        dc = original["_features_dc"].clone()
        rest = original["_features_rest"].clone()
        additions = []
        records = []
        for key, site in objects:
            anchor = read_json(args.views / site / "anchor.json")
            if anchor.get("host_scene") != SCENE:
                raise ValueError("The anchor must belong to the same KITTI-360 coordinate system as the host")
            if (
                anchor["vehicle_clearance"]["minimum_clearance_m"]
                < anchor["vehicle_clearance"]["required_margin_m"]
            ):
                raise ValueError("Unverified vehicle clearance")
            path = args.assets / key
            mesh = RoadMesh(path)
            with np.load(path / "samples.npz", allow_pickle=False) as packed:
                samples = {name: packed[name] for name in packed.files}
            basis = np.asarray(anchor["basis_columns"])
            center = np.asarray(anchor["center_world"])
            pts = samples["xyz"]
            n = samples["normal"]
            lateral = np.where((abs(n[:, 2]) > 0.85)[:, None], [1.0, 0, 0], [0, 0, 1.0])
            tangent = np.cross(lateral, n)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
            rotation = basis @ np.stack([tangent, np.cross(n, tangent), n], axis=2)
            quat = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
            _, direct, sun = geometry_light(anchor)
            vis = 1 - blocked_sun(mesh, pts + n * 0.001, sun) if direct else 1.0
            gain = irradiance(n, samples["arm"][:, 0] / 255.0, anchor, vis, samples.get("sky_visibility"))
            optics = (
                (samples["dielectric_f0"], samples["specular_roughness"], None)
                if "dielectric_f0" in samples
                else None
            )
            rgb = (
                srgb(
                    surface_radiance(
                        samples["albedo"], gain, n, anchor, samples.get("sky_visibility"), vis, optics
                    )
                ).astype("f4")
                / 255.0
            )
            spacing = float(samples["spacing_m"])
            scales = np.broadcast_to([0.83 * spacing, 0.83 * spacing, 0.12 * spacing], pts.shape)
            additions.append(particles(pts @ basis.T + center, quat, scales, np.full(len(pts), 0.94), rgb))
            # Apply ground shadows to existing road surfels within measured receiver cells.
            local = (original["_xyz"].numpy() - center) @ basis
            near = (abs(local[:, 2]) <= 0.085) & (abs(local[:, :2]) < 3.6).all(1)
            rows = np.flatnonzero(near)
            loss = ground_loss(mesh, local[near, :2], anchor) * ground_confidence(anchor, local[near, :2])
            active = loss > 0.012
            rows, loss = rows[active], loss[active]
            if len(rows):
                original_srgb = (dc[rows, 0].numpy() * C0 + 0.5).clip(0, 1)
                shaded = srgb(linear(original_srgb * 255) * (1 - loss[:, None])).astype("f4") / 255
                dc[rows, 0] = torch.from_numpy((shaded - 0.5) / C0)
                rest[rows] *= torch.from_numpy(np.float32((1 - loss) ** 0.45))[:, None, None]
            records.append(
                dict(
                    id=key,
                    asset_sha256=mesh.digest,
                    samples_sha256=digest(path / "samples.npz"),
                    samples=len(pts),
                    anchor_sha256=digest(args.views / site / "anchor.json"),
                    center_world=center.tolist(),
                    basis_columns=basis.tolist(),
                    ground_gaussians_with_shadow=len(rows),
                    minimum_vehicle_clearance_m=anchor["vehicle_clearance"]["minimum_clearance_m"],
                )
            )
        background.update(_features_dc=dc, _features_rest=rest)
        if args.assembly:
            records[0]["instances"] = read_json(args.assets / args.assembly / "metadata.json")["instances"]
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(baseline, layers=dict(baseline["layers"]), roadside_props=records)
        checkpoint["layers"]["background"] = {
            k: torch.cat([background[k], *[item[k] for item in additions]], 0) for k in original
        }
        with TemporaryDirectory(dir=dest) as staging:
            tmp = Path(staging) / "joint.stage.pt"
            torch.save(checkpoint, tmp)
            loaded = torch.load(tmp, map_location="cpu", weights_only=False)
            for layer, tensors in baseline["layers"].items():
                for k, v in tensors.items():
                    new = loaded["layers"][layer][k]
                    if layer != "background":
                        assert torch.equal(v, new)
                    else:
                        assert len(new) == len(v) + sum(row["samples"] for row in records)
                        if k not in ("_features_dc", "_features_rest"):
                            assert torch.equal(new[: len(v)], v)
            tmp.replace(dest / "joint.pt")
        save_json(
            dest / "placement.json",
            dict(
                source_sha256=source,
                checkpoint_sha256=digest(dest / "joint.pt"),
                placements=records,
                original_count=len(original["_xyz"]),
                actors_exact=True,
                original_geometry_exact=True,
            ),
        )
        print(name, [(r["samples"], r["ground_gaussians_with_shadow"]) for r in records], flush=True)


def render(args):
    from omegaconf import OmegaConf
    from run import configure

    official = Path(os.getenv("OFFICIAL_ROOT", ROOT.parent.parent / "01_official")).resolve()
    os.chdir(official / "3dgrut")
    from threedgrut.datasets.dataset_colmap import ColmapDataset

    kind, _ = configure(ROOT, SCENE, BASE, official)
    base = torch.load(BASE / "joint.pt", map_location="cpu", weights_only=False)
    model = kind(base["metadata"], OmegaConf.create(base["renderer"]))
    split, cams = load_rig(RIG)
    ds = ColmapDataset(str(RIG / "sensor_eval"), test_split_interval=0)
    index = {Path(p).name: i for i, p in enumerate(ds.image_paths)}
    low, high = min(split["times"].values()), max(split["times"].values())
    relative, reark = rear_mount()
    model.eval()
    with torch.no_grad():
        for label, path in [("original", BASE / "joint.pt")] + [
            (name, args.out / name / "joint.pt") for name, _ in scenes(args)
        ]:
            model.restore(torch.load(path, map_location="cpu", weights_only=False))
            metrics = {}
            for camera in ["front", "rear"]:
                dest = args.renders / label / camera
                dest.mkdir(exist_ok=True, parents=True)
                changed = []
                errors = []
                template = None
                for group in range(len(split["frames"])):
                    name = f"{group:03}00.png"
                    b = ds.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([ds[index[name]]]))
                    w2c = cams[name]
                    k = split["projections"]["eval"][split["views"][name]["projection"]]
                    if camera == "rear":
                        k = reark
                        w2c = np.linalg.inv(relative) @ w2c
                        if template is None:
                            template = hd_camera(b, k)
                        b = replace(
                            template,
                            T_to_world=b.T_to_world @ b.T_to_world.new_tensor(relative),
                            frame_idx=b.frame_idx,
                        )
                    t = split["times"][name]
                    exposure = model.exposure(group, 0, t, name in split["test"], w2c, k)
                    if camera == "rear":
                        exposure[0].meta["is_novel_view"] = True
                    rgb, _ = model(
                        b,
                        exposure,
                        torch.tensor((t - low) / (high - low), device="cuda", dtype=torch.float32),
                    )
                    if not torch.isfinite(rgb).all():
                        raise ValueError("Nonfinite native replay")
                    arr = rgb[0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
                    if label == "original" and camera == "front":
                        errors.append(
                            int(abs(np.asarray(Image.open(BASE / "frames" / name), int) - arr).max())
                        )
                    if label != "original":
                        base_image = np.asarray(Image.open(args.renders / "original" / camera / name))
                        changed.append(int((abs(arr.astype(int) - base_image).max(2) > 8).sum()))
                    Image.fromarray(arr).save(dest / name, compress_level=2)
                if errors and max(errors) > 2 or changed and max(changed) < 50:
                    raise ValueError(f"No visible geometry or unreproduced original host: {label}/{camera}")
                metrics[camera] = dict(
                    frames=group + 1,
                    peak_changed_pixels=max(changed, default=0),
                    max_baseline_error=max(errors, default=0),
                )
            (args.renders / label / "replay.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(label, metrics, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["build", "render"])
    p.add_argument("--assets", type=Path, default=ROOT / "runs/roadside-props")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/roadside-props")
    p.add_argument("--out", type=Path, default=BASE / "roadside-props")
    p.add_argument("--renders", type=Path, default=ROOT / "outputs/roadside-props/joint")
    p.add_argument("--assembly", help="ID of a baked freestanding layout")
    p.add_argument("--site", help="Anchor subdirectory for the layout")
    args = p.parse_args()
    for key in ["assets", "views", "out", "renders"]:
        setattr(args, key, getattr(args, key).resolve())
    (build if args.task == "build" else render)(args)
