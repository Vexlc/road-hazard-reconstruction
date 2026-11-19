"""Fit selected library assets onto PandaSet roads and jointly rasterize them with scene actors."""
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from create_scene import create_scene
from export_reconstruction import export_splat
from export_gaussians import camera, sample_mesh
from gaussian_assets import C0, ROOT, linear, make_gaussians, srgb, tensor
from gaussian_road import make_road_apron, road_cut_mask
from polish_road_hazard import ATTRS, bake, set_background
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer

LIBRARY = ROOT / "outputs/road_hazard_expansion"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
LABELS = {
    "stereo": "双目实拍坑洞", "rgbd": "RGB-D 实拍坑洞", "pothole": "建模坑洞",
    "missing": "缺盖窨井", "raised": "井盖异常"
}


def rgb_image(arr):
    assert torch.isfinite(arr).all()
    return np.uint8(arr.permute(1, 2, 0).clamp(0, 1).cpu().numpy()*255+.5)


def setter(cam, width, height, focal):
    cam.image_width, cam.image_height = width, height
    cam.set_intrinsic(tensor([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]]))


def road_calibration(source, placement):
    mask = (source["road_mask"] > .85) & (source["alpha"] > .75)
    base = np.median(source["color"][mask, :3] if mask.sum() > 40 else source["color"][:, :3], axis=0)
    color = linear(tensor(np.stack([base, placement["calibration"]["target_srgb"]])))
    lum = color @ tensor([.2126, .7152, .0722])
    gain = float((lum[1]/lum[0].clamp_min(.012)).clamp(.12, 3.0))
    return {
        "source_srgb": base.tolist(), "target_srgb": placement["calibration"]["target_srgb"],
        "linear_gain": [gain]*3, "neutral_color_fraction": .76,
        "target_tint": (color[1]/lum[1].clamp_min(.01)).clamp(.8, 1.24).tolist()
    }


def source_fields(folder, samples, meshes, metadata):
    if metadata["category"] in {"stereo", "rgbd"}:
        with np.load(samples / (metadata["id"]+".npz")) as data:
            src = dict(data)
        src.update(name=metadata["id"], geometric_ao=True)
        return src
    return sample_mesh(meshes / (metadata["id"]+".npz"), folder / "surface.npz", metadata)


def video_options(fps, quality=8):
    return dict(fps=fps, codec="libx264", quality=quality, macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])


def panel(asset_meta, viewport, label, zoom=None, mode=None):
    width, height = 1280, 800
    sheet = Image.new("RGB", (width, height), "#121b24")
    ink = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 24)
    g = ImageFont.truetype(FONT, 18)
    ink.text((21, 13), f'{LABELS[asset_meta["category"]]}  ·  PandaSet {label}', font=f, fill="#eef5fa")
    size = f'{asset_meta["width_m"]*100:.0f} × {asset_meta["length_m"]*100:.0f} cm'
    ink.text((22, 49), size+('  ·  '+mode if mode else ''), font=g, fill="#9dcbd8")
    ink.text((width-610, 49), asset_meta["id"], font=g, fill="#becbd5")
    image = Image.fromarray(viewport)
    if zoom:
        image = image.crop(zoom)
    image = image.resize((width, 720), Image.Resampling.LANCZOS)
    sheet.paste(image, (0, 80))
    return np.asarray(sheet)


@torch.no_grad()
def one_case(scene, renderer, cameras, source_state, meta, placement, folder, data, film, write_scene):
    gaussians = scene.gaussians
    bg = gaussians.background
    center = np.array(placement["center_world"])
    basis = np.array(placement["basis_columns"])
    out = LIBRARY / "scene_trials" / (Path(cfg.source_path).name + "_" + meta["id"])
    out.mkdir(exist_ok=True, parents=True)
    set_background(bg, source_state)
    cut = road_cut_mask(bg, [placement])
    road, road_info = make_road_apron(scene, renderer, cameras, [placement])
    calibration = road_calibration(data, placement)
    gauss, fit_error = make_gaussians(data, center=center, basis=basis, calibration=calibration)
    # Preserve the measured defect, replacing the surrounding texture with the target road.
    local = data["local_xyz"]
    weight = data["road_mask"].astype("f4") * np.clip((local[:, 2]+.023)/.02, 0, 1)
    if meta["category"] in {"missing", "raised"}:
        radius = np.linalg.norm(local[:, :2], axis=1)
        weight *= np.clip((radius-max(meta["width_m"], meta["length_m"])*.69)/.16, 0, 1)
    indices = np.where(weight > .03)[0]
    need = tensor(indices).long()
    c, no_input = bake(scene, renderer, cameras, gauss["xyz"][need])
    shade = .31 + .69*data["ao"][indices]
    rgb = srgb((linear(c.clamp(0, 1))*tensor(shade)).clamp(0, 1))
    mix = tensor(weight[indices, None])
    gauss["feature_dc"][need, 0] = (rgb-.5)/C0*mix + gauss["feature_dc"][need, 0]*(1-mix)
    gauss["feature_rest"][need] *= 1-mix[:, None]
    keep = tensor(~cut).bool()
    merged = {
        key: torch.nn.Parameter(torch.cat([source_state[key][keep], gauss[key], road[key]], dim=0),
                                requires_grad=False)
        for key in ATTRS
    }
    set_background(bg, merged)
    label = Path(cfg.source_path).name
    with imageio.get_writer(out / "drive.mp4", **video_options(10)) as drive:
        for i, cam in enumerate(cameras):
            scene_rgb = rgb_image(renderer.render(cam, gaussians)["rgb"])
            drive.append_data(scene_rgb)
            if i == 39:
                Image.fromarray(scene_rgb).save(out / "driver.jpg", quality=93)
            if 31 <= i < 46:
                presented = panel(meta, scene_rgb, label, mode="原行车轨迹")
                for _ in range(2):
                    film.append_data(presented)
    if write_scene:
        export_splat(gaussians, cameras[40], out / "scene_frame040.ply")
        checkpoint = gaussians.save_state_dict(is_final=True)
        checkpoint["iter"] = 7001
        checkpoint_folder = out / "trained_model"
        checkpoint_folder.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_folder / "iteration_7000.pth")
        del checkpoint
        (out / "render_config.yaml").write_text(
            f"parent_cfg: {ROOT}/work/street_crafter/configs/pandaset_{label}_warmup.yaml\n"
            f"model_path: {out}\nmode: trajectory\nloaded_iter: 7000\nresolution: 1280\n"
        )
    cam = cameras[40]
    old_pose, old_intrinsics = cam.get_extrinsic().clone(), cam.get_intrinsic().clone()
    up = basis[:, 2]
    offset = old_pose[:3, 3].cpu().numpy()-center
    far = offset-(offset@up)*up
    far /= np.linalg.norm(far)
    right = np.cross(up, far)
    very_small = max(meta["width_m"], meta["length_m"]) < .24
    mo = meta["category"] in {"missing", "raised"}
    removed = meta.get("state") == "removed_side"
    if very_small:
        radius, rise, focal = .95, .92, 2300
    elif removed:
        radius, rise, focal = 2.48, 2.01, 1280
    else:
        radius, rise, focal = 1.65, 1.42, 1570
    target = center + (.11*up if mo else .005*up)
    if removed:
        target = target + basis[:, 0]*.30
    setter(cam, 1280, 720, focal)
    if mo and meta["category"] == "raised":
        angle = np.deg2rad(-20 + 135*(.5-.5*np.cos(np.linspace(0, np.pi, 45))))
    else:
        angle = np.deg2rad(23*np.sin(np.linspace(-np.pi/2, np.pi/2, 45)))
    with imageio.get_writer(out / "closeup.mp4", **video_options(20)) as near:
        for i, a in enumerate(angle):
            position = center + radius*(np.cos(a)*far+np.sin(a)*right) + rise*up
            cam.set_extrinsic(tensor(np.linalg.inv(camera(position, target))))
            frame = rgb_image(renderer.render(cam, gaussians)["rgb"])
            near.append_data(frame)
            if i == 20:
                Image.fromarray(frame).save(out / "closeup.jpg", quality=94)
                set_background(bg, source_state)
                before = rgb_image(renderer.render(cam, gaussians)["rgb"])
                Image.fromarray(np.concatenate([before, frame], axis=1)).save(out / "before_after_near.jpg", quality=94)
                set_background(bg, merged)
            crop = (300, 170, 980, 552) if very_small else None
            film.append_data(panel(meta, frame, label, zoom=crop,
                                   mode="插入位置近景" + ("（裁切放大）" if crop else "")))
    cam.set_extrinsic(old_pose)
    cam.set_intrinsic(old_intrinsics)
    set_background(bg, source_state)
    summary = {
        "id": meta["id"], "site": label, "input_asset": str(folder), "placement": placement,
        "removed_road_gaussians": int(cut.sum()), "source_actor_count": gaussians.metadata.get("num_objects")
        if hasattr(gaussians, "metadata") else None,
        "asset_gaussians": len(gauss["xyz"]), "road_apron": road_info,
        "color_fit_unobserved_fraction": no_input, "degree1_mae": fit_error,
        "placement_scale_multiplier": 1.0, "drive_frames": len(cameras), "drive_fps": 10,
        "closeup_frames": len(angle), "closeup_fps": 20, "width": 1280, "height": 720,
        "dynamic_components_reused": True, "full_editable_checkpoint": write_scene
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("INSERTED", label, meta["id"], "cut", int(cut.sum()), flush=True)


@torch.no_grad()
def main():
    ids = os.environ["ROAD_ASSET_IDS"].split(",")
    meshes = Path(os.environ["ROAD_MESH_CACHE"])
    samples = Path(os.environ["ROAD_CAPTURE_CACHE"])
    keep_scene = set(os.environ.get("ROAD_EXPORT_SCENES", "").split(","))
    all_assets = {path.parent.name: path.parent for path in LIBRARY.glob("*/*/metadata.json")}
    site = Path(cfg.source_path).name
    tag = "A" if site == "056" else "B"
    placements = json.loads((ROOT / "outputs" / f"inserted_{tag}_{site}" / "placement.json").read_text())["placements"]
    scene = create_scene()
    renderer = StreetGaussianRenderer()
    cameras = sorted(scene.getTrainCameras()+scene.getTestCameras(), key=lambda c: c.meta["frame"])
    for c in cameras:
        k = c.K.clone()
        k[0] *= 1280 / c.image_width
        k[1] *= 720 / c.image_height
        c.image_width, c.image_height = 1280, 720
        c.set_intrinsic(k)
    base = {key: getattr(scene.gaussians.background, attr) for key, attr in ATTRS.items()}
    film_path = LIBRARY / f"scene_trials_{site}.mp4"
    with imageio.get_writer(film_path, **video_options(20)) as film:
        for name in ids:
            folder = all_assets[name]
            meta = json.loads((folder / "metadata.json").read_text())
            placement = dict(placements[0 if meta["category"] in {"missing", "raised"} else 1])
            placement.update(asset=name, height_field=str(folder / "surface.npz"), close_road_resampling=True)
            data = source_fields(folder, samples, meshes, meta)
            one_case(scene, renderer, cameras, base, meta, placement, folder, data, film, name in keep_scene)
    print("SCENE TRIALS COMPLETE", site, len(ids), flush=True)


if __name__ == "__main__":
    main()
