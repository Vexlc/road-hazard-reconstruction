"""Render the two registered source viewpoints, surface labels and a stopping approach."""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from stereo import load, camera
from build import HERE, Surface
from photo_mesh import LIB
from mesh_transfer import RoadMesh

sys.path.insert(0, str(LIB.parent / "scripts"))
view = import_module("check-verge-chamber")


def source_camera(pair, k):
    width, height = pair.s["size"]
    yy, xx = np.indices((height, width))
    optical = np.stack(
        [(xx - pair.center[0]) / pair.f[k], (yy - pair.center[1]) / pair.f[k], np.ones_like(xx)],
        axis=-1,
    )
    rotation = pair.basis.T if k == 0 else pair.R @ pair.basis.T
    return optical @ rotation, rotation


def native_view(state, eye, rotation, intrinsic, width, height):
    import torch
    from atlas_gaussians import gpu
    from gsplat.rendering import rasterization

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    with torch.inference_mode():
        rendered, alpha, _ = rasterization(
            **state,
            viewmats=gpu(extrinsic)[None],
            Ks=gpu(intrinsic)[None],
            width=width,
            height=height,
            sh_degree=1,
            near_plane=0.05,
            far_plane=120.0,
            rasterize_mode="antialiased",
            backgrounds=gpu([61 / 255] * 3)[None]
        )
    return (rendered[0].cpu().numpy().clip(0, 1) * 255 + 0.5).astype("u1"), alpha[
        0, :, :, 0
    ].cpu().numpy()


def check(folder, out):
    from atlas_gaussians import load as load_native

    out.mkdir(parents=True, exist_ok=True)
    pair = load()
    meta = json.loads((folder / "metadata.json").read_text())
    shift = np.array(meta["asset_world_origin_m"])
    mesh = RoadMesh(folder)
    state = load_native(
        dict(id="P564", gaussians=meta["gaussians"], ply=(folder / "asset.ply").resolve())
    )
    width, height = pair.s["size"]
    report = {"stereo": pair.report(), "source_views": {}}
    tiles = []
    for k, name in enumerate(("west", "northwest")):
        eye = pair.eye(k) - shift
        directions, rotation = source_camera(pair, k)
        source = Image.open(HERE / "references" / (name + ".jpg")).convert("RGB")
        rgb, part, depth = view.render(mesh, eye, directions)
        np.savez_compressed(
            out / (name + "-rays.npz"),
            part=part,
            range_m=(depth * np.linalg.norm(directions, axis=-1)).astype("f4"),
        )
        native, opacity = native_view(
            state, eye, rotation, camera(pair.f[k], pair.center), width, height
        )
        water = (part == 4) & (opacity > 0.8)
        assert water.sum() > 1000 and np.isfinite(opacity).all()
        ref = np.asarray(source).astype(float)
        report["source_views"][name] = dict(
            mesh_visible_pixels=int((part > 0).sum()),
            native_alpha_above_08=int((opacity > 0.8).sum()),
            water_pixels_checked=int(water.sum()),
            water_mean_absolute_channel_error_mesh=float(abs(rgb - ref)[water].mean()),
            water_mean_absolute_channel_error_splats=float(abs(native - ref)[water].mean()),
        )
        probes = [
            dict(x=x, y=y, expected=c, actual=int(part[y, x]))
            for c, x, y in pair.s["field_checks"][name]
        ]
        report["source_views"][name]["component_probes"] = probes
        assert all(p["expected"] == p["actual"] for p in probes), probes
        assert abs(native - ref)[water].mean() < abs(rgb - ref)[water].mean()
        for label, image in [
            ("Source photograph", source),
            ("Reconstructed mesh", Image.fromarray(rgb)),
            ("Native SH1 splats", Image.fromarray(native)),
        ]:
            thumb = image.resize((400, 267), Image.Resampling.LANCZOS)
            tiles.append((name + " | " + label, thumb))
            if label != "Source photograph":
                image.save(
                    out
                    / (name + ("-mesh.jpg" if label == "Reconstructed mesh" else "-splats.jpg")),
                    quality=94,
                )
    trace = np.asarray(meta["blue_centerline_m"])
    # Shoot vertically along the middle stretch. Record the independently intersected layers.
    mid = (trace[2:7] + trace[3:8]) / 2
    origin = mid.copy()
    origin[:, 2] = 2
    down = np.broadcast_to([0.0, 0.0, -1.0], origin.shape)
    layers = []
    for exclude in (set(), {59}, {59, 4}):
        dist, parts, _ = Surface(folder / "asset.glb", exclude=exclude).cast(origin, down)
        layers.append(dict(parts=parts.tolist(), z_m=(2 - dist).tolist()))
    report["vertical_crossing"] = layers
    assert all(p == 59 for p in layers[0]["parts"]) and sum(p == 4 for p in layers[1]["parts"]) >= 3
    assert all(
        layers[2]["z_m"][i] < layers[1]["z_m"][i] < layers[0]["z_m"][i]
        for i, p in enumerate(layers[1]["parts"])
        if p == 4
    )
    heights = np.load(folder / "surface.npz")
    pitch = heights["length"] / (heights["alpha"].shape[0] - 1)
    report["water_visible_plan_area_model_m2"] = float(heights["water_mask"].sum() * pitch**2)
    report["approach"] = film(mesh, state, out, pair, shift)
    sheet = Image.new("RGB", (1200, 590), "#202524")
    draw = ImageDraw.Draw(sheet)
    for i, (title, tile) in enumerate(tiles):
        x, y = (i % 3) * 400, (i // 3) * 295
        draw.text((x + 8, y + 6), title, fill="white")
        sheet.paste(tile, (x, y + 25))
    sheet.save(out / "comparison.jpg", quality=93)
    (out / "quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def film(mesh, state, out, pair, shift):
    # Both endpoints are on the surviving road traced in the west photograph.
    start, stop = pair.plane([[60, 650], [170, 390]], 0) - shift
    target = pair.plane([[373, 297]], -0.19)[0] - shift
    route = np.linspace(start, stop, 36) + [0, 0, 1.0]
    _, types, _ = view.render(mesh, route, np.broadcast_to([0.0, 0.0, -1.0], route.shape))
    assert (types == 13).all()
    writer = imageio_ffmpeg.write_frames(
        str(out / "approach.mp4"), (768, 432), fps=12, codec="libx264", output_params=["-crf", "19"]
    )
    writer.send(None)
    try:
        for t in np.linspace(0, 1, 36):
            eye = stop + (start - stop) * (1 - t) ** 2 + [0, 0, 1.45]
            rays, rotation = view.view_rays(eye, target, 546.0, 768, 432)
            rgb, alpha = native_view(state, eye, rotation, camera(546.0, [384, 216]), 768, 432)
            assert (alpha > 0.8).sum() > 4000
            if t == 1:
                Image.fromarray(rgb).save(out / "approach-stop.jpg", quality=94)
            writer.send(np.ascontiguousarray(rgb))
    finally:
        writer.close()
    lip = pair.plane(pair.s["near_lip_west"], 0) - shift
    a = lip[:-1, :2]
    delta = np.diff(lip[:, :2], axis=0)
    fraction = np.clip(np.sum((stop[:2] - a) * delta, axis=1) / np.sum(delta**2, axis=1), 0, 1)
    return dict(
        camera_height_m=1.45,
        frames=36,
        travel_model_m=float(np.linalg.norm(start - stop)),
        stop_to_nearest_traced_lip_model_m=float(
            np.linalg.norm(stop[:2] - a - fraction[:, None] * delta, axis=1).min()
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.asset, args.out)
