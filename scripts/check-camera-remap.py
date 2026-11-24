"""Compare resampled moving exposures with transported native Cycles ID and range passes."""

import argparse
from importlib import import_module
from pathlib import Path
import numpy as np
from PIL import Image

rolling = import_module("check-rolling-covers")
from road_geometry import save_json
from camera_remap import remap_packet
from host_depth import geometry


def inverse_map(mode, size):
    """Two inverse maps with integer pixel centres, for an ideal 220 degree equidistant host."""
    y, x = np.mgrid[:320, :442]
    if mode == "rotated-crop":
        return np.stack((size * .5 + (y - 159.5) * .82, size * .5 - 35. - (x - 220.5) * .88), -1)
    # Rectilinear output rotated about its optical axis, then projected into the raw fisheye.
    u, v = (x - 220.5) / 197., (y - 159.5) / 197.
    angle = np.deg2rad(52.)
    a, b = u * np.cos(angle) - v * np.sin(angle), u * np.sin(angle) + v * np.cos(angle)
    tilt = np.deg2rad(-15.)
    b, z = b * np.cos(tilt) + np.sin(tilt), np.cos(tilt) - b * np.sin(tilt)
    radius = np.hypot(a, b)
    scale = (size / np.deg2rad(220.)) * np.arctan2(radius, z) / radius
    return np.stack((a * scale + (size - 1) / 2, b * scale + (size - 1) / 2), -1)


def check_packet_edges():
    """Check non-central subrays, signed Z, invalid lens coordinates and exact half-pixel ties."""
    rays = np.broadcast_to([0., 0., 1.], (2, 2, 3, 3)).copy()
    starts = np.zeros_like(rays)
    starts[1, ..., 2] = .125
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    view = dict(rgb=rgb, rays_camera=rays, ray_origins_camera=starts, host_z=np.full((2, 3), 4.),
                camera_to_world=np.eye(4), protected=np.zeros((2, 3), bool),
                receiver_mask=np.ones((2, 3), bool), road_albedo_srgb=rgb, sensor_name=np.array("rear"))
    xy = np.array([[[-.5, 0], [.5, .5], [2.49999, 1], [2.5, 0], [np.nan, 0], [1e30, 1]]])
    mapped = remap_packet(view, xy)
    _, ranges, protected, _, origins = geometry(mapped, with_origins=True)
    np.testing.assert_array_equal(mapped["rgb"][0, :3], rgb[[0, 1, 1], [0, 1, 2]])
    np.testing.assert_array_equal(mapped["road_albedo_srgb"], mapped["rgb"])
    np.testing.assert_array_equal(ranges[:, 0, :3] + origins[:, 0, :3, 2], np.full((2, 3), 4.))
    if (protected[:, :, :3].any() or not protected[:, :, 3:].all() or not np.isinf(ranges[:, :, 3:]).all()
            or mapped["receiver_mask"][:, :, 3:].any() or mapped["rgb"][:, 3:].any()
            or mapped["sensor_name"].item() != "rear"):
        raise ValueError("Invalid lens boundary or changed camera metadata")


def write_oracle(previous, dest):
    """Extract native reference passes after running the eight-pose Cycles experiment."""
    from shutil import copyfile

    pair = "M248-A249"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("top-down", "bottom-up"):
        _, ids, depth, _ = rolling.assemble(previous / "cycles" / pair, name == "bottom-up")
        np.save(dest / (name + "-native-id.npy"), ids)
        np.save(dest / (name + "-native-range.npy"), depth)
        copyfile(previous / "oracle" / pair / (name + ".npz"), dest / (name + ".npz"))
    copyfile(previous / "oracle" / pair / "appearance.json", dest / "appearance.json")


def check(source, assets, recipe, out):
    check_packet_edges()
    dest = out / "maps"
    dest.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for shot in ("top-down", "bottom-up"):
        source_file = source / (shot + ".npz")
        with np.load(source_file, allow_pickle=False) as data:
            raw = dict(data)
        ids = np.load(source / (shot + "-native-id.npy"))
        depth = np.load(source / (shot + "-native-range.npy"))
        for name in ("rotated-crop", "rectilinear"):
            target = out / shot / name
            target.mkdir(parents=True, exist_ok=True)
            xy = inverse_map(name, ids.shape[0])
            np.save(dest / (name + ".npy"), xy)
            rolling.rig.cli("remap-camera", source_file, dest / (name + ".npy"), target / "camera.npz")
            rolling.rig.cli("render", recipe, source / "appearance.json", target / "camera.npz", target, "--asset-root", assets)
            y, x = np.floor(xy[..., ::-1] + .5).astype(int).transpose(2, 0, 1)
            native_ids, native_depth = ids[y, x], depth[y, x]
            rows = raw["camera_to_world_rows"]
            # Deliberately wrong control: assign every ray the pose of its output row's centre pixel.
            with np.load(target / "camera.npz", allow_pickle=False) as packet:
                wrong = dict(packet)
            wrong["rays_camera"] = raw["rays_camera"][y, x]
            wrong.pop("ray_origins_camera")
            wrong["camera_to_world_rows"] = rows[y[:, y.shape[1] // 2]]
            np.savez_compressed(target / "row-only.npz", **wrong)
            rolling.rig.cli("render", recipe, source / "appearance.json", target / "row-only.npz", target / "row-only", "--asset-root", assets)
            keys = recipe.stem.split("-")
            correct = rolling.scores(target, native_ids, native_depth, keys)
            baseline = rolling.scores(target / "row-only", native_ids, native_depth, keys)
            after = np.asarray(Image.open(target / "after.png"))
            blocked = wrong["protected"] | np.isfinite(wrong["occluder_range"])
            if not np.array_equal(after[blocked], raw["rgb"][y[blocked], x[blocked]]):
                raise ValueError("Edited protected foreground")
            for key, row in correct.items():
                if (row["instance_iou"] < .97 or row["part_agreement"] < .98 or row["p95_range_error_m"] > .001
                        or row["part_agreement"] <= baseline[key]["part_agreement"]):
                    raise ValueError(f"Remapped visibility disagrees with native render for {shot}/{name}/{key}: {row} vs {baseline[key]}")
            all_rows[shot + "/" + name] = dict(registered=correct, output_row_pose=baseline, protected_pixels=int(blocked.sum()))
    save_json(out / "check.json", all_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Two registered raw packets, their corresponding native passes and appearance.json")
    p.add_argument("assets", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--from-cycles", type=Path, help="Rebuild the source packet/reference directory from the rolling-covers experiment")
    a = p.parse_args()
    if a.from_cycles:
        write_oracle(a.from_cycles, a.source)
    check(a.source, a.assets, rolling.ROOT / "gen2hole/examples/rolling-covers/M248-A249.json", a.out)
