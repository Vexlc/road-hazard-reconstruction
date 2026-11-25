"""Export native Cycles position passes and compare their registered optics, geometry and occlusion."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
from PIL import Image

reference = import_module("check-cone-cycles")
from host_position import register_positions
from road_geometry import save_json

SHOTS = ("rear", "ortho", "side")
# Change the map origin after capture. Author the Cartesian pass in a local
# left-handed millimetre frame; all three axes differ from the host world axes.
MAP_OFFSET = np.array([503218.0, 5417291.0, 116.0])
BASIS = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def pack(view, dest, pair):
    dest.mkdir(exist_ok=True, parents=True)
    for shot in SHOTS:
        folder = view / pair / shot
        host = folder / "host"
        meta = json.loads((host / "camera.json").read_text())
        size = meta["size"]
        y, x = (np.mgrid[:size, :size] + 0.5) / size - 0.5
        pose = np.array(meta["camera_to_world"])
        field = np.load(host / "position.npy")
        original = pose[:3, 3].copy()
        pose[:3, 3] += MAP_OFFSET
        world_from_position = np.eye(4)
        world_from_position[:3, :3] = BASIS / 1000
        world_from_position[:3, 3] = pose[:3, 3]
        known = np.load(host / "id.npy")
        valid = known > 0
        packet = dict(
            rgb=np.array(Image.open(host / "rgb.png").convert("RGB")),
            camera_to_world=pose,
            surface_position=((field.astype(float) - original) @ BASIS * 1000).astype(
                "f4"
            ),
            position_valid=valid,
            receiver_mask=known == 1,
            occluder_mask=known == 2,
            world_from_position=world_from_position,
        )
        shift = np.zeros_like(x)
        if shot == "ortho":
            width = meta["ortho_width_m"]
            shift = 0.06 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
            packet["ray_origins_camera"] = np.float32(
                np.stack([width * x, width * y, shift], axis=-1)
            )
            rays = np.zeros((size, size, 3), "f4")
            rays[..., 2] = 1
            protected = np.zeros_like(valid)
        else:
            f = size / np.deg2rad(meta["fov_deg"])
            rays = reference.pixel_rays(
                ["OPENCV_FISHEYE", size, size, f, f, size / 2, size / 2, 0, 0, 0, 0]
            )
            protected = x * x + y * y > 0.25
        packet.update(rays_camera=rays, protected=protected)
        target = dest / shot
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / "input.npz", **packet)
        nids = np.load(folder / "native/id.npy")
        np.savez_compressed(
            target / "reference.npz",
            host_range=np.where(valid, np.load(host / "range.npy") - shift, np.inf),
            native_id=nids,
            native_range=np.where(
                nids > 0, np.load(folder / "native/range.npy") - shift, np.inf
            ),
        )
        world = np.array(meta["reference_ground_to_world"])
        world[:3, 3] += MAP_OFFSET
        np.save(dest / "road-frame.npy", world)


def inspect(src, result, packet):
    with np.load(src / "reference.npz") as truth, np.load(src / "input.npz") as raw:
        protected = packet["protected"][0]
        foreground = np.isfinite(packet["occluder_range"][0])
        raw_id = truth["native_id"]
        target = raw_id // 32
        hit = np.array(Image.open(result / "instance.png"))
        part = np.array(Image.open(result / "part.png"))
        range_map = np.load(result / "range.npy")
        color = np.array(Image.open(result / "after.png"))
        if hit[protected | foreground].any() or np.any(
            color[protected | foreground] != packet["rgb"][protected | foreground]
        ):
            raise ValueError("Edited foreground or unregistered position")
        valid = ~protected
        err = abs(packet["host_range"][0][valid] - truth["host_range"][valid])
        scores = []
        for k in (1, 2):
            agree = (hit == k) & (target == k) & (part == raw_id % 32)
            union = (hit == k) | (target == k)
            iou = ((hit == k) & (target == k)).sum() / union.sum()
            accuracy = agree.sum() / (target == k).sum()
            delta = np.percentile(
                abs(range_map[agree] - truth["native_range"][agree]), 95
            )
            if (
                not np.isfinite([iou, accuracy, delta]).all()
                or iou < 0.961
                or accuracy < 0.95
                or delta > 0.003
            ):
                raise ValueError(
                    f"Native geometry mismatch {src}/{k}: {iou}, {accuracy}, {delta}"
                )
            scores.append(
                dict(
                    instance_iou=float(iou),
                    part_recall=float(accuracy),
                    range_p95_m=float(delta),
                    verified_floor_pixels=int((agree & (part == 7)).sum()),
                    matched=int(agree.sum()),
                )
            )
        behind = raw["rays_camera"][..., 2] < 0
        if err.max() > 0.00025:
            raise ValueError("Position pass disagrees with native depth")
        return dict(
            host_samples=int(valid.sum()),
            host_max_error_m=float(err.max()),
            occluder_pixels=int(foreground.sum()),
            protected_pixels=int(protected.sum()),
            protected_edits=0,
            matched_behind_lens=int((behind & (target == hit) & (hit > 0)).sum()),
            instances=scores,
        )


def check_bearing(models):
    import open3d as o3d
    from pavement_layout import asset_folder
    from mesh_transfer import RoadMesh

    x, y = np.meshgrid(np.arange(-0.39, 0.391, 0.003), np.arange(-0.39, 0.391, 0.003))
    uv = np.stack([x, y], axis=-1)
    rays = np.zeros((*x.shape, 6), "f4")
    rays[..., :2] = uv
    rays[..., 2], rays[..., 5] = 1.0, -1.0
    ray = o3d.core.Tensor(rays)
    parts, heights = [], []
    for key in ("A270", "M270"):
        folder = asset_folder(models, key)
        mesh = RoadMesh(folder)
        hit = mesh.scene.cast_rays(ray)
        codes = hit["geometry_ids"].numpy()
        part = np.zeros(x.shape, "u1")
        for index, material in mesh.surfaces.items():
            part[codes == index] = material[0]
        heights.append(1 - hit["t_hit"].numpy())
        parts.append(part)
    folder = asset_folder(models, "A270")
    meta = json.loads((folder / "metadata.json").read_text())
    underside = meta["lid_transform"][2][3] - meta["lid_thickness_m"]
    metal = parts[0] == 9
    gap = underside - heights[1][metal]
    bearing = uv[metal][abs(gap) < 0.0005]
    contacts = [
        int(np.sum((bearing[:, 0] * i > 0) & (bearing[:, 1] * j > 0)))
        for i in (-1, 1)
        for j in (-1, 1)
    ]
    open_corner = int(((parts[0] == 7) & (parts[1] == 7)).sum())
    if gap.min() < -0.0003 or min(contacts) < 12 or open_corner < 80:
        raise ValueError(
            "The rotated square lid must rest on four corners and expose the shaft"
        )
    return dict(
        grid_pitch_m=0.003,
        contact_cells_by_quadrant=contacts,
        exposed_floor_cells=open_corner,
        min_clearance_m=float(gap.min()),
        support_margin_m=meta["support_margin_m"],
    )


def check_contract():
    rays = np.array(
        [[[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], [[0.6, 0.0, 0.8], [1.0, 0.0, 0.0]]]] * 2
    )
    rows = np.repeat(np.eye(4)[None, None], 4, axis=0).reshape(2, 2, 4, 4)
    rows[1, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rows[:, :, 0, 3] = [[0.2, 0.3], [0.45, 0.5]]
    origins = np.zeros_like(rays)
    origins[..., 1] = 0.032
    ref = np.eye(4)
    anchor = np.array([2030000.0, 4910000.0, 100.0])
    ref[:3, 3] = anchor
    ranges = np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.2, 2.1], [3.1, 4.1]]])
    xyz = (
        np.einsum(
            "shij,shwj->shwi", rows[..., :3, :3], origins + rays * ranges[..., None]
        )
        + rows[:, :, :3, 3][:, :, None]
    )
    rows[:, :, :3, 3] += anchor
    rgb = np.full((2, 2, 3), 120, "u1")
    world = np.eye(4)
    world[:3, 3] = anchor
    world[:3, :3] = BASIS / 1000
    known = np.ones(ranges.shape, bool)
    known[1, 1, 1] = False
    source = dict(
        surface_position=xyz @ BASIS * 1000,
        world_from_position=world,
        position_valid=known,
        rays_camera=rays,
        ray_origins_camera=origins,
        camera_to_world=ref,
        camera_to_world_rows=rows,
        protected=np.zeros_like(known),
        rgb=rgb,
        sample_rgb=np.stack([rgb, rgb]),
        sample_weights=np.array([1.0, 2.0]),
    )
    source.update(
        road_albedo_srgb=rgb,
        reflector_labels=np.ones(known.shape, "u2"),
        receiver_band_m=0.016,
    )
    packet = register_positions(source)
    np.testing.assert_array_equal(packet["reflector_labels"], known.astype("u2"))
    np.testing.assert_array_equal(packet["road_albedo_srgb"], rgb)
    assert packet["receiver_band_m"] == 0.016
    np.testing.assert_allclose(
        packet["host_range"][known], ranges[known], atol=1e-8, rtol=0
    )
    assert packet["protected"][~known].all() and "camera_to_world_rows" not in packet
    mirrored = world.copy()
    mirrored[:3, 0] *= -1
    sheared = world.copy()
    sheared[0, 2] += 0.0002
    behind = dict(source, surface_position=-source["surface_position"])
    for invalid in (
        dict(source, protected=known, receiver_mask=known, occluder_mask=known),
        dict(source, world_from_position=mirrored),
        dict(source, world_from_position=sheared),
        behind,
        dict(source, surface_position=source["surface_position"][::-1]),
        dict(source, host_z=ranges),
    ):
        try:
            register_positions(invalid)
        except ValueError:
            continue
        raise ValueError("Accepted an unregistered host pass")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["pack", "check"])
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layout", type=Path, action="append", required=True)
    a = p.parse_args()
    stats = {}
    for layout in a.layout:
        pair = json.loads(layout.read_text())["id"]
        dst = a.out / pair
        dst.mkdir(exist_ok=True, parents=True)
        if a.command == "pack":
            pack(a.views, dst, pair)
            continue
        src = a.views / pair
        world = np.load(src / "road-frame.npy")
        for shot in SHOTS:
            reference.cli(
                "from-position", src / shot / "input.npz", dst / (shot + ".npz")
            )
        reference.cli("road-points", dst / "road.npy", dst / "ortho.npz", "--stride", 3)
        reference.cli(
            "fit",
            dst / "road.npy",
            dst / "anchor.json",
            "--seed",
            *world[:3, 3],
            "--heading",
            *world[:3, 0],
            "--up",
            *world[:3, 2],
            "--radius",
            1.05,
            "--asset",
            layout,
            "--asset-root",
            a.models,
        )
        stats[pair] = {}
        for shot in SHOTS:
            path = dst / (shot + ".npz")
            reference.cli(
                "render",
                layout,
                dst / "anchor.json",
                path,
                dst / shot,
                "--asset-root",
                a.models,
            )
            with np.load(path) as packet:
                row = inspect(src / shot, dst / shot, packet)
            stats[pair][shot] = row
            print(pair, shot, row, flush=True)
        missing = next(i for i, k in enumerate(pair.split("-")) if k.startswith("M"))
        if stats[pair]["side"]["matched_behind_lens"] < 200:
            raise ValueError("No independent rear-hemisphere coverage")
        if stats[pair]["ortho"]["instances"][missing]["verified_floor_pixels"] < 1000:
            raise ValueError("Missing independent aperture floor coverage")
    if a.command == "check":
        check_contract()
        save_json(a.out / "check.json", stats)
        save_json(a.out / "bearing.json", check_bearing(a.models))


if __name__ == "__main__":
    main()
