"""Compare a four-ray exposure with independent native fisheyes as a foreground marker changes color."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
from PIL import Image

rig = import_module("check-cone-cycles")
from host_depth import road_samples
from road_geometry import save_json


def group(image):
    return np.stack([image[y::2, x::2] for y in range(2) for x in range(2)])


def unpack_srgb(rgb):
    v = rgb.astype(np.float64) / 255
    return np.where(v < 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def camera_response(samples, weights):
    v = (unpack_srgb(samples) * weights[..., None]).sum(0) / weights.sum(0)[..., None]
    return np.uint8(np.rint(255 * np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)))


def prepare(raw, cameras, pairs):
    for pair in pairs:
        folder = cameras / pair
        folder.mkdir(exist_ok=True, parents=True)
        points = []
        for shot in ("rear", "side"):
            source = raw / pair / "amber" / shot
            fine, world, _ = rig.packet(source / "host", False)
            points.append(road_samples(fine, stride=5))
            ids = np.load(source / "native/id.npy")
            yy, xx = np.nonzero(ids >= 32)
            y, x = np.maximum([yy.min() - 50, xx.min() - 50], 0) // 2 * 2
            bottom, right = np.minimum(np.array(ids.shape), ([yy.max() + 52, xx.max() + 52])) // 2 * 2
            box = np.s_[y:bottom, x:right]
            packed = {
                name: group(fine[name][box])
                for name in ("rays_camera", "protected", "receiver_mask", "host_range", "occluder_range")
            }
            packed["camera_to_world"] = fine["camera_to_world"]
            h, w = packed["protected"].shape[1:]
            weights = np.broadcast_to(np.array([1, 2, 4, 3], "u1")[:, None, None], (4, h, w)).copy()
            # Spatial filter with a different zero-response ray along alternate detector bands.
            col = np.arange(w)
            for i in range(4):
                weights[i, :, (col // 9) % 5 == i] = 0
            packed["sample_weights"] = weights
            ground = dict(
                native_id=group(ids[box]), native_range=group(np.load(source / "native/range.npy")[box])
            )
            for color in ("amber", "cyan"):
                src = raw / pair / color / shot
                host = np.asarray(Image.open(src / "host/rgb.png").convert("RGB"))
                ground[color] = camera_response(
                    group(np.asarray(Image.open(src / "native/rgb.png").convert("RGB"))[box]), weights
                )
                packed["sample_rgb"] = group(host[box])
                packed["rgb"] = camera_response(packed["sample_rgb"], weights)
                np.savez_compressed(folder / f"{shot}-{color}.npz", **packed)
            np.savez_compressed(folder / f"{shot}-truth.npz", **ground)
        np.save(folder / "road.npy", np.concatenate(points))
        save_json(
            folder / "frame.json",
            dict(seed=world[:3, 3].tolist(), heading=world[:3, 0].tolist(), up=world[:3, 2].tolist()),
        )


def check_sampling_contract():
    from camera_remap import remap_packet
    from host_exposure import exposure
    from host_depth import geometry

    shape = (2, 2, 3)
    planes = np.arange(36, dtype=np.uint8).reshape(*shape, 3) * 7
    weights = np.array([[[1, 0, 2], [2, 1, 0]], [[0, 3, 1], [2, 0, 4]]], dtype=float)
    data = dict(
        sample_rgb=planes,
        sample_weights=weights,
        rgb=camera_response(planes, weights),
        protected=np.zeros(shape, bool),
        host_z=np.full(shape, 4.0),
        rays_camera=np.broadcast_to([0.0, 0.0, 1.0], (*shape, 3)).copy(),
        camera_to_world=np.eye(4),
    )
    data["rays_camera"][0, 0, 1] = np.nan
    data["ray_origins_camera"] = np.zeros((*shape, 3))
    data["ray_origins_camera"][1, ..., 2] = 0.1
    remapped = remap_packet(data, np.array([[[1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]]]))
    colors, norm = exposure(remapped, (2, 1, 3))
    _, dist, masked, _, org = geometry(remapped, with_origins=True)
    assert np.array_equal(colors[:, 0, 0], planes[:, 0, 1]) and not colors[:, 0, 2].any()
    assert masked[:, 0, 2].all() and masked[0, 0, 0] and dist[1, 0, 0] + org[1, 0, 0, 2] == 4.0
    assert np.array_equal(norm[:, 0, 1], [0.5, 0.5])
    for w in (np.zeros(shape), np.full(shape, -1.0), np.full(shape, np.nan), np.zeros((2, 3))):
        try:
            exposure(dict(data, sample_weights=w), shape)
        except ValueError:
            continue
        raise AssertionError("Accepted invalid optical weights")
    shared = np.broadcast_to(np.array([3.0, 1.0])[:, None, None], shape)
    _, norm = exposure(dict(data, sample_weights=[3.0, 1.0], rgb=camera_response(planes, shared)), shape)
    np.testing.assert_allclose(norm.ravel(), [0.75, 0.25])
    try:
        exposure(dict(data, rgb=np.zeros_like(data["rgb"])), shape)
    except ValueError:
        return
    raise AssertionError("Accepted a different camera response")


def check(cameras, layouts, assets, out):
    check_sampling_contract()
    report = {}
    for layout in layouts:
        pair = json.loads(layout.read_text())["id"]
        source, dest = cameras / pair, out / pair
        dest.mkdir(exist_ok=True, parents=True)
        frame = json.loads((source / "frame.json").read_text())
        fit = dest / "anchor.json"
        appearance = dest / "appearance.json"
        rig.cli(
            "fit",
            source / "road.npy",
            fit,
            "--seed",
            *frame["seed"],
            "--heading",
            *frame["heading"],
            "--up",
            *frame["up"],
            "--radius",
            1.25,
            "--asset",
            layout,
            "--asset-root",
            assets,
        )
        rig.cli(
            "calibrate-pavement",
            layout,
            fit,
            appearance,
            source / "rear-amber.npz",
            source / "side-amber.npz",
            "--asset-root",
            assets,
        )
        for shot in ("rear", "side"):
            with np.load(source / f"{shot}-truth.npz") as data:
                native = dict(data)
            host, results = {}, {}
            for color in ("amber", "cyan"):
                packet = source / f"{shot}-{color}.npz"
                folder = dest / f"{shot}-{color}"
                rig.cli("render", layout, appearance, packet, folder, "--asset-root", assets)
                results[color] = np.array(Image.open(folder / "after.png"))
                with np.load(packet) as view:
                    host[color] = view["rgb"]
                    weights = view["sample_weights"]
                    occluder = np.isfinite(view["occluder_range"])
                    rays = view["rays_camera"]
            ids = native["native_id"]
            visible = (ids >= 32) & (weights > 0)
            near = np.where(visible, native["native_range"], np.inf)
            first = near.argmin(0)[None]
            expected = np.where(visible.any(0), np.take_along_axis(ids, first, 0)[0], 0)
            detected = np.array(Image.open(folder / "instance.png"))
            parts = np.array(Image.open(folder / "part.png"))
            distances = np.load(folder / "range.npy")
            # Fractional foreground coverage with at least one positive-weight native asset ray.
            mixed = (occluder & (weights > 0)).any(0) & visible.any(0)
            for name in ("instance.png", "part.png", "range.npy"):
                if (dest / f"{shot}-amber" / name).read_bytes() != (dest / f"{shot}-cyan" / name).read_bytes():
                    raise ValueError("Changing the marker color altered a geometry pass")
            full_bar = (occluder | (weights == 0)).all(0)
            if full_bar.sum() < 30 or detected[full_bar].any():
                raise ValueError("Missing fully occluded pixels or overwritten foreground geometry")
            for color in host:
                if not np.array_equal(results[color][full_bar], host[color][full_bar]):
                    raise ValueError("Changed fully covered marker pixels")
            native_delta = unpack_srgb(native["cyan"]) - unpack_srgb(native["amber"])
            actual = unpack_srgb(results["cyan"]) - unpack_srgb(results["amber"])
            # Giving each surviving host ray the same already integrated RGB counts the foreground coverage twice.
            unchanged_fraction = ((~visible) * weights).sum(0) / weights.sum(0)
            averaged = (unpack_srgb(host["cyan"]) - unpack_srgb(host["amber"])) * unchanged_fraction[
                ..., None
            ]
            stats = {}
            for index, key in enumerate(pair.split("-"), 1):
                truth, hit = expected // 32 == index, detected == index
                same = truth & hit & (parts == expected % 32)
                edge = mixed & truth
                if edge.sum() < 12:
                    raise ValueError(f"Insufficient partial foreground in {pair}/{shot}/{key}: {edge.sum()}")
                stats[key] = dict(
                    iou=float((truth & hit).sum() / (truth | hit).sum()),
                    part_agreement=float(same.sum() / truth.sum()),
                    range_error_p95_m=float(np.percentile(abs(distances[same] - near.min(0)[same]), 95)),
                    mixed_pixels=int(edge.sum()),
                    sampled_mae=float(abs(actual - native_delta)[edge].mean()),
                    resolved_host_mae=float(abs(averaged - native_delta)[edge].mean()),
                    rear_hemisphere_samples=int(
                        ((ids // 32 == index) & (rays[..., 2] < 0) & (weights > 0)).sum()
                    ),
                )
                s = stats[key]
                print(pair, shot, key, s, flush=True)
                if (
                    s["iou"] < 0.965
                    or s["part_agreement"] < 0.945
                    or s["range_error_p95_m"] > 0.004
                    or s["sampled_mae"] > 0.019
                    or s["sampled_mae"] >= s["resolved_host_mae"] * 0.35
                ):
                    raise ValueError("Native pass or foreground radiance check failed")
            report[f"{pair}/{shot}"] = stats
    save_json(out / "check.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("layouts", nargs="+", type=Path)
    parser.add_argument("--from-cycles", type=Path)
    args = parser.parse_args()
    if args.from_cycles:
        prepare(args.from_cycles, args.cameras, [json.loads(p.read_text())["id"] for p in args.layouts])
    check(args.cameras, args.layouts, args.assets, args.out)
