"""Check moving rear fisheye readout independently against Cycles part and range passes."""

import argparse
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

rig = import_module("check-cone-cycles")
from host_depth import geometry, road_samples
from pavement_layout import PavementLayout
from pavement_support import PavementSupport
from road_geometry import save_json

ROOT = rig.ROOT
PAIRS = ("A248-M249", "M248-A249")


def assemble(source, reverse):
    """Read observed rows from eight separately rendered host/native pairs."""
    labels, depth, rows, fields = [], [], [], {}
    for index in range(8):
        slot = 7 - index if reverse else index
        folder = source / f"readout-{slot}"
        frame, world, _ = rig.packet(folder / "host", axial=not reverse)
        size = len(frame["rgb"])
        band = slice(index * size // 8, (index + 1) * size // 8)
        for k, v in frame.items():
            if k != "camera_to_world":
                fields.setdefault(k, []).append(v[band])
        n = band.stop - band.start
        rows.extend([frame["camera_to_world"]] * n)
        labels.append(np.load(folder / "native/id.npy")[band])
        depth.append(np.load(folder / "native/range.npy")[band])
    view = {k: np.concatenate(v) for k, v in fields.items()}
    row_poses = np.stack(rows)
    reference_pose = row_poses[(len(rows) - 1) // 2]
    view.update(camera_to_world=reference_pose, camera_to_world_rows=row_poses)
    return view, np.concatenate(labels), np.concatenate(depth), world


def scores(dest, ids, native_depth, keys):
    parts = np.asarray(Image.open(dest / "part.png"))
    hits = np.asarray(Image.open(dest / "instance.png"))
    depth = np.load(dest / "range.npy")
    result = {}
    for i, key in enumerate(keys, 1):
        expected, hit = ids // 32 == i, hits == i
        same = expected & hit & (parts == ids % 32)
        if not same.any():
            raise ValueError(f"No correctly registered pixels for {key}")
        result[key] = {
            "native_pixels": int(expected.sum()),
            "instance_iou": float((expected & hit).sum() / (expected | hit).sum()),
            "part_agreement": float(same.sum() / expected.sum()),
            "p95_range_error_m": float(np.percentile(abs(depth[same] - native_depth[same]), 95)),
            "parts": {str(p): int((same & (parts == p)).sum()) for p in np.unique(parts[same])},
        }
    return result


def compare_origins(frame):
    """Unproject the same physical road points from translated, per-subray optical origins."""
    single = road_samples(frame, stride=8)
    rays = frame["rays_camera"]
    shifted = dict(frame)
    shifts = np.array([0.031, 0.047], dtype="f4")[:, None, None]
    shifted["rays_camera"] = np.stack([rays, rays])
    shifted["ray_origins_camera"] = shifts[..., None] * rays
    shifted["occluder_range"] = frame["occluder_range"] - shifts
    if "host_range" in frame:
        shifted["host_range"] = frame["host_range"] - shifts
    shifted["camera_to_world_rows"] = np.stack([frame["camera_to_world_rows"]] * 2)
    both = road_samples(shifted, stride=8).reshape(2, *single.shape)
    return float(abs(both - single).max())


def check_support(packet, native_ids, world, anchor, layout, radii):
    """Compare the localization support with native ID pixels and distant intact road."""
    rays, depth, protected, pose, origins = geometry(packet, with_origins=True)
    direction = rays[0] @ pose[:3, :3].T
    starts = origins[0] @ pose[:3, :3].T + pose[:3, 3]
    hit = np.isfinite(depth[0])
    contact = starts + direction * np.where(hit, depth[0], 0.0)[..., None]
    distant = (native_ids == 1) & hit & ~protected[0]
    for seat in anchor["instances"]:
        xy = (contact - seat["anchor"]["center_world"]) @ world[:3, :2]
        distant &= np.linalg.norm(xy, axis=-1) > radii[seat["asset"]] + 0.25
    # Sample distant background, but check every independently labelled asset pixel.
    grid = np.zeros_like(distant)
    grid[::7, ::7] = True
    native = native_ids >= 32
    sample = native | (distant & grid)
    weights = []
    for seat in anchor["instances"]:
        gate = PavementSupport(layout.meshes[seat["asset"]], 0.18)
        weights.append(gate.weights(starts[sample], direction[sample], seat["anchor"]))
    w = np.maximum.reduce(weights)
    inside = native[sample]
    if not inside.any() or (~inside).sum() < 100 or np.any(w[inside] != 1) or np.any(w[~inside] != 0):
        raise ValueError("Localized changes cut into native geometry or leak onto distant pavement")
    return dict(native_asset_pixels=int(inside.sum()), unchanged_distant_road=int((~inside).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/rolling-covers")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/rolling-covers")
    args = p.parse_args()
    results = {}
    for pair in PAIRS:
        source = args.views / "cycles" / pair
        dest = args.views / "oracle" / pair
        recipe = ROOT / "gen2hole/examples/rolling-covers" / (pair + ".json")
        fit = dest / "anchor.json"
        appearance = dest / "appearance.json"
        layout = PavementLayout(recipe, args.assets)
        anchor = json.loads(fit.read_text())
        layout.bind(anchor)
        radii = {}
        for path in recipe.parent.glob("*/survey.json"):
            survey = json.loads(path.read_text())
            profile = survey["perforations"]
            for state in survey["states"]:
                radii[state["id"]] = profile["size_m"][0] / 2 + profile["frame_band_m"]
        rig.cli(
            "calibrate-pavement", recipe, fit, appearance, dest / "ortho.npz", "--asset-root", args.assets
        )
        results[pair] = {}
        for name in ("top-down", "bottom-up"):
            packet, native_ids, native_depth, world = assemble(source, reverse=name == "bottom-up")
            path = dest / (name + ".npz")
            reverse = name == "bottom-up"
            measured = packet.pop("camera_to_world_rows")
            # Use nanoseconds around a real-scale epoch; each band has its own exposure midpoint.
            times = np.int64(1_765_000_000_000_000_000) + (
                np.arange(8, dtype="int64") * 3_000_000 + 1_500_000
            )
            band = np.arange(len(measured)) * 8 // len(measured)
            packet["row_time_ns"] = times[(7 - band) if reverse else band]
            raw = dest / (name + "-timed.npz")
            np.savez_compressed(raw, **packet)
            trajectory = dest / "trajectory.npz"
            if not reverse:
                np.savez_compressed(trajectory, time_ns=times[[0, -1]], camera_to_world=measured[[0, -1]])
            rig.cli("register-readout", raw, trajectory, path)
            with np.load(path) as calibrated:
                packet = dict(calibrated)
            pose_error = float(abs(packet["camera_to_world_rows"] - measured).max())
            if pose_error > 1e-10:
                raise ValueError("Interpolated readout disagrees with the eight native exposures")

            def render(src, target):
                rig.cli("render", recipe, appearance, src, target, "--asset-root", args.assets)
                return scores(target, native_ids, native_depth, pair.split("-"))

            result = render(path, dest / name)
            with TemporaryDirectory() as tmp:
                folder = Path(tmp)
                fixed = {
                    k: v for k, v in packet.items() if k not in ("camera_to_world_rows", "row_time_ns")
                }
                path = folder / "single.npz"
                np.savez_compressed(path, **fixed)
                baseline = render(path, folder / "single")
            blocked = packet["protected"] | np.isfinite(packet["occluder_range"])
            img = np.array(Image.open(dest / name / "after.png"))
            hits = np.array(Image.open(dest / name / "instance.png"))
            assert not hits[blocked].any() and np.array_equal(packet["rgb"][blocked], img[blocked])
            pts = road_samples(packet, stride=4)
            residual = np.abs((pts - world[:3, 3]) @ world[:3, 2])
            p95 = float(np.percentile(residual, 95))
            fixed_pts = road_samples(fixed, stride=4)
            fixed_p95 = float(np.percentile(abs((fixed_pts - world[:3, 3]) @ world[:3, 2]), 95))
            subpixel = compare_origins(packet)
            record = dict(
                support_check=check_support(packet, native_ids, world, anchor, layout, radii),
                max_pose_entry_error=pose_error,
                registered=result,
                single_pose=baseline,
                road_p95_m=p95,
                single_pose_road_p95_m=fixed_p95,
                subray_origin_max_error_m=subpixel,
                protected_pixels=int(blocked.sum()),
            )
            print(pair, name, json.dumps(record), flush=True)
            if p95 > 0.001 or subpixel > 0.0001:
                raise ValueError("Metric road reconstruction disagrees with the native world plane")
            for key, row in result.items():
                if (
                    row["instance_iou"] < 0.965
                    or row["part_agreement"] < 0.95
                    or row["p95_range_error_m"] > 0.003
                ):
                    raise ValueError(f"Unverified moving readout for {key} in {name}")
                if row["part_agreement"] <= baseline[key]["part_agreement"]:
                    raise ValueError(
                        "This fixture did not distinguish the calibrated and single-pose cameras"
                    )
            results[pair][name] = record
    save_json(args.views / "readout-check.json", results)


if __name__ == "__main__":
    main()
