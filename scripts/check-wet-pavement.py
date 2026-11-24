"""Verify measured water budgets, surveyed shoreline depth and independently rendered reflectance."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
from PIL import Image

rig = import_module("check-cone-cycles")
from pavement_layout import PavementLayout, asset_folder
from pavement_water import DirectionalProbe, interface
from prop_lighting import srgb
from road_geometry import save_json

ROOT = rig.ROOT
SRC = ROOT / "gen2hole/examples/wet-pavement"
PAIRS = ("P252-P255", "P254-P253")


def check_swatches(path):
    native = json.loads(path.read_text())
    direction = np.array([row["incident"] for row in native])
    sky = np.array([row["sky_linear"] for row in native])
    observed = np.array([row["water_linear"] for row in native])
    mirror, transmitted, fraction = interface(direction, np.array([0.0, 0.0, 1.0]))
    probe = DirectionalProbe(np.array([row["sky_direction"] for row in native]), srgb(sky), 2.0)
    sampled, support = probe.sample(mirror)
    _, unseen = probe.sample(-mirror)
    err = np.max(abs(observed / sky - fraction[:, None]), axis=1)
    radiance = np.max(abs(sampled * fraction[:, None] - observed), axis=1)
    if (
        err.max() > 0.002
        or radiance.max() > 0.004
        or min(support) < 0.9999
        or unseen.any()
        or np.max(abs(np.linalg.norm(transmitted, axis=1) - 1)) > 1e-8
        or np.ptp(sky[:, 0]) < 0.3
    ):
        raise ValueError("Water response disagrees with native glass or unobserved light leaked in")
    return [
        dict(angle_deg=r["angle_deg"], fresnel=float(f), native_error=float(e), rgb_linear_error=float(c))
        for r, f, e, c in zip(native, fraction, err, radiance)
    ]


def audit_pool(folder, seat, up):
    record = import_module("review-photo-puddle").verify_mesh(folder)
    from road_geometry import glb_triangles

    basis = np.asarray(seat["anchor"]["basis_columns"])
    normal = np.asarray(up)
    ranges = [
        xyz @ basis.T @ normal
        for name, xyz, _ in glb_triangles(folder / "asset.glb", names=True)
        if name == "Water surface"
    ]
    height = np.concatenate(ranges)
    record["world_elevation_span_m"] = float(np.ptp(height))
    if record["world_elevation_span_m"] > 2e-6:
        raise ValueError("The seated water surface is tilted in the host world")
    return record


def crown(out, models):
    layout = SRC / "crown-wet.json"
    placed = models / "crown"
    paths = []
    oracle = out / "oracle"
    oracle.mkdir(parents=True, exist_ok=True)
    for name in ("center", "edge", "rear"):
        raw, _, _ = rig.packet(out / "crown/wet" / name / "host", axial=name == "edge", flat_reference=False)
        rays = raw["rays_camera"] @ raw["camera_to_world"][:3, :3].T
        mask = (~raw["protected"]) & (rays[..., 2] > 0) & np.isfinite(rays).all(-1)
        subs = np.zeros_like(mask)
        subs[::5, ::5] = True
        keep = mask & subs
        if keep.any():
            raw.update(DirectionalProbe(rays[keep], raw["rgb"][keep], 2.0).packet())
        path = oracle / (name + ".npz")
        np.savez_compressed(path, **raw)
        paths.append(path)
    rig.cli(
        "calibrate-pavement",
        layout,
        placed / "anchor.json",
        oracle / "appearance.json",
        *paths[:2],
        "--asset-root",
        placed,
    )
    summary = {}
    total = 0
    for path in paths:
        name = path.stem
        dest = oracle / name
        rig.cli("render", layout, oracle / "appearance.json", path, dest, "--asset-root", placed)
        ids = np.load(out / "crown/wet" / name / "native/id.npy")
        water = (ids == 16) | (ids == 17)
        ref = np.load(out / "crown/wet" / name / "native/range.npy")
        parts = np.array(Image.open(dest / "part.png"))
        predicted = parts == 4
        occ = np.load(out / "crown/wet" / name / "host/id.npy") == 2
        if np.any(parts[occ]):
            raise ValueError("Foreground occluder was edited")
        instance = np.asarray(Image.open(dest / "instance.png"))
        agreement = (water & predicted) & (instance == ids.astype(int) - 15)
        if not agreement.any():
            raise ValueError("View did not resolve water")
        iou = float(agreement.sum() / (water | predicted).sum())
        depth = np.load(dest / "range.npy")
        err = float(np.percentile(abs(ref[agreement] - depth[agreement]), 95))
        if iou < 0.96 or err > 0.002:
            raise ValueError(f"Wrong native waterline or range {name}: {iou}, {err}")
        total += int(agreement.sum())
        summary[name] = dict(
            native_pixels=int(water.sum()),
            part_instance_iou=iou,
            p95_range_error_m=err,
            by_id={str(k): int((ids == k).sum()) for k in (16, 17)},
            occluder_pixels=int(occ.sum()),
        )
    if total < 200:
        raise ValueError("Insufficient wet pixels in the three control cameras")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/wet-pavement")
    p.add_argument("--models", type=Path, default=ROOT / "runs/wet-pavement")
    a = p.parse_args()
    result = dict(
        interface=check_swatches(a.views / "interface/native.json"),
        crown=crown(a.views, a.models),
        states={},
    )
    for site, pair in [(s, p) for s in ("kitti", "audi") for p in PAIRS] + [("crown", "crown-wet")]:
        models = a.models / site if site == "crown" else a.models / site / pair
        anchor = json.loads((models / "anchor.json").read_text())
        layout = PavementLayout(SRC / (pair + ".json"), models)
        layout.bind(anchor)
        for seat in anchor["instances"]:
            if seat["id"] not in layout.water_states:
                continue
            folder = asset_folder(models, seat["asset"])
            result["states"][site + "/" + seat["asset"]] = audit_pool(
                folder, seat, anchor["gravity_up_world"]
            )
        wrong = dict(anchor, gravity_up_world=(-np.asarray(anchor["gravity_up_world"])).tolist())
        try:
            layout.bind(wrong)
        except ValueError as exc:
            if "Re-solve" not in str(exc):
                raise
        else:
            raise ValueError("Accepted a fitted water state with inverted gravity")
    save_json(a.views / "verification.json", result)


if __name__ == "__main__":
    main()
