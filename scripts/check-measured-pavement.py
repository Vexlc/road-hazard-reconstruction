"""Verify source depth holdouts, paired physical views, and the saved native scene and its actor layers."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from pavement_layout import PavementLayout, asset_folder
from road_geometry import sha256, save_json
from importlib import import_module

cli = import_module("check-cone-cycles").cli


def load(path):
    return json.loads(path.read_text())


def audit(a):
    recipe = ROOT / "gen2hole/examples/measured-pavement/layout.json"
    model = PavementLayout(recipe, a.assets)
    soil = {}
    for row, folder in model.members:
        key = row["asset"]
        meta = load(folder / "metadata.json")
        calibration = a.assets / key / "series/calibration"
        observation = load(calibration / (key + "_capture.json"))
        test = load(a.assets / key / "evaluation/measurements.json")[key]
        hashes = {
            f: sha256(folder / f)
            for f in ("asset.glb", "asset.ply", "surface.npz", "metadata.json", "splat_material.npz")
        }
        assert test["glb_sha256"] == hashes["asset.glb"] and test["fit_rebuilt_without_reserved_ranges"]
        assert (
            test["meshed_fraction"] > 0.98
            and test["withheld_mesh"]["p95_mm"] < test["withheld_road_plane"]["p95_mm"] / 5
        )
        if a.reference_models:
            for k, h in hashes.items():
                assert sha256(asset_folder(a.reference_models, key) / k) == h
        soil[key] = dict(
            dimensions_m=[meta["width_m"], meta["length_m"]],
            depth_p95_m=meta["depth_p95_m"],
            measured_fraction=meta["measured_pit_fraction"],
            source=observation["inputs"],
            holdout=test,
            files=hashes,
        )
    if a.assets_only:
        return dict(assets=soil)
    physical = {}
    for site in ["kitti", "suburb", "audi"]:
        root = a.views / site
        anchor = load(root / "anchor.json")
        model.bind(anchor)
        records = {}
        for sensor, row in load(root / "index.json")[model.key]["views"].items():
            folder = root / model.key / sensor
            with TemporaryDirectory() as d:
                dest = Path(d)
                cli(
                    "render",
                    recipe,
                    root / "anchor.json",
                    folder / "view.npz",
                    dest,
                    "--asset-root",
                    a.assets,
                )
                for name in ("after.png", "instance.png", "part.png", "range.npy"):
                    if sha256(folder / name) != sha256(dest / name):
                        raise ValueError(f"Different {site}/{sensor}/{name}")
            before = np.asarray(Image.open(folder / "before.png"))
            after = np.asarray(Image.open(folder / "after.png"))
            owners = np.asarray(Image.open(folder / "instance.png"))
            part = np.asarray(Image.open(folder / "part.png"))
            with np.load(folder / "view.npz") as packet:
                mask = packet["protected"].all(0)
            assert not np.any((after != before).any(-1) & (mask | (owners == 0)))
            assert np.array_equal(owners > 0, part > 0) and len(row["frames_showing_all_instances"]) > 0
            counts = [int((owners == i).sum()) for i in (1, 2)]
            assert min(counts) > 40
            records[sensor] = dict(
                frame=row["frame"],
                pixels=counts,
                simultaneous_exposures=len(row["frames_showing_all_instances"]),
                replay_sha256=sha256(folder / "after.png"),
                protected_edits=0,
            )
        cap = cv2.VideoCapture(str(root / "driving.mp4"))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        assert n >= 16
        physical[site] = dict(
            cameras=records, frames=n, support=[s["anchor"]["footprint_support"] for s in anchor["instances"]]
        )
    origin = a.host / "joint.pt"
    dest = a.host / "measured-pavement" / model.key
    record = load(dest / "placement.json")
    assert record["source_sha256"] == sha256(origin) and record["checkpoint_sha256"] == sha256(
        dest / "joint.pt"
    )
    assert len(record["placements"]) == 2 and all(r["scale"] == 1 for r in record["placements"])
    import torch

    source = torch.load(origin, map_location="cpu", weights_only=False)
    result = torch.load(dest / "joint.pt", map_location="cpu", weights_only=False)
    # Audit each moving / tracked layer and its full time-dependent metadata.
    actors = 0
    for key, actor in source["metadata"]["obj_meta"].items():
        idx = int(actor["id"])
        poses = source["metadata"]["camera_tracklets"][:, :, idx]
        if actor["class"] != "untracked residual" or not np.allclose(poses, poses[0, 0]):
            layer = f"obj_{idx:03}"
            if layer in source["layers"]:
                assert all(
                    torch.equal(v, result["layers"][layer][k]) for k, v in source["layers"][layer].items()
                )
                actors += 1
    assert torch.equal(
        torch.as_tensor(source["metadata"]["camera_tracklets"]),
        torch.as_tensor(result["metadata"]["camera_tracklets"]),
    )
    replay = load(a.views / "joint" / model.key / "replay.json")
    assert all(v["frames"] == 48 and v["peak_changed_pixels"] > 100 for v in replay.values())
    assert load(a.views / "joint/original/replay.json")["front"]["max_baseline_error"] == 0
    return dict(
        assets=soil,
        physical=physical,
        cycles=load(a.views / "cycles-check/check.json"),
        appearance=load(a.texture),
        native=dict(
            placement={k: v for k, v in record.items() if k != "projected_edges"},
            persistent_tracked_layers_checked=actors,
            replay=replay,
            comparison=load(a.native_comparison),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/measured-pavement")
    p.add_argument("--host", type=Path, default=ROOT / "runs/streetgs-joint/kitti360-0000/30000")
    p.add_argument("--reference-models", type=Path)
    p.add_argument("--assets-only", action="store_true")
    p.add_argument("--texture", type=Path, default=ROOT / "outputs/measured-pavement/texture.json")
    p.add_argument(
        "--native-comparison", type=Path, default=ROOT / "outputs/measured-pavement/native-comparison.json"
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    save_json(args.out, audit(args))
    print("Verified measured pavement.", flush=True)
