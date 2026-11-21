"""Check held-out camera isolation, raw RGB-D file import, and rejection of a different pit."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole"))
from pit_captures import features, read_views, registration


def read(path):
    return json.loads(path.read_text())


def capture(spec):
    return read_views(spec, ROOT / "work/reconstruction-inputs", ROOT / "catalog/capture-inputs.json", ROOT)


def check_isolation(spec):
    rows = {}
    for key, view in capture(spec).items():
        if key not in spec["holdout"]:
            continue
        rgb = view.rgb.copy()
        depth = view.depth.copy()
        rgb[view.mask] = np.random.default_rng(704).integers(0, 256, (view.mask.sum(), 3), dtype=np.uint8)
        depth[view.mask] = 0
        original = features(view)
        modified = features(replace(view, rgb=rgb, depth=depth))
        if not all(np.array_equal(a, b) for a, b in zip(original, modified)):
            raise ValueError(key + " held-out interior affects road features")
        rows[key] = dict(
            overwritten_pixels=int(view.mask.sum()), identical_road_descriptors=len(original[0])
        )
    return rows


def export_files(spec, folder):
    frames = {}
    for key, view in capture(spec).items():
        rgb, depth, mask = (key + suffix for suffix in (".png", "-z.npy", "-mask.png"))
        Image.fromarray(view.rgb).save(folder / rgb)
        Image.fromarray(view.mask.astype("uint8") * 255).save(folder / mask)
        np.save(folder / depth, np.rint(view.depth * 1000).astype("uint16"))
        k = view.K
        frames[key] = dict(
            rgb=rgb,
            depth=depth,
            mask=mask,
            depth_scale=0.001,
            fx_fy_cx_cy=[k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
        )
    return dict(spec, source="files", frames=frames)


def build_mesh(spec, folder):
    survey = folder / "survey.json"
    survey.write_text(json.dumps([spec]))
    out = folder / "result"
    process = subprocess.run(
        [
            sys.executable,
            ROOT / "gen2hole/multiview_pits.py",
            "--survey",
            survey,
            "--out",
            out,
            "--mesh-only",
        ],
        capture_output=True,
        text=True,
    )
    return process, out


def check(run, tmp):
    specs = read(run / "series/calibration/survey.json")
    cases = {spec["id"]: check_isolation(spec) for spec in specs}
    source = next(spec for spec in specs if spec["id"] == "P112")
    custom = tmp / "files"
    custom.mkdir()
    task, out = build_mesh(export_files(source, custom), custom)
    if task.returncode:
        raise RuntimeError(task.stderr)
    original = read(run / "evaluation/measurements.json")["P112"]
    actual = read(out / "evaluation/measurements.json")["P112"]
    if original["glb_sha256"] != actual["glb_sha256"] or original["heldout"] != actual["heldout"]:
        raise ValueError("File import changed P112 geometry, texture or measurements")

    negative = dict(
        id="P115",
        label="异坑对照",
        source="PothRGBD",
        anchor="20250305_052438",
        fit=["20250305_052438", "20250305_052530"],
        holdout=["20250305_052434"],
    )
    _, alignment = registration(negative, capture(negative))
    other = tmp / "other-pit"
    other.mkdir()
    task, out = build_mesh(negative, other)
    if (
        not task.returncode
        or "No resolved overlap with the withheld target" not in task.stderr
        or out.exists()
    ):
        raise ValueError("A different target was accepted or failed for an unrelated reason")
    return dict(
        heldout_interior_replaced=cases,
        normalized_files=dict(
            id=source["id"], glb_sha256=actual["glb_sha256"], heldout_metrics_identical=True
        ),
        distinct_pit=dict(
            fit=negative["fit"],
            holdout=negative["holdout"],
            road_inliers=alignment[negative["holdout"][0]]["inliers"],
            rejected_for_missing_target_overlap=True,
            destination_absent=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(2)
    with TemporaryDirectory() as tmp:
        report = check(args.run, Path(tmp))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
