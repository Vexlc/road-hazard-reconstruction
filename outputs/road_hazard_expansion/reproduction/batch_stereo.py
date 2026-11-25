"""Reconstruct stereo capture frames and keep only geometrically supported surfaces."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_pothole import reconstruct

REPOSITORY = "https://github.com/ruirangerfan/stereo_pothole_datasets"
REVISION = "a6cc8e7354da4503b0ee03d15058df131a645195"
ALREADY_PUBLISHED = {(1, 2), (2, 25)}
EXCLUDED_BY_SOURCE = {6, 7, 8, 13, 14, 15, 21, 22, 23, 37, 38, 39}


def build(job):
    source, output, samples, dataset, frame, force = job
    name = f"stereo_pothole_fan_{dataset}_{frame:02d}"
    dest = output / name
    sample = samples / f"{name}.npz"
    report_path = dest / "metadata.json"
    if not force and report_path.exists() and sample.exists():
        return json.loads(report_path.read_text())
    reasons = []
    for border in (.065, .04, .026):
        try:
            with redirect_stdout(StringIO()):
                reconstruct(SimpleNamespace(
                    rgb=source / "rgb" / f"{frame:02d}.png",
                    mask=source / "label" / f"{frame:02d}.png",
                    xyz=source / "ptcloud" / f"{frame:02d}.mat",
                    xyz_key="xyzPoints", metres_per_unit=.001, border=border,
                    output=dest, samples=sample,
                    source=f"{REPOSITORY}/tree/{REVISION}/dataset{dataset} ({frame:02d})"
                ), fit_around_primary=True)
            detail = json.loads((dest / "capture_metadata.json").read_text())
            d50, depth, _ = detail["depth_mm_p50_p95_p99_in_mask"]
            w, l = detail["pothole_extent_xy_m_p0_4_to_p99_6"]
            with np.load(dest / "depth_diagnostics.npz") as data:
                z = data["height"][data["projected_mask"]]
            tests = {
                "inside_coverage": detail["stereo_near_sample_coverage_in_mask"] > .86,
                "patch_coverage": detail["stereo_near_sample_coverage_patch"] > .64,
                "positive_depression": 3.0 < depth < 165 and d50 > -.4 and bool(np.mean(z < -.0006) > .49),
                "plane_fit": detail["road_fit_rms_mm"] < 5.0,
                "footprint_size": min(w, l) > .022 and max(w, l) < 2.0
            }
            if all(tests.values()):
                result = {
                    "id": name, "category": "stereo", "family": f"Fan {dataset}/{frame:02d}",
                    "material": "captured_rgb", "source": detail["source"], "source_dataset": dataset,
                    "source_frame": frame, "units": "metres", "geometry_source": "stereo_xyz",
                    "width_m": w, "length_m": l, "depth_p95_m": depth / 1000,
                    "metric_border_m": border, "camera_alignment_rms_px": detail["reprojection_rms_pixels"],
                    "measured_support_in_mask": detail["stereo_near_sample_coverage_in_mask"],
                    "author_excluded_from_evaluation": dataset == 2 and frame in EXCLUDED_BY_SOURCE,
                    "mesh_name": "pothole.glb", "quality_checks": tests,
                    "limitations": ["Frames may depict the same physical defect.", "A height field interpolates sparse areas and cannot represent overhangs."]
                }
                report_path.write_text(json.dumps(result, indent=2) + "\n")
                return result
            reasons.append({str(border): [k for k, passed in tests.items() if not passed]})
        except (ValueError, AssertionError) as error:
            reasons.append({str(border): str(error)})
    return {"id": name, "category": "stereo", "source_dataset": dataset, "source_frame": frame,
            "rejected": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Clone of the Fan stereo dataset.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset / f"dataset{d}", args.out, args.samples, d, i, args.force)
            for d, n in enumerate((22, 40, 5), 1) for i in range(1, n+1) if (d, i) not in ALREADY_PUBLISHED]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(build, jobs):
            results.append(result)
            print(result["id"], "reject" if "rejected" in result else "OK", flush=True)
    (args.out / "stereo_screening.json").write_text(json.dumps(results, indent=2) + "\n")
    print("STEREO", sum("rejected" not in r for r in results), "accepted of", len(results))


if __name__ == "__main__":
    main()
