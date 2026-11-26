"""Evaluate candidate surfaces using three spatial folds, excluding the final holdout."""

import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
from geometry import triangles, scene, hits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage/points"))
from scan import frame


def evaluate(run, output):
    rows = {}
    for rec in json.loads((HERE / "survey.json").read_text())["states"]:
        key = rec["id"]
        kind = rec["section_parameters"][0]
        data = np.load(run / "evaluation" / key / "survey.npz")
        xyz, labels, reserved = data["xyz"], data["defect"], data["reserved"]
        block = np.floor(xyz[:, [0, 2]] / 0.028).astype(int)
        fold = (block[:, 0] + 5 * block[:, 1]) % 11
        assert np.array_equal(fold == 0, reserved)
        rows[key] = []
        heights = [1.0] if kind == "bowl" else [0.11, 0.20, 0.34]
        for h, k, blend in itertools.product(heights, [8, 16], [0.0, 0.65, 1.0]):
            if blend == 0 and k != 8:
                continue
            error = []
            closest = []
            miss = 0
            for ix in [1, 4, 8]:
                train = ~reserved & (fold != ix)
                test = (fold == ix) & labels
                origin, basis = frame(xyz, labels, train)
                points = (xyz - origin) @ basis
                low = points[train & labels, :2].min(0)
                high = points[train & labels, :2].max(0)
                points[:, :2] -= (low + high) / 2
                v, f, _, _ = triangles(
                    points[train], labels[train], high - low + 0.4, kind, h, k, blend
                )
                mesh = scene(v, f)
                z, _ = hits(mesh, points[test, :2])
                valid = np.isfinite(z)
                miss += int((~valid).sum())
                error.extend(abs(z[valid] - points[test, 2][valid]) * 1000)
                closest.extend(
                    mesh.compute_distance(o3d.core.Tensor(points[test].astype("f4"))).numpy() * 1000
                )
            rows[key].append(
                dict(
                    h_m=h,
                    k=k,
                    blend=blend,
                    vertical_mae_mm=np.mean(error),
                    vertical_p90_mm=np.percentile(error, 90),
                    closest_p90_mm=np.percentile(closest, 90),
                    missing=miss,
                    hits=len(error),
                )
            )
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output"):
        parser.add_argument(name, type=Path)
    evaluate(**vars(parser.parse_args()))
