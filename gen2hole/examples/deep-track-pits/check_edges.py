"""Probe the exported contextual faces against held-out high returns and plot the joint scan."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import open3d as o3d
import trimesh
from geometry import scene
from sections import Section, main_component


def check(run, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), constrained_layout=True)
    for rec, axes_row in zip(
        json.loads(Path(__file__).with_name("survey.json").read_text())["states"], axes
    ):
        key = rec["id"]
        dest = run / "series/assets/pothole" / key
        data = np.load(run / "evaluation" / key / "survey.npz")
        m = json.loads((dest / "metadata.json").read_text())
        p = (data["xyz"] - m["source_origin_xyz"]) @ m["source_local_basis_columns"]
        lab = data["defect"]
        train = ~data["reserved"]
        primary = lab.copy()
        primary[lab] = main_component(p[lab, :2])
        dims = sorted(cv2.minAreaRect(p[primary, :2].astype("f4"))[1], reverse=True)
        row = dict(
            main_label_extent_m=dims,
            main_label_points=int(primary.sum()),
            disconnected_labels_in_source=int((lab & ~primary).sum()),
        )
        ax, profile = axes_row
        ax.scatter(*p[~lab, :2].T, s=1, c="#bfc3c3", rasterized=True)
        ax.scatter(*p[primary, :2].T, s=2, c=p[primary, 2], cmap="viridis", vmin=-0.16, vmax=0)
        profile.scatter(p[~lab, 0], p[~lab, 2], s=1, c="#bfc3c3")
        profile.scatter(
            p[primary, 0], p[primary, 2], s=2, c=p[primary, 2], cmap="viridis", vmin=-0.16, vmax=0
        )
        kind = rec["section_parameters"][0]
        if kind != "bowl":
            section = Section(p[train], lab[train], kind)
            station, offset = section.coordinates(p[:, :2])
            if kind == "straight":
                offset = offset - np.median(offset[train & ~lab & (p[:, 2] > 0.10)])
            tested = ~train & ~lab & (p[:, 2] > 0.006) & (abs(offset) < 0.018)
            obj = trimesh.load(dest / "raised-edge.glb", force="mesh", process=False)
            v = np.asarray(obj.vertices)[:, [0, 2, 1]] * [1, -1, 1]
            f = np.asarray(obj.faces)
            distance = (
                scene(v, f).compute_distance(o3d.core.Tensor(p[tested].astype("f4"))).numpy() * 1000
            )
            assert len(distance) >= 20 and np.percentile(distance, 90) < 35
            row.update(
                context_test_returns=len(distance),
                context_closest_p90_mm=float(np.percentile(distance, 90)),
                context_closest_median_mm=float(np.median(distance)),
                context_triangles=len(f),
            )
            for a, j in [(ax, 1), (profile, 2)]:
                a.add_collection(
                    PolyCollection(
                        v[f][:, :, [0, j]], facecolors="#b96c36", alpha=0.45, edgecolors="none"
                    )
                )
                a.scatter(
                    p[tested, 0],
                    p[tested, j],
                    s=4,
                    facecolors="none",
                    edgecolors="#5c351f",
                    linewidths=0.45,
                )
        ax.set(title=key + " | annotated plan and raised edge", ylabel="across (m)", aspect="equal")
        profile.set(title=key + " | longitudinal elevation", ylabel="height (m)", aspect="equal")
        for a in axes_row:
            a.set_xlabel("along (m)")
            a.grid(alpha=0.16)
        rows[key] = row
    fig.savefig(output / "capture.jpg", dpi=140)
    plt.close(fig)
    (output / "edges.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output"):
        parser.add_argument(name, type=Path)
    check(**vars(parser.parse_args()))
