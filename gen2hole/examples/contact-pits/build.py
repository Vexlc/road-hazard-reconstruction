"""Reconstruct a bounded road patch while retaining measured, unlabelled low pavement."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree, Delaunay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage/points"))
from build import build, grid_faces
from scan import local_height


def reconstruct_capture(model, labels, record):
    points = model["points"]
    # Keep low apron returns even where annotators ended the class-1 zone.
    use = ~model["hold"] & (labels | (points[:, 2] <= 0.06))
    cloud = points[use]
    xy = model["xy"].reshape(-1, 2)
    k = model["fit"]["neighbours"]
    dist, ids = cKDTree(cloud[:, :2]).query(xy, k=k)
    hull = Delaunay(cloud[:, :2])
    simplex = hull.find_simplex(xy)
    # A long triangle across a gap is insufficient evidence of a continuous road.
    vertices = cloud[hull.simplices, :2]
    longest = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2).max(axis=1)
    radius = model["limit"]
    supported = (simplex >= 0) & (longest[simplex] <= 2 * radius) & (dist[:, 0] < 0.83 * radius)
    faces = grid_faces(*model["z"].shape)
    use_faces = supported[faces].all(axis=1)
    covered = np.zeros(len(xy), bool)
    covered[faces[use_faces]] = True
    height, _ = local_height(cloud, xy[covered], k)
    neighbours = cloud[ids[covered], 2]
    z = np.zeros(len(xy), np.float32)
    z[covered] = np.clip(height, neighbours.min(axis=1), neighbours.max(axis=1))
    shape = model["z"].shape
    model.update(z=z.reshape(shape), support=covered.reshape(shape), supported_faces=use_faces)
    model["mask"] &= model["support"]
    model["capture_reconstruction"] = dict(
        grid_pitch_m=record["pitch_m"],
        max_delaunay_edge_m=2 * radius,
        unlabelled_high_returns_removed=int((~model["hold"] & ~use).sum()),
        local_range_clipped_vertices=int(
            (height != np.clip(height, neighbours.min(1), neighbours.max(1))).sum()
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "materials", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()), survey=HERE / "survey.json", refine=reconstruct_capture)
