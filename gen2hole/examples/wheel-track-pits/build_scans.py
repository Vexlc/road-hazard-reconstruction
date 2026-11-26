"""Build road failures only where the registered XYZ observations support a surface."""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree, Delaunay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage/points"))
from build import build, grid_faces
from scan import local_height
from steep import reconstruct_wall, texture_walls


def restrict_to_capture(m, labels, record):
    points = m["points"]
    train = ~m["hold"]
    # The surrounding road fit is robust to extraneous objects. Exclude high,
    # unlabelled returns from the actual road mesh as well.
    pavement = points[train & ~labels, 2]
    scale = np.median(abs(pavement - np.median(pavement)))
    train &= labels | (abs(points[:, 2] - np.median(pavement)) < max(0.022, 6 * scale))
    p = points[train]
    query = m["xy"].reshape(-1, 2)
    if record.get("steep_walls"):
        reconstruct_wall(m, p, labels[train])
        return
    inside = Delaunay(p[:, :2]).find_simplex(query) >= 0
    tree = cKDTree(p[:, :2])
    distance, _ = tree.query(query)
    support = inside & (distance < m["limit"] * 0.83)
    z = m["z"].ravel()
    # Preserve the depth up to a clipped endpoint. Fading to horizontal at
    # that boundary would introduce a ramp that the scan never saw.
    height, _ = local_height(p, query[support], m["fit"]["neighbours"])
    z[support] = height
    tri = grid_faces(*m["z"].shape)
    supported_faces = support[tri].all(axis=1)
    covered = np.zeros(len(z), bool)
    covered[tri[supported_faces]] = True
    m["supported_faces"] = supported_faces
    m["support"] = covered.reshape(m["z"].shape)
    m["mask"] &= m["support"]
    z[~covered] = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "materials", "out"):
        parser.add_argument(arg, type=Path)
    build(
        **vars(parser.parse_args()),
        survey=HERE / "survey.json",
        refine=restrict_to_capture,
        mesh_layout=texture_walls
    )
