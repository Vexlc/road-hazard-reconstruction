"""Build native scan meshes and detachable raised context in the same GLB coordinate frame."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage/points"))
sys.path.insert(0, str(HERE.parent / "wheel-track-pits"))
from build import build
from steep import texture_walls
from geometry import hits, scene, triangles
from sections import main_component


def edge_mesh(out, points, lab, section):
    import trimesh

    station, offset = section.coordinates(points[:, :2])
    if section.kind == "straight":
        offset = offset - np.median(offset[(points[:, 2] > 0.10) & ~lab])
    use = ~lab & (points[:, 2] > 0.006) & (abs(offset) < 0.018)
    vertices = points[use]
    faces = Delaunay(np.c_[station[use], vertices[:, 2]]).simplices
    tri = vertices[faces]
    faces = faces[(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2) < 0.11).all(1)]
    # glTF uses Y up. Match the axes used by the shared Mesh export.
    vertices = vertices[:, [0, 2, 1]] * [1, 1, -1]
    shell = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    shell.remove_unreferenced_vertices()
    shell.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="Unclassified raised surface",
            baseColorFactor=[118, 115, 108, 255],
            doubleSided=True,
            metallicFactor=0,
            roughnessFactor=0.9,
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shell.export(out)
    return dict(
        triangles=len(faces),
        selected_train_returns=int(use.sum()),
        median_offset_mm=float(np.median(abs(offset[use])) * 1000),
        raised_edge_radius_m=float(section.radius) if section.kind == "curved" else None,
    )


def reconstruct(archive, materials, out):
    def refine(model, labels, record):
        train = ~model["hold"]
        p = model["points"][train].copy()
        lab = labels[train]
        main = p[lab][main_component(p[lab, :2])]
        lo, hi = main[:, :2].min(0), main[:, :2].max(0)
        shift = (lo + hi) / 2
        p[:, :2] -= shift
        model["origin"] += model["basis"][:, :2] @ shift
        pitch = record["pitch_m"]
        radius = np.ceil(((hi - lo) / 2 + record["apron_m"]) / pitch).astype(int)
        x, y = np.meshgrid(
            np.arange(-radius[0], radius[0] + 1) * pitch,
            np.arange(radius[1], -radius[1] - 1, -1) * pitch,
        )
        xy = np.stack([x, y], -1)
        kind, h, k, blend = record["section_parameters"]
        v, f, bed, sec = triangles(p, lab, 2 * radius * pitch, kind, h, k, blend)
        z, face = hits(scene(v, f), xy)
        support = np.isfinite(z)
        mask = np.zeros_like(support)
        mask[support] = bed[face[support]]
        capture = dict(
            method="Training-point TIN in local section coordinates",
            parameters=record["section_parameters"],
            first_hit_grid_pitch_m=pitch,
            triangles=len(f),
            main_training_labels=len(main),
            ignored_disconnected_labels=int(lab.sum() - len(main)),
            acquisition_boundary=record["boundary"],
        )
        if sec is not None:
            capture["raised_context"] = edge_mesh(
                out / "series/assets/pothole" / record["id"] / "raised-edge.glb", p, lab, sec
            )
        # The legacy grade fit is used to establish the frame. Record the actual surface parameters separately.
        model.update(
            xy=xy,
            z=np.where(support, z, 0).astype("f4"),
            support=support,
            mask=mask,
            confidence=mask.astype("f4"),
            observed_mesh=(v, f, bed),
            capture_reconstruction=capture,
        )

    build(
        archive,
        materials,
        out,
        survey=HERE / "survey.json",
        refine=refine,
        mesh_layout=texture_walls,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "materials", "out"):
        parser.add_argument(arg, type=Path)
    reconstruct(**vars(parser.parse_args()))
