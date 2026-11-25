"""Reconstruct four supported road-debris models from hash-pinned, textured metric meshes."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from roadside_props import ROOT, read_parts, save_json, source_files, write_asset
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from surface_normals import tangent_mesh

RECIPE = ROOT / "gen2hole/examples/road-debris/sources.json"


def seat(parts, initial):
    vertices = np.concatenate([a["POSITION"] for _, _, a, _ in parts]) @ initial.T
    triangles = np.concatenate([a["POSITION"][f] for _, _, a, f in parts]) @ initial.T
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    center = np.average(triangles.mean(1), axis=0, weights=area)
    hull = ConvexHull(vertices)
    directions = np.unique(np.round(hull.equations[:, :3], 7), axis=0)
    for down in directions[np.argsort(directions[:, 1], kind="stable")]:
        down = down / np.linalg.norm(down)
        if -down[1] < np.cos(np.deg2rad(12)):
            break
        up = -down
        tangent = np.array([1.0, 0.0, 0.0]) - up * up[0]
        tangent /= np.linalg.norm(tangent)
        basis = np.stack([tangent, up, np.cross(tangent, up)])
        heights = vertices @ up
        contacts = (vertices[heights <= heights.min() + 0.0008] - center) @ basis[[0, 2]].T
        if len(contacts) < 3 or np.linalg.matrix_rank(contacts - contacts.mean(0)) < 2:
            continue
        margin = float(-ConvexHull(contacts).equations[:, -1].max())
        if margin > 0.004:
            correction = float(np.rad2deg(np.arccos(np.clip(up[1], -1, 1))))
            return basis @ initial, {
                "contact_band_m": 0.0008,
                "estimated_support_margin_m": margin,
                "correction_deg": correction,
                "contact_vertices": len(contacts),
                "mass_proxy": "area-weighted shell centroid",
            }
    raise ValueError("No supporting plane contains the estimated centre for this orientation")


def build(out, sources, fetch, recipe_path=RECIPE):
    recipe = json.loads(recipe_path.read_text())
    for name, source in recipe["sources"].items():
        source_files(sources / name, source, fetch)
    for key, spec in recipe["assets"].items():
        name = spec["source"]
        folder = sources / name
        source = recipe["sources"][name]
        doc = json.loads((folder / (name + "_1k.gltf")).read_text())
        parts = read_parts(doc, folder, spec["nodes"])
        transform, support = seat(
            parts, Rotation.from_euler("xyz", spec["orientation_xyz_deg"], degrees=True).as_matrix()
        )
        for _, _, attrs, _ in parts:
            for field in ("POSITION", "NORMAL"):
                attrs[field] = np.ascontiguousarray(attrs[field] @ transform.T, dtype="f4")
            if "TANGENT" in attrs:
                attrs["TANGENT"][:, :3] = attrs["TANGENT"][:, :3] @ transform.T
        parts = [
            (name, mat, *tangent_mesh(attrs, faces, authored_v_up=True)) for name, mat, attrs, faces in parts
        ]
        low = min(float(a["POSITION"][:, 1].min()) for _, _, a, _ in parts)
        spec = dict(
            spec,
            bake_sky=True,
            credit=source["author"] + " / Poly Haven, CC0",
            geometry="Published metric mesh oriented on a support hull face",
        )
        write_asset(key, spec, source, folder, out / key, doc, parts, contact_plane=low, contact_band=0.006)
        path = out / key / "metadata.json"
        meta = json.loads(path.read_text())
        meta.update(
            support_pose=dict(support, rotation_from_source=transform.tolist()),
            source_input_sha256={k: v["sha256"] for k, v in source["files"].items()},
        )
        save_json(path, meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, default=RECIPE, help="Source fingerprints and per-object orientation")
    p.add_argument("--sources", type=Path, help="Source cache; defaults to original/ next to the recipe")
    p.add_argument("--fetch", action="store_true", help="Download and verify missing original files")
    p.add_argument("--out", type=Path, default=ROOT / "runs/road-debris")
    a = p.parse_args()
    if a.out.exists():
        p.error("Choose a fresh output directory")
    a.out.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=a.out.parent) as tmp:
        stage = Path(tmp) / "assets"
        build(stage, a.sources or a.recipe.parent / "original", a.fetch, a.recipe)
        stage.rename(a.out)


if __name__ == "__main__":
    main()
