"""Register the upright and mechanically supported fallen poses of a textured traffic cone."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from roadside_props import ROOT, read_parts, source_files, write_asset
from road_geometry import save_json, sha256

SOURCE = ROOT / "gen2hole/examples/traffic-cones"
CREDIT = "Traffic Cone by hinndia (CC BY 4.0), KhronosGroup glTF adaptation; RoadGS support poses."


def stable_side(vertices, faces):
    """Find a near-side support face containing the projected centre of mass with positive margin."""
    triangles = vertices[faces]
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    # Use uniform thin-shell density. The resulting COM is an estimate, not a material measurement.
    center = np.average(triangles.mean(1), axis=0, weights=area)
    hull = ConvexHull(vertices)
    directions = np.unique(np.round(hull.equations[:, :3], 5), axis=0)
    for down in sorted(directions, key=lambda v: -v[0]):
        down = down / np.linalg.norm(down)
        if down[0] < 0.85:
            break
        depth = vertices @ down
        support = vertices[depth > depth.max() - 0.001]
        if len(support) < 3:
            continue
        across = np.cross(down, [0, 1, 0])
        across /= np.linalg.norm(across)
        tangent = np.cross(down, across)
        xy = (support - center) @ np.stack([across, tangent]).T
        if np.linalg.matrix_rank(xy - xy.mean(0)) < 2:
            continue
        margin = -ConvexHull(xy).equations[:, -1].max()
        if margin <= 0.002:
            continue
        # Keep the cone's longitudinal axis in the X/Y plane of the seated model.
        up = -down
        heading = np.array([0, 1, 0]) - up * up[1]
        heading /= np.linalg.norm(heading)
        matrix = np.stack([heading, up, np.cross(heading, up)])
        return matrix, dict(
            support_margin_m=float(margin),
            contact_vertices=len(support),
            com_model="uniform thin shell",
            contact_tolerance_m=0.001,
            tilt_from_vertical_deg=float(np.degrees(np.arccos(up[1]))),
        )
    raise ValueError("No side equilibrium covering the model's centre of mass")


def prepare(folder, source, out, key, fallen):
    doc = json.loads((folder / "TrafficCone.gltf").read_text())
    # Use a single cone from the distributor's demonstration scene. Its parent supplies the Z-up to Y-up rotation.
    node = doc["nodes"][1]
    parent = next(n for n in doc["nodes"] if 1 in n.get("children", []))
    rotation = Rotation.from_quat(parent["rotation"]).as_matrix()
    ((name, material, attrs, faces),) = read_parts(doc, folder, [1])
    verts = (attrs["POSITION"] - node["translation"]) @ rotation.T
    norm = attrs["NORMAL"] @ rotation.T
    matrix, stats = stable_side(verts, faces) if fallen else (np.eye(3), dict(tilt_from_vertical_deg=0.0))
    attrs["POSITION"] = np.ascontiguousarray(verts @ matrix.T, dtype="f4")
    attrs["NORMAL"] = np.ascontiguousarray(norm @ matrix.T, dtype="f4")
    material_spec = doc["materials"][material]
    # Khronos supplies roughness and has no AO texture. Keep G/B and use unity for our ambient channel.
    tex = material_spec["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    image_id = doc["textures"][tex]["source"]
    filename = doc["images"][image_id]["uri"]
    with Image.open(folder / filename) as image:
        packed = np.array(image.convert("RGB"))
    packed[..., 0] = 255
    with TemporaryDirectory() as temp:
        temp = Path(temp)
        for img in doc["images"]:
            if (folder / img["uri"]).exists() and img["uri"] != filename:
                shutil.copyfile(folder / img["uri"], temp / img["uri"])
        Image.fromarray(packed).save(temp / filename)
        label = "侧倒交通锥" if fallen else "直立交通锥"
        spec = dict(
            label=label,
            nodes=[1],
            bake_sky=True,
            credit=CREDIT,
            mesh_prefix="Traffic cone",
            geometry="Original textured cone seated by its supporting geometry",
        )
        write_asset(
            key,
            spec,
            source,
            temp,
            out,
            doc,
            [(name, material, attrs, faces)],
            contact_plane=float(attrs["POSITION"][:, 1].min()),
            contact_band=0.006,
        )
    meta = json.loads((out / "metadata.json").read_text())
    meta["support_pose"] = dict(**stats, rotation_from_upright=matrix.tolist())
    meta["source_manifest_sha256"] = sha256(folder / "source.json")
    save_json(out / "metadata.json", meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=ROOT / "runs/traffic-cones")
    args = p.parse_args()
    if args.out.exists():
        p.error("Choose a fresh output directory")
    source = json.loads((args.source / "source.json").read_text())
    source_files(args.source, source, False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "cones"
        for key, fallen in (("C236", False), ("C237", True)):
            prepare(args.source, source, stage / key, key, fallen)
        stage.rename(args.out)


if __name__ == "__main__":
    main()
