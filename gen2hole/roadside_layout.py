"""Bake supported freestanding asset instances into one metric ray scene with mutual occlusion and shadows."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from roadside_props import read_parts, write_asset
from road_geometry import save_json, sha256


def image_slots(material):
    pbr = material["pbrMetallicRoughness"]
    yield pbr["baseColorTexture"]
    yield pbr["metallicRoughnessTexture"]
    for key in ("normalTexture", "occlusionTexture"):
        if key in material:
            yield material[key]
    for ext in material.get("extensions", {}).values():
        for k, v in ext.items():
            if k.endswith("Texture"):
                yield v


def append_asset(folder, instance, stage, materials, images):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["id"] != instance["asset"]:
        raise ValueError("Asset ID and metadata disagree")
    for name in ("asset.glb", "surface.npz"):
        if sha256(folder / name) != meta["files"][name]:
            raise ValueError("Changed asset: " + str(folder / name))
    with np.load(folder / "surface.npz") as fields:
        if "contact_mask" not in fields or fields["defect_mask"].any():
            raise ValueError("A common ground plane layout requires freestanding objects")
    blob = (folder / "asset.glb").read_bytes()
    length = struct.unpack_from("<I", blob, 12)[0]
    doc = json.loads(blob[20 : 20 + length])
    binary = memoryview(blob)[28 + length :]
    (stage / "mesh.bin").write_bytes(binary)
    doc["buffers"][0]["uri"] = "mesh.bin"
    base = len(materials)
    for m in doc["materials"]:
        combined = deepcopy(m)
        for slot in image_slots(combined):
            tex = doc["textures"][slot["index"]]
            img = doc["images"][tex["source"]]
            view = doc["bufferViews"][img["bufferView"]]
            start = view.get("byteOffset", 0)
            data = binary[start : start + view["byteLength"]]
            filename = (
                hashlib.sha256(data).hexdigest()
                + {"image/png": ".png", "image/jpeg": ".jpg"}[img["mimeType"]]
            )
            if filename not in images:
                images.append(filename)
                (stage / filename).write_bytes(data)
            slot["index"] = images.index(filename)
        materials.append(combined)
    xy = np.asarray(instance["xy_m"], float)
    yaw = float(instance["heading_deg"])
    if xy.shape != (2,) or not np.isfinite([*xy, yaw]).all():
        raise ValueError("Expected finite XY and heading")
    rotation = Rotation.from_euler("y", yaw, degrees=True).as_matrix().astype("f4")
    local = doc["scenes"][doc.get("scene", 0)]["nodes"]
    parts = []
    for name, mat, attrs, faces in read_parts(doc, stage, local):
        attrs["POSITION"] = np.float32(attrs["POSITION"] @ rotation.T + [xy[0], 0, -xy[1]])
        attrs["NORMAL"] = np.float32(attrs["NORMAL"] @ rotation.T)
        if "TANGENT" in attrs:
            attrs["TANGENT"][:, :3] = attrs["TANGENT"][:, :3] @ rotation.T
        parts.append((name + " @" + instance["id"], base + mat, attrs, faces))
    xyz = np.concatenate([a["POSITION"] for _, _, a, _ in parts])
    outline = cv2.convexHull(np.ascontiguousarray(xyz[:, [0, 2]]))
    return (
        parts,
        outline,
        dict(
            **instance,
            source=meta["source"],
            credit=meta["source_credit"],
            license=meta["license"],
            glb_sha256=meta["files"]["asset.glb"],
        ),
    )


def build(assets, recipe, out):
    layout = json.loads(recipe.read_text())
    placements = layout["instances"]
    names = [v["id"] for v in placements]
    if len(set(names)) != len(names) or not names or not all(names):
        raise ValueError("Provide one unique nonempty ID for each instance")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)
        parts = []
        outlines = []
        records = []
        materials = []
        images = []
        for row in placements:
            more, outline, record = append_asset(assets / row["asset"], row, stage, materials, images)
            if any(cv2.intersectConvexConvex(outline, prev)[0] > 1e-6 for prev in outlines):
                raise ValueError("Placed objects have overlapping plan envelopes")
            parts.extend(more)
            outlines.append(outline)
            records.append(record)
        src = dict(
            materials=materials,
            images=[dict(uri=n) for n in images],
            textures=[dict(source=i) for i in range(len(images))],
        )
        sources = dict(
            page="; ".join(sorted({r["source"] for r in records})),
            author=", ".join(sorted({r["credit"] for r in records})),
            license=", ".join(sorted({r["license"] for r in records})),
        )
        spec = dict(
            label=layout["label"],
            nodes=list(range(len(parts))),
            keep_mesh_names=True,
            bake_sky=True,
            credit=" | ".join(
                sorted({r["credit"] + " " + r["license"] + " " + r["source"] for r in records})
            ),
            geometry="Rigid measured-scale object layout, instance transforms in local Z-up metres",
        )
        dest = stage / "layout"
        write_asset(
            layout["id"], spec, sources, stage, dest, src, parts, contact_band=0.01, fixed_origin=[0, 0, 0]
        )
        meta = json.loads((dest / "metadata.json").read_text())
        meta["instances"] = records
        save_json(dest / "metadata.json", meta)
        dest.rename(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets", type=Path)
    p.add_argument("layout", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    build(a.assets, a.layout, a.out)
