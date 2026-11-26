"""Install the original Poly Haven water frame with the lid missing or displaced on grade."""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[1]
sys.path[:0] = [str(GEN), str(GEN / "recipes/series"), str(GEN / "recipes/common")]
from export_splats import make_one
from mesh_io import Mesh, grid_faces, smooth, write_json
from roadside_props import read_parts

SPAN, DEPTH = 1.8, 0.84
STATES = [("M354", "铸铁井圈完整的缺盖井口", 0.0), ("A355", "井盖滑移后露出井口", 0.36)]


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def add_band(mesh, name, rows, uv, mat, road=False):
    grid = np.array(rows)
    mesh.add(name, grid, grid_faces(*grid.shape[:2], flip=True), uv, mat, road)


def contour(points):
    hull = points[ConvexHull(points).vertices]
    return np.vstack([hull, hull[0]])


def flatten_scene(path):
    """Drop trimesh's identity-only world container for the project's flat GLB readers."""
    data = path.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + size])
    scene = doc["scenes"][doc.get("scene", 0)]
    if all("mesh" in doc["nodes"][i] for i in scene["nodes"]):
        return
    (root,) = scene["nodes"]
    node = doc["nodes"][root]
    if set(node) != {"children", "name"} or set(node["children"]) != set(
        range(len(doc["nodes"]))
    ) - {root}:
        raise ValueError("Cannot flatten transformed or nested source")
    scene["nodes"] = [i - (i > root) for i in node["children"]]
    del doc["nodes"][root]
    payload = json.dumps(doc, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    binary = data[20 + size :]
    path.write_bytes(
        struct.pack(
            "<4sIIII",
            b"glTF",
            2,
            20 + len(payload) + len(binary),
            len(payload),
            0x4E4F534A,
        )
        + payload
        + binary
    )


def build(source, out, key, label, shift):
    source_info = json.loads((HERE / "chamber-source.json").read_text())[
        "water_manhole_cover"
    ]
    for filename, info in source_info["files"].items():
        if (
            hashlib.sha256((source / filename).read_bytes()).hexdigest()
            != info["sha256"]
        ):
            raise ValueError("Source digest differs for " + filename)
    doc = json.loads(next(source.glob("*.gltf")).read_text())
    frame, lid = read_parts(doc, source, [0, 1])
    mesh = Mesh()
    tex = [pixels(source / x["uri"]) for x in doc["images"]]
    iron = mesh.material("Original cast iron", tex[1], 0.78, 0.62, lossless_color=True)
    mesh.normal_maps[iron], mesh.orm_maps[iron] = tex[0], tex[2]
    common = HERE.parent / "perforated-covers"
    road = mesh.material(
        "Scanned pavement", pixels(common / "split-iron/pavement.jpg"), 0.93, 0
    )
    lining = pixels(common / "lining.jpg")
    wall = mesh.material("Designed shaft lining", np.tile(lining, (1, 3, 1)), 0.95, 0)
    floor = mesh.material("Designed shaft floor", lining, 0.96, 0)

    def shape(part):
        _, _, attrs, faces = part
        xyz = attrs["POSITION"][:, [0, 2, 1]].copy()
        xyz[:, 1] *= -1
        uv = attrs["TEXCOORD_0"].copy()
        uv[:, 1] = 1 - uv[:, 1]
        return xyz, faces, uv

    v, faces, uv = shape(frame)
    datum = float(v[:, 2].max())
    v[:, 2] -= datum
    mesh.add("Cast ring and cover seat", v, faces, uv, iron)
    r = np.linalg.norm(v[:, :2], axis=1)
    outside = contour(v[r > 0.344, :2])
    # Bury the very edge under grade to seal the original rounded casting.
    outside *= 0.991
    square = outside / np.abs(outside).max(1)[:, None] * (SPAN / 2)
    apron = np.stack([(1 - t) * square + t * outside for t in np.linspace(0, 1, 19)])
    add_band(
        mesh,
        "Road around the aperture",
        np.dstack([apron, np.zeros(apron.shape[:2])]),
        0.5 + apron / SPAN,
        road,
        True,
    )
    bore = contour(v[r < r.min() + 0.0001, :2])
    ends = [float(v[:, 2].min() + 0.0003), -DEPTH]
    rows = np.stack([np.c_[bore, np.full(len(bore), z)] for z in ends])
    u = np.linspace(0, 1, len(bore))
    add_band(
        mesh,
        "Inspection shaft",
        rows,
        np.stack([np.c_[u, np.full_like(u, t)] for t in [1, 0]]),
        wall,
    )
    floorxy = np.vstack([[0, 0], bore[:-1]])
    i = np.arange(1, len(floorxy))
    mesh.add(
        "Shaft bottom",
        np.c_[floorxy, np.full(len(floorxy), -DEPTH)],
        np.c_[i * 0, i, np.roll(i, -1)],
        0.5 + floorxy / 0.72,
        floor,
    )
    if shift:
        lidv, lf, luv = shape(lid)
        lidv[:, 0] += shift
        lidv[:, 2] -= lidv[:, 2].min()
        f = lidv[lf]
        normal = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
        z = normal[:, 2] / np.maximum(1e-14, np.linalg.norm(normal, axis=1))
        for part, keep in [
            ("Lid face", z > 0.30),
            ("Lid underside", z < -0.30),
            ("Lid rolled edge", abs(z) <= 0.30),
        ]:
            mesh.add(part, lidv, lf[keep], luv, iron)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    for name, v, faces, *_ in mesh.parts:
        scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(faces.astype("u4")))
    axis = np.linspace(-SPAN / 2, SPAN / 2, 321, dtype="f4")
    x, y = np.meshgrid(axis, -axis)
    rays = np.stack([x, y, x * 0 + 1, x * 0, x * 0, x * 0 - 1], -1)
    hits = scene.cast_rays(o3d.core.Tensor(rays))
    height = 1 - hits["t_hit"].numpy()
    if not np.isfinite(height[1:-1, 1:-1]).all():
        raise ValueError("Unclosed rim-to-road or rim-to-shaft seam")
    height[~np.isfinite(height)] = 0
    names = np.array([p[0] for p in mesh.parts] + ["Outside"])
    part = names[np.minimum(hits["geometry_ids"].numpy(), len(mesh.parts))]
    opening = np.isin(part, ["Inspection shaft", "Shaft bottom"])
    defect = ~np.isin(part, ["Road around the aperture", "Outside"])
    from scipy.ndimage import distance_transform_edt

    alpha = smooth(
        1 - distance_transform_edt(~defect) * float(axis[1] - axis[0]) / 0.065
    )
    field = dict(
        height=height,
        alpha=alpha,
        defect_mask=defect,
        opening_mask=opening,
        width=SPAN,
        length=SPAN,
    )
    category = "raised" if shift else "missing"
    dest = out / "series/assets" / category / key
    meta = dict(
        id=key,
        label=label,
        category=category,
        family="water_manhole_cover",
        seed=int(key[1:]) + 475,
        source=source_info["page"],
        author=source_info["author"],
        source_license="CC0",
        original_frame_triangles=len(frame[3]),
        road_datum_in_source_y_m=datum,
        frame_outside_diameter_m=float(2 * r.max()),
        clear_diameter_at_lower_lip_m=float(2 * r.min()),
        designed_floor_depth_m=DEPTH,
        designed_lid_shift_x_m=shift,
        open_area_m2=float(opening.sum() * (SPAN / 320) ** 2),
        geometry_source="Published cast parts at source scale. Shaft, floor, surrounding road and installed pose are designed.",
    )
    mesh.save(dest, out / "series/mesh_cache", meta, field)
    flatten_scene(dest / "asset.glb")
    return make_one(dest, out / "series/mesh_cache")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    entries = [build(a.source, a.out, *state) for state in STATES]
    write_json(a.out / "series/assets/asset_manifest.json", entries)
    write_json(a.out / "run.json", dict(editions=["series"]))
