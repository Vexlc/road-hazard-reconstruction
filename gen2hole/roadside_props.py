"""Prepare metric, freestanding road assets from pinned Poly Haven glTF sources."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import struct
import time
from http.client import HTTPException
from urllib.error import URLError
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream/sensors"))
from road_geometry import dielectric_spec, save_json, sha256

RECIPE = ROOT / "gen2hole/examples/roadside-props/sources.json"


def source_files(directory, source, fetch):
    for name, spec in source["files"].items():
        path = directory / name
        if not path.exists() and fetch:
            path.parent.mkdir(exist_ok=True, parents=True)
            with TemporaryDirectory(dir=path.parent) as tmp:
                dest = Path(tmp) / "download"
                request = Request(spec["url"], headers={"User-Agent": "RoadGS asset reproduction/1.0"})
                for attempt in range(4):
                    try:
                        with urlopen(request, timeout=90) as response, dest.open("wb") as out:
                            shutil.copyfileobj(response, out)
                        break
                    except (HTTPException, URLError, TimeoutError):
                        if attempt == 3:
                            raise
                        time.sleep(2**attempt)
                if sha256(dest) != spec["sha256"]:
                    raise ValueError("Downloaded source hash differs: " + name)
                dest.replace(path)
        if not path.is_file() or path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(
                f"Missing or changed source {path}. Use --fetch with an empty source cache to restore it."
            )


class Glb:
    def __init__(self):
        self.doc = dict(
            asset={"version": "2.0"},
            scene=0,
            scenes=[{"nodes": []}],
            nodes=[],
            meshes=[],
            buffers=[],
            bufferViews=[],
            accessors=[],
            images=[],
            textures=[],
            materials=[],
        )
        self.binary = bytearray()

    def view(self, payload):
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        index = len(self.doc["bufferViews"])
        self.doc["bufferViews"].append(dict(buffer=0, byteOffset=len(self.binary), byteLength=len(payload)))
        self.binary.extend(payload)
        return index

    def accessor(self, array, kind, component, bounds=False):
        array = np.ascontiguousarray(array, dtype="<u4" if component == 5125 else "<f4")
        spec = dict(
            bufferView=self.view(array.tobytes()), componentType=component, count=len(array), type=kind
        )
        if bounds:
            spec.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        self.doc["accessors"].append(spec)
        return len(self.doc["accessors"]) - 1

    def image(self, filename):
        idx = len(self.doc["images"])
        self.doc["images"].append(
            dict(
                mimeType="image/png" if filename.suffix.lower() == ".png" else "image/jpeg",
                bufferView=self.view(filename.read_bytes()),
            )
        )
        self.doc["textures"].append(dict(source=idx))
        return idx

    def write(self, filename):
        self.doc["buffers"] = [dict(byteLength=len(self.binary))]
        raw = json.dumps(self.doc, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 4)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        filename.write_bytes(
            struct.pack("<4sIIII", b"glTF", 2, 28 + len(raw) + len(self.binary), len(raw), 0x4E4F534A)
            + raw
            + struct.pack("<II", len(self.binary), 0x004E4942)
            + self.binary
        )


def read_parts(src, folder, chosen):
    binary = (folder / src["buffers"][0]["uri"]).read_bytes()

    def access(index):
        a = src["accessors"][index]
        view = src["bufferViews"][a["bufferView"]]
        dt = np.dtype({5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]])
        count = {"VEC2": 2, "VEC3": 3, "VEC4": 4, "SCALAR": 1}[a["type"]]
        return np.ndarray(
            (a["count"], count),
            dt,
            buffer=binary,
            offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
            strides=(view.get("byteStride", count * dt.itemsize), dt.itemsize),
        ).copy()

    parts = []
    for index in chosen:
        node = src["nodes"][index]
        if set(node) - {"name", "mesh", "translation"}:
            raise ValueError("Unexpected transform in pinned source")
        for p in src["meshes"][node["mesh"]]["primitives"]:
            attrs = {k: access(v) for k, v in p["attributes"].items()}
            attrs["POSITION"] += np.float32(node.get("translation", [0, 0, 0]))
            faces = access(p["indices"]).ravel().reshape(-1, 3).astype("u4")
            parts.append((node["name"], p["material"], attrs, faces))
    return parts


def scene_for(parts):
    scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    for _, _, attrs, faces in parts:
        xyz = attrs["POSITION"]
        scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(faces))
    return scene


def bake_sky(parts, directions=96):
    """Fraction of the surface's unoccluded upper-sky irradiance, in source Y-up coordinates."""
    scene = scene_for(parts)
    for _, _, attrs, _ in parts:
        normals = attrs["NORMAL"]
        origin = attrs["POSITION"] + 0.00045 * normals
        free = np.zeros(len(origin), "f4")
        possible = np.zeros_like(free)
        for i in range(directions):
            y = (i + 0.5) / directions
            r = np.sqrt(1 - y * y)
            phi = i * np.pi * (3 - np.sqrt(5))
            ray = np.float32([r * np.cos(phi), y, r * np.sin(phi)])
            cosine = np.maximum(normals @ ray, 0.0)
            active = cosine > 0
            if active.any():
                rays = np.column_stack([origin[active], np.broadcast_to(ray, (active.sum(), 3))])
                blocked = scene.test_occlusions(o3d.core.Tensor(rays), nthreads=2).numpy()
                free[active] += cosine[active] * ~blocked
            possible += cosine
        attrs["_SKY_VISIBILITY"] = np.divide(free, possible, out=np.zeros_like(free), where=possible > 0)[
            :, None
        ]


def vertically(scene, x, z, start, sign):
    rays = np.zeros((*x.shape, 6), "f4")
    rays[..., 0], rays[..., 1], rays[..., 2], rays[..., 4] = x, start, z, sign
    return start + sign * scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)["t_hit"].numpy()


def texture_at(img, uv):
    # Source samplers use REPEAT, including islands that touch the wrap boundary.
    import cv2

    xy = uv * np.array(img.shape[1::-1]) - 0.5
    parts = []
    for block in np.array_split(xy, max(1, int(np.ceil(len(xy) / 30000)))):
        parts.append(
            cv2.remap(img, block[:, None].astype("f4"), None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[
                :, 0
            ]
        )
    return np.concatenate(parts)


def build(key, spec, source, folder, out):
    src = json.loads((folder / (spec["source"] + "_1k.gltf")).read_text())
    parts = read_parts(src, folder, spec["nodes"])
    write_asset(key, spec, source, folder, out, src, parts)


def write_asset(
    key, spec, source, folder, out, src, parts, *, contact_plane=None, contact_band=0.035, fixed_origin=None
):
    points = np.concatenate([a["POSITION"] for _, _, a, _ in parts])
    low, high = points.min(0), points.max(0)
    axis_x = np.linspace(low[0], high[0], 111)
    axis_z = np.linspace(low[2], high[2], 81)
    x, z = np.meshgrid(axis_x, axis_z)
    bottom = vertically(scene_for(parts), x, z, float(low[1] - 1), 1)
    base = np.isfinite(bottom) & (bottom < low[1] + contact_band)
    if base.sum() < 20:
        raise ValueError("Unresolved support base")
    origin = np.array(
        [
            (x[base].min() + x[base].max()) / 2,
            np.percentile(bottom[base], 25) if contact_plane is None else contact_plane,
            (z[base].min() + z[base].max()) / 2,
        ],
        dtype="f4",
    )
    if fixed_origin is not None:
        origin = np.asarray(fixed_origin, dtype="f4")
    if spec.get("bake_sky"):
        bake_sky(parts)
    glb = Glb()
    if "credit" in spec:
        glb.doc["asset"]["copyright"] = spec["credit"]
    image_cache, materials, maps = {}, {}, {}
    triangles, normals, uvs, material_ids = [], [], [], []
    sky_values = []
    face_count = 0
    for name, mat, attrs, faces in parts:
        attrs["POSITION"] -= origin
        if mat not in materials:
            material = deepcopy(src["materials"][mat])
            props = material["pbrMetallicRoughness"]
            for slot in (
                props["baseColorTexture"],
                props["metallicRoughnessTexture"],
                *([material["normalTexture"]] if "normalTexture" in material else []),
                *([material["occlusionTexture"]] if "occlusionTexture" in material else []),
                *(
                    v
                    for extension in material.get("extensions", {}).values()
                    for k, v in extension.items()
                    if k.endswith("Texture")
                ),
            ):
                entry = src["images"][src["textures"][slot["index"]]["source"]]["uri"]
                if entry not in image_cache:
                    image_cache[entry] = glb.image(folder / entry)
                slot["index"] = image_cache[entry]
            if material.get("extensions"):
                glb.doc["extensionsUsed"] = sorted(
                    set(glb.doc.get("extensionsUsed", [])) | set(material["extensions"])
                )
            materials[mat] = len(glb.doc["materials"])
            glb.doc["materials"].append(material)
            original = src["materials"][mat]["pbrMetallicRoughness"]
            maps[mat] = [
                np.asarray(
                    Image.open(
                        folder / src["images"][src["textures"][original[k]["index"]]["source"]]["uri"]
                    ).convert("RGB")
                )
                for k in ["baseColorTexture", "metallicRoughnessTexture"]
            ]
        mapping = {
            k: glb.accessor(
                v, "SCALAR" if v.shape[1] == 1 else "VEC" + str(v.shape[1]), 5126, bounds=k == "POSITION"
            )
            for k, v in attrs.items()
        }
        primitive = dict(
            attributes=mapping, indices=glb.accessor(faces.ravel(), "SCALAR", 5125), material=materials[mat]
        )
        idx = len(glb.doc["nodes"])
        label = (
            spec.get("mesh_prefix", "Barrier" if key.startswith("B") else "Hydrant")
            + " "
            + name
            + " "
            + str(idx)
        )
        if spec.get("keep_mesh_names"):
            label = name
        glb.doc["meshes"].append(dict(name=label, primitives=[primitive]))
        glb.doc["nodes"].append(dict(name=label, mesh=idx))
        glb.doc["scenes"][0]["nodes"].append(idx)
        triangles.append(attrs["POSITION"][faces])
        normals.append(attrs["NORMAL"][faces])
        uvs.append(attrs["TEXCOORD_0"][faces])
        if spec.get("bake_sky"):
            sky_values.append(attrs["_SKY_VISIBILITY"][faces])
        material_ids.append(np.full(len(faces), mat))
        face_count += len(faces)
    out.mkdir(parents=True)
    glb.write(out / "asset.glb")
    # glTF Y-up, source Z points towards our negative local Y.
    scene = scene_for(parts)
    height = float(high[1] - origin[1])
    extent = (high - low)[[0, 2]] + 2 * max(0.4, 1.5 * height)
    if fixed_origin is not None:
        extent = 2 * np.maximum(high - origin, origin - low)[[0, 2]] + 2 * max(0.4, 1.5 * height)
    n = 193
    x, z = np.meshgrid(
        np.linspace(-extent[0] / 2, extent[0] / 2, n), np.linspace(-extent[1] / 2, extent[1] / 2, n)
    )
    top = vertically(scene, x, z, height + 1, -1)
    under = vertically(scene, x, z, -1, 1)
    support = np.isfinite(under) & (under < contact_band)
    xyz = np.stack([x, np.full_like(x, 0.002), z], -1).astype("f4")
    occluded = np.zeros((n, n), "f4")
    # Cosine-weighted integration over the upper hemisphere from original ground pixels.
    for k in range(96):
        r = np.sqrt((k + 0.5) / 96)
        phi = k * np.pi * (3 - np.sqrt(5))
        direction = np.float32([r * np.cos(phi), np.sqrt(1 - r * r), r * np.sin(phi)])
        rays = np.concatenate([xyz, np.broadcast_to(direction, xyz.shape)], axis=-1)
        occluded += scene.test_occlusions(o3d.core.Tensor(rays), nthreads=3).numpy()
    occluded = gaussian_filter(occluded / 96, 1.0)
    fade = np.clip(np.minimum(1 - 2 * abs(x) / extent[0], 1 - 2 * abs(z) / extent[1]) / 0.12, 0, 1)
    np.savez_compressed(
        out / "surface.npz",
        width=extent[0],
        length=extent[1],
        height=np.where(np.isfinite(top), top, 0).astype("f4"),
        defect_mask=np.zeros_like(support),
        contact_mask=support,
        sky_blocked=(occluded * fade).astype("f4"),
    )
    tri = np.concatenate(triangles)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    count = int(np.clip(area.sum() * 26000, 24000, 130000))
    sample = qmc.Sobol(3, scramble=True, seed=232).random_base2(int(np.ceil(np.log2(count))))[:count]
    chosen = np.searchsorted(np.cumsum(area, dtype="f8"), sample[:, 0] * area.sum(dtype="f8"), side="right")
    a = np.sqrt(sample[:, 1])
    bary = np.stack([1 - a, a * (1 - sample[:, 2]), a * sample[:, 2]], 1).astype("f4")

    def interp(values):
        return (values[chosen] * bary[:, :, None]).sum(1)

    xyz = interp(tri)
    norm = interp(np.concatenate(normals))
    norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-8)
    uv = interp(np.concatenate(uvs))
    ids = np.concatenate(material_ids)[chosen]
    base_rgb = np.empty((count, 3), "u1")
    arm = np.empty_like(base_rgb)
    optics = {mat: dielectric_spec(src["materials"][mat]) for mat in maps}
    reflectance = np.zeros((count, 3), "f4")
    rough = np.ones(count, "f4")
    for mat, (diffuse, packed) in maps.items():
        hits = ids == mat
        base_rgb[hits], arm[hits] = texture_at(diffuse, uv[hits]), texture_at(packed, uv[hits])

        if optics[mat] is not None:
            f0, factor = optics[mat]
            reflectance[hits] = f0
            rough[hits] = arm[hits, 1] / 255.0 * factor

    def local_z(a):
        return np.float32(a[:, [0, 2, 1]] * [1, -1, 1])

    np.savez_compressed(
        out / "samples.npz",
        xyz=local_z(xyz),
        normal=local_z(norm),
        albedo=base_rgb,
        arm=arm,
        **(
            {"dielectric_f0": reflectance, "specular_roughness": rough}
            if any(x is not None for x in optics.values())
            else {}
        ),
        spacing_m=np.float32(np.sqrt(area.sum() / count)),
        **({"sky_visibility": interp(np.concatenate(sky_values))[:, 0]} if sky_values else {}),
    )
    save_json(
        out / "metadata.json",
        dict(
            id=key,
            label=spec["label"],
            source=source["page"],
            source_credit=source["author"],
            license=source["license"],
            geometry=spec.get(
                "geometry",
                "Published textured metric model; source nodes flattened, independent finish selected and base registered",
            ),
            source_nodes=spec["nodes"],
            source_origin_y_up_m=origin.tolist(),
            size_xyz_m=(high - low)[[0, 2, 1]].tolist(),
            base_sample_count=int(base.sum()),
            base_p90_abs_m=float(np.percentile(abs(bottom[base] - origin[1]), 90)),
            triangles=face_count,
            surface_samples=count,
            files={f: sha256(out / f) for f in ["asset.glb", "surface.npz", "samples.npz"]},
        ),
    )
    print(key, face_count, count, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=ROOT / "work/roadside-scans")
    p.add_argument("--out", type=Path, default=ROOT / "runs/roadside-props")
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    recipe = json.loads(RECIPE.read_text())
    for name, source in recipe["sources"].items():
        source_files(args.sources / name, source, args.fetch)
    if args.out.exists():
        p.error("Choose a fresh output directory")
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "assets"
        for key, spec in recipe["assets"].items():
            name = spec["source"]
            build(key, spec, recipe["sources"][name], args.sources / name, stage / key)
        stage.rename(args.out)


if __name__ == "__main__":
    main()
