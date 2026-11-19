"""Audit each generated GLB, 3DGS PLY, preview, scale, and provenance."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from plyfile import PlyData

ORDER = ["stereo", "rgbd", "pothole", "missing", "raised"]
DTYPES = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}


def read_accessor(binary, document, accessor, width):
    spec = document["accessors"][accessor]
    view = document["bufferViews"][spec["bufferView"]]
    dtype = np.dtype(DTYPES[spec["componentType"]])
    offset = spec.get("byteOffset", 0) + view.get("byteOffset", 0)
    stride = view.get("byteStride", width*dtype.itemsize)
    return np.ndarray((spec["count"], width), dtype=dtype, buffer=binary, offset=offset,
                      strides=(stride, dtype.itemsize))


def glb_check(path):
    raw = path.read_bytes()
    magic, version, total, jar, chunk = struct.unpack("<4sIII4s", raw[:20])
    assert (magic, version, total, chunk) == (b"glTF", 2, len(raw), b"JSON")
    model = json.loads(raw[20:20+jar])
    binary_count, binary_title = struct.unpack_from("<I4s", raw, 20+jar)
    assert binary_title == b"BIN\0"
    binary = memoryview(raw)[28+jar:28+jar+binary_count]
    assert model["materials"] and model["images"] and model["scenes"]
    assert all("bufferView" in im and "uri" not in im for im in model["images"])
    for image in model["images"]:
        view = model["bufferViews"][image["bufferView"]]
        offset = view.get("byteOffset", 0)
        assert offset+view["byteLength"] <= len(binary)
        assert bytes(binary[offset:offset+8]).startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8"))
    nfaces = 0
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for m in model["meshes"]:
        for primitive in m["primitives"]:
            assert primitive.get("mode", 4) == 4
            vertices = read_accessor(binary, model, primitive["attributes"]["POSITION"], 3)
            uv = read_accessor(binary, model, primitive["attributes"]["TEXCOORD_0"], 2)
            indices = read_accessor(binary, model, primitive["indices"], 1)
            assert len(vertices) == len(uv) and len(indices) % 3 == 0 and indices.max() < len(vertices)
            assert np.isfinite(vertices).all() and np.isfinite(uv).all()
            low = np.minimum(low, vertices.min(0))
            high = np.maximum(high, vertices.max(0))
            nfaces += len(indices)//3
    assert nfaces > 1000 and low[1] < -.002 and high[1] < 1.6
    return {"glb_triangles": nfaces, "embedded_images": len(model["images"]),
            "glb_bounds_y_up": [low.tolist(), high.tolist()],
            "glb_sha256": sha256(raw).hexdigest()}


def ply_check(path, count):
    ply = PlyData.read(path, mmap="r")
    assert not ply.text and ply.byte_order == "<"
    data = ply["vertex"].data
    expected = {"x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)),
                *(f"rot_{i}" for i in range(4)), *(f"f_dc_{i}" for i in range(3)),
                *(f"f_rest_{i}" for i in range(9))}
    assert len(data) == count and len(data) > 1000 and expected.issubset(data.dtype.names)
    assert all(np.isfinite(data[k]).all() for k in expected)
    norm2 = sum(data[f"rot_{i}"]**2 for i in range(4))
    assert abs(norm2-1).max() < 1e-4
    assert np.max(data["scale_0"]) < 0 and np.max(data["scale_2"]) < 0
    assert np.min(data["z"]) < -.002
    with path.open("rb") as stream:
        checksum = sha256()
        for block in iter(lambda: stream.read(2**21), b""):
            checksum.update(block)
    return checksum.hexdigest()


def audit(job):
    library, path = job
    folder = path.parent
    try:
        m = json.loads(path.read_text())
        assert m["id"] == folder.name and m["category"] == folder.parent.name and m["units"] == "metres"
        assert m["width_m"] > 0 and m["length_m"] > 0
        mesh = folder / m["mesh_name"]
        ply = folder / m["ply_name"]
        preview = folder / m["standalone_preview"]
        info = glb_check(mesh)
        ply_hash = ply_check(ply, m["gaussians"])
        with Image.open(preview) as rendered:
            rendered.load()
            assert rendered.size == (660, 480)
            assert np.asarray(rendered.convert("L")).std() > 2.5
        with np.load(folder / "surface.npz") as field:
            assert field["height"].shape == field["alpha"].shape
            assert np.isfinite(field["height"]).all() and np.isfinite(field["alpha"]).all()
            assert field["alpha"].min() >= 0 and field["alpha"].max() <= 1
            assert np.mean(field["alpha"] > .5) > .13 and field["height"].min() < -.002
        if m["category"] in {"stereo", "rgbd"}:
            cap = json.loads((folder / "capture_metadata.json").read_text())
            assert m["quality_checks"] and all(m["quality_checks"].values())
            assert (folder / "source_crop.png").is_file() and cap["inputs"]["rgb"]["sha256"]
            info["source_rgb_hash"] = cap["inputs"]["rgb"]["sha256"]
            if m["category"] == "rgbd":
                info["source_depth_hash"] = cap["inputs"]["depth"]["sha256"]
        return dict(m, **info, status="validated",
                    glb=str(mesh.relative_to(library)), glb_bytes=mesh.stat().st_size,
                    ply=str(ply.relative_to(library)), ply_bytes=ply.stat().st_size,
                    ply_sha256=ply_hash, preview=str(preview.relative_to(library)),
                    surface=str((folder / "surface.npz").relative_to(library)),
                    metadata=str(path.relative_to(library)))
    except (AssertionError, KeyError, OSError, ValueError) as exc:
        return {"id": folder.name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def verify(root):
    files = sorted(p for kind in ORDER for p in (root / kind).glob("*/metadata.json"))
    with ThreadPoolExecutor(max_workers=7) as executor:
        results = []
        for i, result in enumerate(executor.map(audit, ((root,p) for p in files)), 1):
            results.append(result)
            if i%100 == 0 or result["status"] != "validated":
                print(i, result["id"], result["status"], result.get("error", ""), flush=True)
    good = [r for r in results if r["status"] == "validated"]
    bad = [r for r in results if r["status"] != "validated"]
    print("Validated", len(good), "failed", len(bad), Counter(x["category"] for x in good), flush=True)
    if bad:
        (root / "validation_failures.json").write_text(json.dumps(bad, indent=2) + "\n")
        raise SystemExit(1)
    assert len({m["id"] for m in good}) == len(good)
    assert len({m["glb_sha256"] for m in good}) == len(good), "Duplicate mesh bytes"
    assert len({m["ply_sha256"] for m in good}) == len(good), "Duplicate Gaussian bytes"
    (root / "asset_manifest.json").write_text(json.dumps(good, ensure_ascii=False, indent=2) + "\n")
    (root / "validation.json").write_text(json.dumps({
        "verified_assets": len(good),
        "counts": dict(Counter(x["category"] for x in good)),
        "all_glbs_and_gaussian_files_are_byte_distinct": True,
        "checks": ["glTF 2.0 embedded texture and buffer bounds", "mesh vertices and UV finite",
                   "triangle indices in range", "finite 3DGS positions, opacity, scales and SH coefficients",
                   "unit Gaussian quaternions", "per-asset rendered preview available",
                   "valid local road cutout field", "recorded source and depth screening for real captures"]
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    verify(parser.parse_args().library)
