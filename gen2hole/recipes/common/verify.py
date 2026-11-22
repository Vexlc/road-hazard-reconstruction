"""Decode every requested asset and check the geometry, metric field and per-Gaussian labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from verify_formats import glb, ply


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_asset(directory, manifest_row, run):
    meta = read(directory / "metadata.json")
    name = directory.name
    require(meta["id"] == name == manifest_row["id"], f"{name}: asset identity")
    require(all(manifest_row.get(key) == value for key, value in meta.items()), f"{name}: manifest vs metadata")
    require(meta["units"] == "metres" and meta["glb_up"] == "Y" and meta["gaussian_up"] == "Z", f"{name}: units/axes")
    require(sha(ROOT / meta["source_image"]) == meta["source_sha256"], f"{name}: source image")
    if meta.get("depth_profile") in ("masked_groove", "scanned_relief"):
        require(sha(ROOT/"calibration"/(name+"_outline.png")) == meta["mask_sha256"], f"{name}: source crack mask")
    mesh = directory / "asset.glb"
    geometry_hash, faces, bottom, top = glb(mesh)
    require(faces == sum(meta["geometry_parts"].values()) and bottom < 0, f"{name}: triangle count/road datum")
    require(np.allclose([bottom, top], [meta["lowest_z_m"], meta["highest_z_m"]], atol=2e-5), f"{name}: GLB Y vs local Z")
    with np.load(directory / "surface.npz") as field:
        height, alpha, defect = (field[k] for k in ("height", "alpha", "defect_mask"))
        require(height.ndim == 2 and height.shape == alpha.shape == defect.shape, f"{name}: field shape")
        require(all(np.isfinite(field[k]).all() for k in field.files), f"{name}: nonfinite field")
        require(float(field["width"]) > 0 and float(field["length"]) > 0, f"{name}: field extent")
        require(alpha.min() >= 0 and alpha.max() <= 1 and (defect > 0.5).any(), f"{name}: cutout/blend")
        if meta.get("depth_profile") == "scanned_relief":
            with np.load(ROOT/"calibration"/(name+"_samples.npz"), allow_pickle=False) as samples:
                measured = samples["height"]
            span = float(field["width"])
            pitch = meta["source_grid_pitch_m"]
            require(measured.shape == height.shape and np.isfinite(measured).all(), f"{name}: scan samples")
            require(np.isclose(span, (height.shape[1]-1)*pitch), f"{name}: scan spacing")
            axis = np.arange(len(height))
            distance = np.minimum(axis, len(axis) - 1 - axis) * pitch
            t = np.clip(np.minimum.outer(distance, distance) / max(.012, 4*pitch), 0, 1)
            expected = measured * t**2 * (3 - 2*t)
            require(np.allclose(height, expected, rtol=0, atol=1e-7), f"{name}: scan heights vs metric field")
        mask = "water_mask" if meta["category"] == "pothole" else "opening_mask"
        require(field[mask].shape == height.shape, f"{name}: {mask}")
        if mask == "opening_mask":
            substrate = field["substrate_height"] if "substrate_height" in field else height
            require(substrate.shape == height.shape and (substrate[field[mask] > 0.5] < 0).any(), f"{name}: shaft opening")
            if "substrate_height" in field:
                require(np.all(height >= substrate - 2e-5), f"{name}: cover surface below its substrate")
    files = ["asset.glb", "surface.npz", "metadata.json"]
    record = dict(triangles=faces, geometry_sha256=geometry_hash, lowest_z_m=bottom, highest_z_m=top)
    if not run["mesh_only"]:
        count = ply(directory / "asset.ply", meta)
        require(meta["sh_degree"] == 1 and count > 0, f"{name}: SH degree/count")
        with np.load(directory / "splat_material.npz") as labels:
            for key in ("road", "part", "ambient_visibility"):
                require(labels[key].shape == (count,), f"{name}: {key} vs PLY")
            require(labels["part"].max() < len(labels["part_names"]), f"{name}: part index")
        record["gaussians"] = count
        files.extend(["asset.ply", "splat_material.npz"])
    if run["preview"]:
        for suffix in ("inspection", "reverse"):
            with Image.open(ROOT / "views" / f"{name}_{suffix}.jpg") as view:
                require(view.size == (840, 568), f"{name}: preview dimensions")
                view.verify()
    record["sha256"] = {f: sha(directory / f) for f in files}
    original = run["published"].get(name)
    if original:
        record["published_match"] = dict(triangles=faces == original["triangles"],
                                         glb_bytes=record["sha256"]["asset.glb"] == original["glb_sha256"])
        if not run["mesh_only"]:
            record["published_match"].update(gaussians=count == original["gaussians"],
                                            ply_bytes=record["sha256"]["asset.ply"] == original["ply_sha256"])
    return record


def main():
    run = read(ROOT / "run.json")
    for image, expected in run["reference_sha256"].items():
        require(sha(ROOT / "references" / image) == expected, f"{image}: changed reference")
    assets = ROOT / "assets"
    manifest = assets / "asset_manifest.json"
    metas = sorted(assets.glob("*/*/metadata.json"))
    if run["mesh_only"] and not manifest.exists() and not (ROOT / "checks.json").exists():
        rows = []
        for path in metas:
            row = read(path)
            rel = path.parent.relative_to(assets)
            row.update(glb=str(rel / "asset.glb"), surface=str(rel / "surface.npz"), metadata=str(rel / path.name))
            rows.append(row)
        write(manifest, rows)
    rows = read(manifest)
    expected = set(run["ids"])
    require(len(rows) == len(expected) and {r["id"] for r in rows} == expected, "manifest: requested IDs")
    require({p.parent.name for p in metas} == expected, "asset directories vs requested IDs")
    report = {}
    for row in rows:
        directory = assets / row["category"] / row["id"]
        for key in ("glb", "surface", "metadata") + (() if run["mesh_only"] else ("ply",)):
            require((assets / row[key]).is_file() and (assets / row[key]).parent == directory, f"{row['id']}: {key} path")
        report[row["id"]] = verify_asset(directory, row, run)
    output = ROOT / "checks.json"
    if output.exists():
        require(read(output) == report, "产物与首次构建时记录的结果不同")
    else:
        write(output, report)
    print(f"已核验 {len(report)} 件资产", flush=True)


if __name__ == "__main__":
    main()
