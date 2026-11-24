"""Compare calibrated four-sample terrain texturing to 64 independent triangle hits per real sensor pixel."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from pavement_layout import PavementLayout
from pavement_texture import filtered, projected_axes
from road_geometry import save_json


def texture_hits(mesh, origin, rays, derivatives=None):
    packet = np.float32(np.c_[np.broadcast_to(origin, rays.shape), rays])
    hit = mesh.scene.cast_rays(o3d.core.Tensor(packet), nthreads=2)
    geom = hit["geometry_ids"].numpy()
    out = np.zeros((len(rays), 3))
    filtered_rgb = out.copy()
    good = np.zeros(len(rays), bool)
    for gid, (part, tri, uv, tex) in mesh.surfaces.items():
        if part != 2:
            continue
        select = geom == gid
        if not select.any():
            continue
        good |= select
        f = tri[hit["primitive_ids"].numpy()[select]]
        w = hit["primitive_uvs"].numpy()[select]
        point = (uv[f] * np.column_stack([1 - w.sum(1), w])[:, :, None]).sum(1)
        coords = (point * tex.shape[1::-1] - 0.5)[:, ::-1].T
        out[select] = np.column_stack(
            [
                map_coordinates(ch, coords, order=1, mode="nearest", output="f4")
                for ch in tex.transpose(2, 0, 1)
            ]
        )
        if derivatives is not None:
            xyz, levels = mesh.filtered_textures[gid]
            axes = projected_axes(
                xyz[f],
                uv[f],
                rays[select],
                hit["t_hit"].numpy()[select],
                [v[select] for v in derivatives],
                tex.shape[1::-1],
                0.5,
            )
            filtered_rgb[select] = filtered(levels, point, *axes)
    return good, out, filtered_rgb


def measure(mesh, pose, seat, ray_samples, ids, label):
    rows, cols = np.nonzero(ids == label)
    ok = (rows > 0) & (cols > 0) & (rows < ids.shape[0] - 1) & (cols < ids.shape[1] - 1)
    rows, cols = rows[ok], cols[ok]
    take = np.random.default_rng(2026).choice(len(rows), min(1300, len(rows)), replace=False)
    rows, cols = rows[take], cols[take]
    b = np.array(seat["basis_columns"])
    origin = (pose[:3, 3] - seat["center_world"]) @ b
    rotate = pose[:3, :3].T @ b
    base = ray_samples[:, rows, cols] @ rotate
    raw = []
    smooth = []
    valid = np.ones(len(take), bool)
    for i, rays in enumerate(base):
        d = [
            (ray_samples[i, rows, cols + 1] - ray_samples[i, rows, cols - 1]) @ rotate * 0.5,
            (ray_samples[i, rows + 1, cols] - ray_samples[i, rows - 1, cols]) @ rotate * 0.5,
        ]
        ok, point, antialiased = texture_hits(mesh, origin, rays, d)
        valid &= ok
        raw.append(point)
        smooth.append(antialiased)
    dense = []
    # Bilinear interpolation of the four registered bearings on this single sensor pixel.
    # Each new direction is normalized and intersected with the original triangles independently.
    for y in (np.arange(8) + 0.5) / 4 - 0.5:
        for x in (np.arange(8) + 0.5) / 4 - 0.5:
            rays = (1 - y) * ((1 - x) * base[0] + x * base[1]) + y * ((1 - x) * base[2] + x * base[3])
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            ok, signal, _ = texture_hits(mesh, origin, rays)
            valid &= ok
            dense.append(signal)
    if valid.sum() < 80:
        raise ValueError("Insufficient continuously visible texture")
    ref = np.mean(dense, axis=0)[valid]

    def error(rgb):
        return float(np.abs(np.mean(rgb, axis=0)[valid] - ref).mean())

    return dict(samples=int(valid.sum()), bilinear_mae=error(raw), filtered_mae=error(smooth))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/measured-pavement")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    model = PavementLayout(ROOT / "gen2hole/examples/measured-pavement/layout.json", a.assets)
    result = {}
    for site in ["kitti", "suburb", "audi"]:
        seats = json.loads((a.views / site / "anchor.json").read_text())["instances"]
        for f in sorted((a.views / site / model.key).glob("*/view.npz")):
            ids = np.asarray(Image.open(f.parent / "instance.png"))
            with np.load(f) as v:
                rays, pose = v["rays_camera"], v["camera_to_world"]
            for i, seat in enumerate(seats, 1):
                key = f"{site}/{f.parent.name}/{seat['asset']}"
                result[key] = measure(model.meshes[seat["asset"]], pose, seat["anchor"], rays, ids, i)
                print(key, result[key], flush=True)

    def weighted(field):
        return sum(row[field] * row["samples"] for row in result.values()) / sum(
            row["samples"] for row in result.values()
        )

    totals = {k: weighted(k) for k in ("filtered_mae", "bilinear_mae")}
    if totals["filtered_mae"] >= totals["bilinear_mae"]:
        raise ValueError("No measured antialiasing improvement")
    save_json(a.out, dict(comparisons=result, weighted=totals))


if __name__ == "__main__":
    main()
