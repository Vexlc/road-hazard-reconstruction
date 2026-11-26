"""Check traced parts, unused edge observations, road sections and camera-scale sensitivity."""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

from camera import Camera, HERE, load

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from photo_mesh import xyz

sys.path.insert(0, str(HERE.parent / "whites-hill"))
from render import Renderer


def check(models, out):
    s, cam = load()

    def field(folder, surf, meta, spec, run):
        y, x = np.indices((480, 640))
        ray = cam.rays(np.stack((x, y), axis=-1))
        photo, part, dist = Renderer(folder / "inspection.glb").render(cam.eye, ray)
        portable, typ, alt = Renderer(folder / "asset.glb").render(cam.eye, ray)
        valid = np.isfinite(dist)
        if (
            not np.array_equal(valid, np.isfinite(alt))
            or np.max(abs(dist[valid] - alt[valid])) > 0.0003
        ):
            raise ValueError("Portable geometry differs")
        checks = []
        for u, v, p in s["appearance_probes"]:
            checks.append(dict(pixel=[u, v], reference=p, mesh=int(part[v, u])))
            if part[v, u] != p:
                raise ValueError(checks[-1])
        edges = []
        for upper, toe in s["heldout_front_top_toe_px"]:
            col = toe[0]
            ys = np.arange(upper[1] - 6, toe[1] + 9, 0.1)
            r = cam.rays(np.c_[np.full(ys.shape, col), ys])
            _, k, _ = surf.cast(np.broadcast_to(cam.eye, r.shape), r)
            face = ys[k == 14]
            if not len(face):
                raise ValueError("Missing face")
            edges.append(
                dict(
                    x_px=col,
                    observed_toe_y_px=toe[1],
                    rendered_toe_y_px=float(face.max()),
                    error_px=float(face.max() - toe[1]),
                )
            )
        if max(abs(e["error_px"]) for e in edges) > 2.5:
            raise ValueError(edges)
        sweep = []
        for focal in (440, 520, 600, 680):
            for scale in (0.5, 0.75):
                c = Camera(s, focal, scale)
                rises = {}
                lengths = {}
                for rec in s["sheets"]:
                    h = c.lips(rec["crest"], rec["toe"])
                    pos = c.plane(rec["crest"], h)
                    rises[rec["name"]] = float(h.max())
                    lengths[rec["name"]] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
                sweep.append(
                    dict(
                        focal_px=focal,
                        assumed_cone_height_m=scale,
                        heldout_cone_error_px=c.cone_error_px[2],
                        max_station_rise_m=rises,
                        crest_length_m=lengths,
                    )
                )
        sections = {}
        profiles = {}
        for rec in s["sheets"]:
            j = int(np.argmax(cam.lips(rec["crest"], rec["toe"])))
            h = cam.lips(rec["crest"][j], rec["toe"][j])
            front = cam.plane(rec["crest"][j], h)[:2]
            back = cam.plane(rec["return_to_grade"][j])[:2]
            delta = back - front
            length = np.linalg.norm(delta)
            axis = delta / length
            t = np.arange(-0.30, length + 0.30, 0.006)
            uv = front + t[:, None] * axis
            p = xyz(uv, 1.0)
            d, k, _ = surf.cast(p, np.broadcast_to([0.0, 0.0, -1.0], p.shape))
            z = np.where(np.isfinite(d), 1 - d, np.nan)
            on = k == 13
            if not on.any() or np.nanmax(z[on]) < h - 0.025:
                raise ValueError("No supported elevated profile")
            name = rec["name"]
            sections[name + "_distance_m"] = t
            sections[name + "_z_m"] = z
            sections[name + "_part"] = k
            sections[name + "_xy_m"] = uv
            profiles[name] = dict(
                max_z_m=float(np.nanmax(z)),
                unknown_samples=int((k == 0).sum()),
                samples=len(k),
                crest_station=j,
            )
        source = np.array(Image.open(HERE / "references/2559905.jpg").convert("RGB"))
        Image.fromarray(photo).save(out / "source-view.png")
        Image.fromarray(portable).save(out / "transfer-view.png")
        np.savez_compressed(
            out / "source-rays.npz", range_m=dist, first_part=part, portable_part=typ
        )
        np.savez_compressed(out / "sections.npz", **sections)
        return dict(
            appearance_probes=checks,
            unused_front_edge_pixels=edges,
            cone_errors_px=cam.cone_error_px,
            building_line_errors_px=cam.upright_error_px,
            scale_sweep=sweep,
            sections=profiles,
            source_appearance_mae_srgb=float(abs(photo.astype(float) - source)[part > 1].mean()),
            same_geometry_range_p99_mm=float(
                1000 * np.percentile(abs(dist[valid] - alt[valid]), 99)
            ),
        )

    out.mkdir(parents=True, exist_ok=True)
    inspect(models, out, field, ids={"P581"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    check(**vars(p.parse_args()))
