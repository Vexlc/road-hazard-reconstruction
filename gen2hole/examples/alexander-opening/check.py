"""Check casts against both original cameras, the interior step, road ties and image silhouettes."""

import argparse
from importlib import import_module
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from build import Surface, fit, SIZE, HERE
from inspect_assets import inspect
from mesh_transfer import RoadMesh

view = import_module("check-verge-chamber")


def check(models, out):
    cams, reg = fit()

    def verify(folder, surface, meta, spec, run):
        rows = {}
        a, b = reg["pairs"].transpose(1, 0, 2)
        out.mkdir(parents=True, exist_ok=True)

        def quant(v):
            return np.percentile(v, [10, 50, 90]).tolist() if len(v) else []

        road = RoadMesh(folder)
        yy, xx = np.indices((600, 800))
        rays_px = np.stack((2 * xx + 0.5, 2 * yy + 0.5), -1)
        obs = Surface(folder / "bounded.glb")
        for i, (cam, n) in enumerate(zip(cams, ["detail", "street"])):
            dirs = cam.rays(rays_px)
            rgb, part, dist = view.render(road, cam.eye, dirs)
            origin = np.broadcast_to(cam.eye, dirs.shape)
            bounded, _, _ = obs.cast(origin, dirs)
            src = np.asarray(
                Image.open(HERE / "references" / (n + ".jpg")).resize(
                    (800, 600), Image.Resampling.LANCZOS
                )
            )
            Image.fromarray(rgb).save(out / (n + "-render.jpg"), quality=94)
            overlay = src.copy()
            row = {}
            for key, code in [("rim", [5, 186, 187, 189]), ("lid", [185, 190])]:
                traced = np.zeros(part.shape, "u1")
                cv2.fillPoly(traced, [np.int32(np.rint((reg["traces"][i][key] - 0.5) / 2))], 1)
                test = np.isin(part, code)
                row[key + "_iou"] = float(
                    (traced.astype(bool) & test).sum() / (traced.astype(bool) | test).sum()
                )
                c, _ = cv2.findContours(
                    test.astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    overlay,
                    c,
                    -1,
                    (20, 236, 224) if key == "rim" else (245, 194, 51),
                    1,
                    cv2.LINE_AA,
                )
            c, _ = cv2.findContours(
                (part == 187).astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, c, -1, (94, 252, 93), 1, cv2.LINE_AA)
            row["step_pixels"] = int((part == 187).sum())
            row["marker_pixels"] = int((part == 188).sum())
            row["without_bounded_return"] = int(
                (np.isin(part, [186, 189]) & ~np.isfinite(bounded)).sum()
            )
            rows[n] = row
            Image.fromarray(overlay).save(out / (n + "-overlay.jpg"), quality=94)
            np.savez_compressed(
                out / (n + "-rays.npz"),
                first_part=part,
                optical_range_m=dist,
                bounded_range_m=bounded,
            )
        # Place local parts back in the scene and compare their ray intersections.
        for filename, offset, codes in [
            ("cover", np.r_[reg["lid_center"][:2], 0], (185, 190)),
            ("marker", np.asarray(meta["marker"]["base_origin_m"]), (188,)),
        ]:
            subset = Surface(folder / (filename + ".glb"))
            c = cams[1]
            dirs = c.rays(rays_px)
            o = np.broadcast_to(c.eye, dirs.shape)
            full, first, _ = surface.cast(o, dirs)
            g, pg, _ = subset.cast(o - offset, dirs)
            same = np.isin(first, codes) & np.isin(pg, codes)
            if same.sum() < 100:
                raise ValueError(filename)
            err = float(np.percentile(abs(full[same] - g[same]), 99))
            if err > 0.0001:
                raise ValueError((filename, err))
            rows[filename + "_replacement"] = dict(
                checked_pixels=int(same.sum()), range_error_p99_m=err
            )
        without = Surface(folder / "opening.glb")
        directions = np.array([[0.0, 0.0, -1.0]])
        origin = np.array([[*reg["lid_center"][:2], 2.0]])
        before = surface.cast(origin, directions)[1].item()
        after = without.cast(origin, directions)[1].item()
        if (before, after) != (185, 1):
            raise ValueError((before, after))
        rows["cover_removed_center"] = [before, after]
        for name in ("ground", "cover"):
            hold = reg[name] & ~reg["train"]
            w = (
                cams[0].plane(a[hold])
                if name == "ground"
                else cams[0].plane(
                    a[hold], reg["lid_center"] @ reg["lid_normal"], reg["lid_normal"]
                )
            )
            rows[name + "_reserved"] = dict(
                count=int(hold.sum()),
                reprojection_px=quant(np.linalg.norm(cams[1].project(w) - b[hold], axis=1)),
            )
        with np.load(folder / "stereo.npz") as data:
            for i, n in enumerate(("detail", "street")):
                mask = data["valid" + str(i)]
                y, x = np.where(mask)
                q = np.c_[x, y] + data["offset" + str(i)]
                dirs = cams[i].rays(q)
                d, part, _ = surface.cast(np.broadcast_to(cams[i].eye, dirs.shape), dirs)
                predicted = (cams[i].eye + dirs * d[:, None])[:, 2]
                discrepancy = abs(predicted - data["z" + str(i)][mask])
                rows[n].update(
                    two_way_pixels=len(x),
                    shell_depth_difference_m=quant(discrepancy[part == 186]),
                    shared_geometry_pixels=int((part == 186).sum()),
                )
        if min(row["rim_iou"] for row in (rows["detail"], rows["street"])) < 0.9:
            raise ValueError(rows)
        if min(row["step_pixels"] for row in (rows["detail"], rows["street"])) < 3:
            raise ValueError(rows)
        return rows

    inspect(models, out, verify, fit_tall=True, ids={"M585"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "out"):
        p.add_argument(k, type=Path)
    check(**vars(p.parse_args()))
