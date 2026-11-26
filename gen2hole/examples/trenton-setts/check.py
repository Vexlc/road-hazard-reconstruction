"""Compare metric field geometry against unused feature pairs and sample both road-camera views."""

import argparse
from pathlib import Path
import sys
from importlib import import_module
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from surface import HERE, S, labels, native, read_pairs, points_on_grade
from cameras import photograph
from stereo import pose

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from photo_mesh import Surface
from mesh_transfer import RoadMesh

view = import_module("check-verge-chamber")


def quant(x):
    return {
        name: float(v) for name, v in zip(("p10", "median", "p90"), np.percentile(x, [10, 50, 90]))
    }


def check(models, out):
    pairs = read_pairs()
    a, b = pairs[:, :2], pairs[:, 2:4]
    c0, c1, reg = pose(pairs)

    def evidence(folder, portable, meta, spec, run):
        offset = np.array(meta["model_offset_xyz_m"])
        field = Surface(folder / "field.glb")
        u, l, r, bottom = S["crop_bounds"]
        with np.load(folder / "stereo.npz") as m:
            lab = m["labels"]
            accepted = m["accepted"]
            grade = m["grade"]
            q0, q1 = m["atlas_minmax"]
            pitch = float(m["pitch"])
            eye = c0.eye - offset
            pix = c0.ray(a)
            d, part, _ = field.cast(np.broadcast_to(eye, pix.shape), pix)
            hit = c0.eye + pix * np.where(np.isfinite(d), d, 0)[:, None]
            recovered = c1.project(hit)
            err = np.linalg.norm(recovered - b, axis=1)
            flat = points_on_grade(c0, pix, grade)
            uv = np.rint((flat[:, :2] - q0) / [pitch, -pitch]).astype(int)
            hh, ww = lab.shape
            ii = uv[:, 1].clip(0, hh - 1)
            jj = uv[:, 0].clip(0, ww - 1)
            local = points_on_grade(c0, pix, grade, m["road_dz_m"][ii, jj])
            base_err = np.linalg.norm(c1.project(local) - b, axis=1)
            holds = reg["heldout"]
            marked = labels(a) > 1
            # Filter only by the triangulated pair's own reprojection agreement,
            # independently of the height field being tested.
            represented = marked[holds] & np.isfinite(d[holds])
            good = represented & (reg["errors"][holds] < 2.5)
            by_region = {}
            for k in (2, 3):
                use = (lab == k) & accepted
                mapped = np.count_nonzero(lab == k)
                dif = (m["source_dz_m"] - m["road_dz_m"])[use] * 1000
                by_region[str(k)] = dict(
                    mapped_area_m2=mapped * pitch**2,
                    stereo_cells=int(use.sum()),
                    stereo_fraction=float(use.sum() / mapped),
                    supported_relief_mm=quant(dif),
                    cycle_error_mm=quant(m["cycle_error_m"][use] * 1000),
                )
            h = holds[good]
            if len(h) < 20:
                raise ValueError("Insufficient retained cross-view checks")
            support = np.column_stack(
                (reg["xyz"][h], a[h], b[h], recovered[h], reg["errors"][h], err[h], base_err[h])
            )
            np.savez_compressed(
                out / "heldout.npz", pair_index=h, xyz_ref_context_predicted_px_errors=support
            )
            report = dict(
                source_rim_absolute_heldout_px=quant(abs(c0.heldout)),
                baseline_m=float(np.linalg.norm(c0.eye - c1.eye)),
                training_pairs=len(reg["train"]),
                reserved_pairs=len(holds),
                mapped_reserved=int(represented.sum()),
                checked_reserved=len(h),
                field_to_context_px=quant(err[h]),
                local_road_to_context_px=quant(base_err[h]),
                regions=by_region,
                grade_ties=len(m["sparse_grade_indices"]),
            )
            # Overlay the recovered native-pixel outline on the independently taken context photo.
            y, x = np.nonzero(lab > 1)
            rect = np.column_stack((q0[0] + x * pitch, q0[1] - y * pitch))
            for side, c in [("detail", c0), ("street", c1)]:
                source = photograph(side)
                plane = np.column_stack((rect, rect @ grade[:2] + grade[2]))
                ref = c0.project(plane)
                ray = c0.ray(ref)
                xyz = points_on_grade(c0, ray, grade, m["source_dz_m"][y, x])
                target = c.project(xyz)
                lo = np.maximum(np.floor(target.min(0) - [48, 55]).astype(int), [0, 0])
                hi = np.minimum(np.ceil(target.max(0) + [48, 55]).astype(int), S["size"])
                yy, xx = np.indices(tuple((hi - lo)[::-1]))
                rays = c.ray(np.stack((xx + lo[0], yy + lo[1]), axis=-1))
                _, mask, _ = field.cast(np.broadcast_to(c.eye - offset, rays.shape), rays)
                shot = source[lo[1] : hi[1], lo[0] : hi[0]].copy()
                over = shot.copy()
                for k, color in [(178, (100, 232, 242)), (179, (225, 190, 75))]:
                    cont, _ = cv2.findContours(
                        (mask == k).astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(over, cont, -1, color, 2, cv2.LINE_AA)
                cv2.imwrite(str(out / (side + "-source.jpg")), shot)
                cv2.imwrite(str(out / (side + "-outline.jpg")), over)
        with np.load(folder / "surface.npz") as surface:
            mask = surface["alpha"]
            interior_voids = int(np.count_nonzero(binary_fill_holes(mask) & ~mask))
            if interior_voids or not np.all(np.isfinite(surface["height"][mask])):
                raise ValueError(("Discontinuous portable road", interior_voids))
            report["interior_void_cells"] = interior_voids
            report["extent_m"] = [float(surface["width"]), float(surface["length"])]
        if not all(0 < v["stereo_fraction"] < 1 for v in report["regions"].values()):
            raise ValueError("Missing measured / inferred distinction")
        # Numerically intersect both the intact tongue and each separate stone surface.
        src = native([[240, 209], [714, 165], [827, 150], [1142, 138], [1140, 90]])
        directions = c0.ray(src)
        _, first, _ = field.cast(np.broadcast_to(eye, directions.shape), directions)
        report["source_witness_parts"] = first.tolist()
        print(report, flush=True)
        assert first.tolist() == [178, 1, 179, 179, 1]
        # Resolve the portable surface with more field of view than the source crop.
        r, _ = view.view_rays(
            np.array([0.61, -1.64, 1.11]), np.array([0.04, -0.04, 0.0]), 585, 860, 490
        )
        rgb, _, _ = view.render(RoadMesh(folder), np.array([0.61, -1.64, 1.11]), r)
        Image.fromarray(rgb).save(out / "vehicle-height.jpg", quality=95)
        return report

    inspect(models, out, evidence, ids={"P583"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "out"):
        p.add_argument(name, type=Path)
    check(**vars(p.parse_args()))
