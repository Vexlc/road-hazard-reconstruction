"""Compare native exports against the held-out opposing photograph and sample deck profiles."""

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cameras import HERE, S, SIZE, SCALE, photo
from surface import reconstruct, polygon
from build import Surface, xyz
from inspect_assets import inspect


def check(models, out):
    out.mkdir(parents=True, exist_ok=True)
    s = reconstruct()
    m = s["matches"]
    cams = s["cams"]

    def evidence(folder, mesh, meta, spec, run):
        origin = np.array(meta["local_origin_xyz_m"])
        observed = Surface(folder / "observed.glb")
        inside = s["inside"]
        for k in ["reverse_pipe_mask", "occluded_end_mask"]:
            inside &= ~polygon(m["px"][1], S[k])
        valid = m["hold"] & m["epipolar"] & inside
        flows = []
        for j, c in enumerate(cams):
            q = m["px"][j][valid]
            target = m["px"][1 - j][valid]
            r = c.rays(q)
            r = r / np.linalg.norm(r, axis=1)[:, None]
            eye = c.eye - origin
            d, part, _ = mesh.cast(np.broadcast_to(eye, r.shape), r)
            use = np.isfinite(d) & (part == 2)
            actual = c.eye + d[use, None] * r[use]
            opposite = cams[1 - j]
            error = np.linalg.norm((opposite.project(actual) - target[use]) * SCALE, axis=1)
            grade = np.linalg.norm((opposite.project(c.at(q[use])) - target[use]) * SCALE, axis=1)
            direct, src, _ = observed.cast(np.broadcast_to(eye, r.shape), r)
            near = use & np.isfinite(direct)
            flows.append(
                dict(
                    source=j,
                    heldout_inside=int(valid.sum()),
                    panel_first_hit=int(use.sum()),
                    model_error_median_px=float(np.median(error)),
                    model_error_p90_px=float(np.percentile(error, 90)),
                    grade_error_median_px=float(np.median(grade)),
                    grade_error_p90_px=float(np.percentile(grade, 90)),
                    tie_only_mesh_hits=int(near.sum()),
                )
            )
        with np.load(folder / "surface.npz") as field:
            z = field["height"]
            parts = field["first_part"]
            stereo = field["stereo_valid"]
            span = np.array([field["width"], field["length"]])
            assert (parts == 3).any()
        mask = parts == 2
        yy, xx = np.nonzero(mask)
        size = (np.array([xx.ptp(), yy.ptp()]) / (np.array(z.shape[::-1]) - 1)) * span
        xy = s["matches"]["xyz"][valid] - origin
        t, l, _ = mesh.cast(xyz(xy[:, :2], 1), np.broadcast_to([0, 0, -1.0], xy.shape))
        test = l == 2
        delta = np.abs(1 - t[test] - xy[test, 2])
        xx = np.linspace(-span[0] / 2, span[0] / 2, z.shape[1])
        yy = np.linspace(span[1] / 2, -span[1] / 2, z.shape[0])
        offsets = np.quantile(yy[np.where(mask)[0]], [0.22, 0.5, 0.78])
        fig, (ax, section) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        depth_map = ax.imshow(
            np.ma.masked_where(~mask, -100 * z),
            cmap="viridis",
            extent=[xx[0], xx[-1], yy[-1], yy[0]],
            vmin=0,
            origin="upper",
        )
        fig.colorbar(depth_map, ax=ax, shrink=.72, label='Depth below grade (cm)')
        ax.contour(xx, yy, stereo.astype(float), levels=[0.5], colors=["white"], linewidths=0.55)
        ax.set(
            xlabel="along road (m)",
            ylabel="across road (m)",
            title="Depth and two-view TIN support",
        )
        profiles = []
        ranges = []
        labels = []
        for off, color in zip(offsets, ["#1968a0", "#b34b2d", "#46834d"]):
            query = np.c_[xx, np.full_like(xx, off)]
            t, l, _ = mesh.cast(xyz(query, 1), np.broadcast_to([0, 0, -1.0], (len(query), 3)))
            zline = 1 - t
            zline[~np.isfinite(zline)] = np.nan
            zobs, _, _ = observed.cast(
                xyz(query, 1), np.broadcast_to([0, 0, -1.0], (len(query), 3))
            )
            ax.axhline(off, color=color, lw=1)
            section.plot(xx, zline * 100, color=color, label=f"y={off:+.2f} m")
            supported = np.isfinite(zobs) & (l == 2)
            section.scatter(xx[supported][::6], zline[supported][::6] * 100, s=5, color=color)
            ranges.append(zline)
            labels.append(l)
            profiles.append(
                dict(
                    offset_y_m=float(off),
                    max_drop_m=float(-np.nanmin(zline)),
                    support_samples=int(supported.sum()),
                )
            )
        section.set(
            xlabel="along road (m)",
            ylabel="height relative to grade (cm)",
            title="Three longitudinal profiles",
        )
        section.grid(alpha=0.22)
        section.legend()
        fig.savefig(out / "profiles.png", dpi=140)
        plt.close(fig)
        np.savez_compressed(
            out / "profiles.npz", x_m=xx, offset_y_m=offsets, height_m=ranges, part=labels
        )
        sensitivity = []
        for lane, prior in [(3.0, 650), (3.6, 650), (3.3, 520), (3.3, 825)]:
            ss = reconstruct(lane, prior)
            v = ss["matches"]["xyz"][ss["use"]]
            sensitivity.append(
                dict(
                    lane_m=lane,
                    focal_prior_px=prior,
                    training_depth_p95_m=float(np.percentile(-v[:, 2], 95)),
                )
            )
        return dict(
            raw_feature_pairs=len(inside),
            training_inliers=int(m["train"].sum()),
            trained_panel_ties=len(s["use"]),
            reserved_total=int(m["hold"].sum()),
            reserved_epipolar_inliers=int((m["hold"] & m["epipolar"]).sum()),
            reserved_panel_candidates=int((m["hold"] & inside).sum()),
            reserved_panel_inliers=int(valid.sum()),
            coordinate_units="conditional on lane scale",
            recovered_deck_extent_xy_m=size.tolist(),
            training_depth_p95_m=float(np.percentile(-m["xyz"][s["use"], 2], 95)),
            model_area_weighted_depth_p95_m=float(np.percentile(-z[mask], 95)),
            two_view_tin_area_fraction=float(stereo.sum() / mask.sum()),
            heldout_vertical_panel_hits=int(test.sum()),
            heldout_vertical_median_mm=float(np.median(delta) * 1000),
            heldout_vertical_p90_mm=float(np.percentile(delta, 90) * 1000),
            cross_view=flows,
            sensitivity=sensitivity,
            profiles=profiles,
        )

    inspect(models, out, evidence, ids={"P590"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("models", "out"):
        p.add_argument(n, type=Path)
    check(**vars(p.parse_args()))
