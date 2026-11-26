"""Compare exported meshes to reserved XYZ returns and examine three longitudinal sections."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from inspect_assets import inspect
from photo_mesh import xyz


def check(models, out, survey=HERE / "survey.json"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [row["id"] for row in json.loads(survey.read_text())["states"]]
    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 10), constrained_layout=True)

    def evidence(folder, surface, meta, spec, run):
        key = folder.name
        with np.load(run / "evaluation" / key / "survey.npz") as survey:
            local = (survey["xyz"] - meta["source_origin_xyz"]) @ np.array(
                meta["source_local_basis_columns"]
            )
            defect = survey["defect"].astype(bool)
            hold = survey["reserved"].astype(bool)
        points = local[hold & defect]
        ray = np.broadcast_to([0.0, 0.0, -1.0], points.shape)
        t, parts, _ = surface.cast(xyz(points[:, :2], 1.0), ray)
        hit = np.isfinite(t)
        delta = (1 - t[hit] - points[hit, 2]) * 1000
        closest = (
            surface.scene.compute_distance(
                o3d.core.Tensor(np.ascontiguousarray(points, dtype="f4"))
            ).numpy()
            * 1000
        )
        dims = sorted(cv2.minAreaRect(local[defect, :2].astype("f4"))[1], reverse=True)
        assert hit.mean() > 0.86 and np.isfinite(closest).all()
        with np.load(folder / "surface.npz") as field:
            support = field["observed_footprint"].astype(bool)
            bed = field["defect_mask"].astype(bool)
            alpha = field["alpha"]
            height = field["height"]
            span = np.array([field["width"], field["length"]])
            assert np.all(alpha[~support] == 0) and np.all(height[~support] == 0)
        yy, xx = np.indices(support.shape)
        xy = (
            (np.stack([xx, yy], axis=-1) / (np.array(support.shape[::-1]) - 1) - 0.5)
            * span
            * [1, -1]
        )
        # Probe centres of unsupported grid cells away from the boundary in the actual exported mesh.
        outside = ~support & (cv2.distanceTransform(np.uint8(~support), cv2.DIST_L2, 5) > 2)
        test = xy[outside][::13]
        external = span * np.array([[-0.51, -0.51], [-0.51, 0.51], [0.51, -0.51], [0.51, 0.51]])
        internal_test_count = len(test)
        test = np.vstack([test, external])
        missing, _, _ = surface.cast(xyz(test, 1.0), np.broadcast_to([0, 0, -1.0], (len(test), 3)))
        assert np.isinf(missing).all()
        # Same physical XY coordinates for colour plots, analytical probes and ground-truth rays.
        offsets = np.quantile(local[defect & ~hold, 1], [0.2, 0.5, 0.8])
        line_x = np.arange(-span[0] / 2 + 0.002, span[0] / 2, 0.005)
        profiles = []
        heights, parts_by_track = [], []
        ax, section = axes[ids.index(key)]
        ax.set_title(f"{key}  |  observed footprint", loc="left")
        masked = np.ma.masked_where(~support, -height * 100)
        image = ax.imshow(
            masked,
            extent=[-span[0] / 2, span[0] / 2, -span[1] / 2, span[1] / 2],
            cmap="cividis",
            vmin=0,
            vmax=max(5, np.quantile(-height[bed] * 100, 0.99)),
            origin="upper",
            aspect="equal",
        )
        ax.set_facecolor("#e8e9e7")
        fig.colorbar(image, ax=ax, label="depth (cm)", shrink=0.7)
        section.set_title(f"{key}  |  ray sections and reserved returns", loc="left")
        for y, col in zip(offsets, ["#197498", "#c16427", "#4a8064"]):
            p = np.column_stack([line_x, np.full_like(line_x, y)])
            t, part, _ = surface.cast(xyz(p, 1.0), np.broadcast_to([0, 0, -1.0], (len(p), 3)))
            sampled = t - 1.0
            valid = np.isfinite(sampled)
            sampled[~valid] = np.nan
            heights.append(-sampled)
            parts_by_track.append(part)
            pit = part == 2
            assert pit.any()
            profiles.append(
                dict(
                    offset_m=float(y),
                    first_hit_samples=int(valid.sum()),
                    unobserved_samples=int((~valid).sum()),
                    pit_samples=int(pit.sum()),
                    pit_depth_p95_m=float(np.percentile(sampled[pit], 95)),
                )
            )
            ax.plot(line_x[valid], np.full(valid.sum(), y), ".", color=col, ms=1)
            section.plot(line_x, -sampled * 100, color=col, lw=1.5, label=f"y={y:+.2f} m")
            selected = points[abs(points[:, 1] - y) < 0.012]
            section.scatter(
                selected[:, 0],
                selected[:, 2] * 100,
                s=9,
                facecolors="none",
                edgecolors=col,
                linewidths=0.8,
            )
        ax.set_ylabel("transverse (m)")
        ax.set_xlabel("longitudinal (m)")
        section.set_xlabel("longitudinal (m)")
        section.set_ylabel("height (cm)")
        section.grid(alpha=0.24)
        section.legend(fontsize=7, ncol=3)
        np.savez_compressed(
            out / (key + "-sections.npz"),
            x_m=line_x,
            offset_y_m=offsets,
            z_m=np.stack(heights),
            first_part=np.stack(parts_by_track),
        )
        return dict(
            source_points=len(local),
            reserved_all=int(hold.sum()),
            reserved_pit_points=len(points),
            vertical_hits=int(hit.sum()),
            misses_outside_capture=int((~hit).sum()),
            vertical_first_hits_on_bed=int((parts == 2).sum()),
            heldout_vertical_mae_mm=float(np.mean(abs(delta))),
            heldout_vertical_p90_mm=float(np.percentile(abs(delta), 90)),
            heldout_closest_surface_p90_mm=float(np.percentile(closest, 90)),
            observed_label_bounds_m=dims,
            source_depth_p95_m=float(np.percentile(-local[defect, 2], 95)),
            unsupported_grid_ray_misses=internal_test_count,
            external_misses=len(external),
            profiles=profiles,
        )

    out.mkdir(parents=True, exist_ok=True)
    inspect(models, out, evidence, ids=set(ids))
    fig.savefig(out / "wheel-sections.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--survey", type=Path, default=HERE / "survey.json")
    check(**vars(parser.parse_args()))
