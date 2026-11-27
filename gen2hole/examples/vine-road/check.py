"""Check the native GLB, splats, support mask and simulated ray tables, and plot their extent."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData
from build import ASSET, Surface
from road_geometry import glb_triangles


def check(model, replay, out):
    out.mkdir(parents=True, exist_ok=True)
    asset = model / ASSET
    meta = json.loads((asset / "metadata.json").read_text())
    with np.load(asset / "surface.npz") as a:
        field = dict(a)
    mesh = Surface(asset / "asset.glb")
    verts = PlyData.read(str(asset / "asset.ply"))["vertex"]
    xyz = np.stack([verts[k][::41] for k in ("x", "y", "z")], axis=1).astype("f4")
    nearest = mesh.scene.compute_distance(o3d.core.Tensor(xyz), nthreads=2).numpy()
    assert len(verts) == meta["gaussians"] and np.max(nearest) < 0.001
    names = meta["included_panels"]
    strict_parts = {}
    for name, v, f in glb_triangles(asset / "supported.glb", names=True):
        edge = np.linalg.norm(v[f] - np.roll(v[f], 1, axis=1), axis=2)
        assert edge.max() < 1.50001
        strict_parts[name] = len(f)
    x, y = field["x"], field["y"]
    m = field["installation_mask"]
    support = field["stereo_valid"]
    assert np.all(~support | m) and np.array_equal(field["plate_id"] > 0, m)
    sample = m.nonzero()
    ii, jj = sample[0][::103], sample[1][::103]
    points = np.c_[x[jj], y[ii], np.full(len(ii), 8.0)]
    rays = np.tile([0.0, 0.0, -1.0], (len(ii), 1))
    d, _, _ = mesh.cast(points, rays)
    np.testing.assert_allclose(8 - d, field["height"][ii, jj], atol=1e-4)
    pose = json.loads((replay / "camera.json").read_text())
    clip = np.load(replay / "rays.npz")
    for kd, ki in [
        ("model_range_m", "model_part"),
        ("supported_range_m", "supported_part"),
        ("camera_range_m", "camera_part"),
    ]:
        assert np.array_equal(np.isfinite(clip[kd]), clip[ki] > 0)
    np.testing.assert_allclose(np.linalg.norm(clip["beam_dirs"], axis=-1), 1, atol=1e-6)
    for i, frame in enumerate(pose["frames"]):
        # Independently recast the panel-only scene on the stored beam directions.
        r = clip["beam_dirs"][i]
        origin = np.broadcast_to(frame["eye_local_m"], r.shape)
        d, _, _ = mesh.cast(origin, r)
        hits = clip["model_part"][i] >= 2
        np.testing.assert_allclose(d[hits], clip["model_range_m"][i][hits], atol=2e-4)
    grade_ok = support & (field["before_radius_m"] < 2.5)
    depth = field["before_grade"] - field["supported_height"]
    pct = np.percentile(depth[grade_ok], [50, 95])
    np.testing.assert_allclose(pct, meta["drop_from_before_m_p50_p95"], atol=0.00001)
    report = dict(
        plates=len(names),
        mesh_facets=sum(meta["geometry_parts"].values()),
        strict_facets=sum(strict_parts.values()),
        splats=len(verts),
        sampled_gaussians=len(xyz),
        max_splat_distance_to_mesh_m=float(nearest.max()),
        full_top_area_m2=round(m.sum() * 0.01, 2),
        strict_top_area_m2=round(support.sum() * 0.01, 2),
        strict_top_fraction=float(support.sum() / m.sum()),
        valid_grade_cells=int(grade_ok.sum()),
        strict_drop_m_p50_p95=pct.tolist(),
        camera_frames=len(pose["frames"]),
        lidar_rays=int(clip["model_part"].size),
        mesh_lidar_hits=int((clip["model_part"] >= 2).sum()),
        strict_lidar_hits=int((clip["supported_part"] >= 2).sum()),
        same_plate_both_lidar_hits=int(
            ((clip["model_part"] >= 2) & (clip["model_part"] == clip["supported_part"])).sum()
        ),
        third_camera=meta["third_view_holdout"],
        intact_control=meta["intact_road_holdout"],
    )
    (out / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(field, pose, replay, out)
    print(json.dumps(report, indent=2))


def plot(field, pose, replay, out):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    x, y = field["x"], field["y"]
    m = field["installation_mask"]
    z = field["height"]
    before = field["before_grade"]
    fig = plt.figure(figsize=(12, 6.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
    left = gs[:, 0].subgridspec(1, 2)
    extent = [x[0] - 0.05, x[-1] + 0.05, y[-1] - 0.05, y[0] + 0.05]
    panels = [
        np.where(m, before - z, np.nan),
        np.where(field["stereo_valid"], before - field["supported_height"], np.nan),
    ]
    for i, im in enumerate(panels):
        ax = fig.add_subplot(left[i])
        artist = ax.imshow(
            im, extent=extent, vmin=-0.6, vmax=3, cmap="viridis", interpolation="nearest"
        )
        ax.set_facecolor("#e6e9eb")
        ax.set_title(
            ["01   Traced plates", "02   Short-edge TIN"][i], loc="left", fontweight="bold"
        )
        ax.set_xlabel("East (m)")
        if i == 0:
            ax.set_ylabel("North (m)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(
        artist,
        ax=fig.axes[:2],
        location="bottom",
        shrink=0.89,
        pad=0.04,
        label="2011 ground minus reconstructed surface (m)",
    )
    ax = fig.add_subplot(gs[0, 1])
    for pos, color in [(-1.5, "#16877b"), (1.5, "#ab5426")]:
        j = np.argmin(abs(x - pos))
        k = m[:, j]
        ax.plot(
            y, np.where(k, z[:, j], np.nan), color=color, lw=1.7, label=f"Plate, E = {x[j]:+.1f} m"
        )
        ax.plot(y, np.where(k, before[:, j], np.nan), color=color, lw=1, ls=":")
    ax.set(title="03   Two longitudinal sections", xlabel="North (m)", ylabel="Local elevation (m)")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.03, "Dotted lines: pre-earthquake ground", transform=ax.transAxes, fontsize=8)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(Image.open(replay / "frame-08.jpg"))
    ax.set_axis_off()
    ax.set_title("04   Approach from the north, camera at 1.55 m", loc="left")
    fig.suptitle(
        "Vine Road  /  Displaced carriageway after the 2018 Alaska earthquake",
        x=0.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out / "overview.png", dpi=155)
    plt.close(fig)
    with np.load(replay / "rays.npz") as r:
        fig, axes = plt.subplots(
            2, 3, figsize=(12, 4.8), layout="constrained", gridspec_kw={"height_ratios": [1.65, 1]}
        )
        for j, i in enumerate((0, 6, 11)):
            axes[0, j].imshow(Image.open(replay / f"frame-{i:02}.jpg"))
            axes[0, j].set_axis_off()
            axes[0, j].set_title(f"Camera   /   station {i+1:02}", loc="left", fontweight="bold")
            scan = np.where(r["model_part"][i] >= 2, r["model_range_m"][i], np.nan)
            ax = axes[1, j]
            im = ax.imshow(
                scan,
                aspect="auto",
                origin="lower",
                extent=[-42, 42, -23, 3],
                cmap="cividis_r",
                vmin=3,
                vmax=70,
                interpolation="nearest",
            )
            ax.set_facecolor("#e6e9eb")
            ax.set_xlabel("Beam azimuth (deg)")
            if j == 0:
                ax.set_ylabel("Elevation (deg)")
        fig.colorbar(
            im,
            ax=axes[1, :],
            location="bottom",
            shrink=0.45,
            pad=0.07,
            label="Range to first visible modeled plate (m)",
        )
        fig.suptitle(
            "North approach  /  Synthetic forward camera and 32-channel scan",
            x=0.01,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        fig.savefig(out / "approach.png", dpi=145)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "replay", "out"):
        p.add_argument(name, type=Path)
    check(**vars(p.parse_args()))
