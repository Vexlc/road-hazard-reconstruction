"""Draw matched vehicle crops, native two-view comparisons and a first-hit beam comparison."""

import argparse
from pathlib import Path
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from casting import S
from build import KEY


def figures(models, drive, views, lidar, out):
    out.mkdir(parents=True, exist_ok=True)
    states = list(S["relative_grate_poses"])
    native = Image.new("RGB", (1050, 550), (39, 41, 40))
    car = native.copy()
    for i, state in enumerate(states):
        for j, side in enumerate(["near", "opposite"]):
            im = Image.open(views / state / (KEY + "-" + side + ".jpg")).resize(
                (346, 249), Image.Resampling.LANCZOS
            )
            native.paste(im, (350 * i + 2, 275 * j + 24))
            ImageDraw.Draw(native).text(
                (350 * i + 5, 275 * j + 5), state + "  /  " + side, fill="white"
            )
    native.save(out / "native.jpg", quality=94)
    for j, side in enumerate(("front", "back")):
        runs = [drive / s / side / KEY for s in states]
        frames = [json.loads((r / "placement.json").read_text())["best_frame"] for r in runs]
        if len(set(frames)) != 1:
            raise ValueError("Representative frames differ")
        offsets = []
        limits = []
        windows = []
        for r, frame in zip(runs, frames):
            record = json.loads((r / "placement.json").read_text())
            off = next(e["crop_xy"] for e in record["exposures"] if e["frame"] == frame)
            offsets.append(off)
            yy, xx = np.nonzero(np.load(r / "rays.npz")["part"])
            limits.append(
                [xx.min() + off[0], yy.min() + off[1], xx.max() + off[0], yy.max() + off[1]]
            )
            w, h = Image.open(r / "after.png").size
            windows.append([*off, w + off[0], h + off[1]])
        limits = np.array(limits)
        windows = np.array(windows)
        x0, y0 = limits[:, :2].min(0)
        x1, y1 = limits[:, 2:].max(0)
        width = max(124, max(x1 - x0 + 1, (y1 - y0 + 1) * 1.42) * 1.55)
        height = width * 249 / 346
        a, b = windows[:, :2].max(0)
        c, d = windows[:, 2:].min(0)
        cx = np.clip((x0 + x1) / 2, a + width / 2, c - width / 2)
        cy = np.clip((y0 + y1) / 2, b + height / 2, d - height / 2)
        world = np.rint([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]).astype(
            int
        )
        original = None
        for i, (s, r, off) in enumerate(zip(states, runs, offsets)):
            crop = tuple(world - np.tile(off, 2))
            before = np.asarray(Image.open(r / "before.png").crop(crop))
            if original is not None and not np.array_equal(original, before):
                raise ValueError("Host pixels differ")
            original = before
            im = Image.open(r / "after.png").crop(crop).resize((346, 249), Image.Resampling.LANCZOS)
            car.paste(im, (350 * i + 2, 275 * j + 24))
            ImageDraw.Draw(car).text(
                (350 * i + 5, 275 * j + 5), f"{s}  /  {side}  /  {frames[i]:02}", fill="white"
            )
    car.save(out / "drive.jpg", quality=94)
    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.4), constrained_layout=True, sharex=True, sharey=True
    )
    for k, (ax, state) in enumerate(zip(axes, states)):
        field = np.load(models / state / "model/series/assets/missing/M589/surface.npz")
        ax.imshow(
            field["first_part"] == 9,
            extent=[-0.82, 0.82, -0.48, 0.48],
            origin="upper",
            alpha=0.36,
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        points = [[], []]
        for p in sorted(lidar.glob("*.npz")):
            with np.load(p) as data:
                crossing = (
                    data["origin_local_m"]
                    + data["grade_plane_range_m"][:, None] * data["unit_ray_local"]
                )
                for group, mask in zip(
                    points, [data["inserted_part"][k] == 9, data["unresolved"][k]]
                ):
                    group.extend(crossing[mask, :2])
        for pts, label, color in zip(
            points, ["painted face", "scenario surface"], ["#bc612c", "#2b7181"]
        ):
            xx, yy = np.asarray(pts).T
            ax.scatter(xx, yy, s=4, alpha=0.7, label=label, c=color)
        ax.set(
            title=state,
            xlabel="local X (m)",
            aspect="equal",
            xlim=(-0.35, 0.82),
            ylim=(-0.45, 0.38),
        )
        if k == 0:
            ax.set_ylabel("local Y (m)")
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=7)
    fig.savefig(out / "beams.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "drive", "views", "lidar", "out"):
        p.add_argument(k, type=Path)
    figures(**vars(p.parse_args()))
