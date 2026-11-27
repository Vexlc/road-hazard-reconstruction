"""Render source evidence, observed vs completed texture, and matched calibrated installations."""

import argparse, json
from pathlib import Path
from subprocess import check_output
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from camera import photo, S
from build import KEY


def plate(rows, out, width, height):
    font = ImageFont.truetype(
        check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True), 18
    )
    gap = 12
    board = Image.new(
        "RGB", (gap + (width + gap) * len(rows[0]), gap + (height + 38) * len(rows)), "#f3f3ef"
    )
    for y, row in enumerate(rows):
        for x, (im, title) in enumerate(row):
            im = ImageOps.contain(im.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
            left = gap + x * (width + gap)
            top = gap + y * (height + 38)
            board.paste(im, (left + (width - im.width) // 2, top + (height - im.height) // 2))
            ImageDraw.Draw(board).text((left, top + height + 3), title, font=font, fill="#282c2d")
    board.save(out, quality=95, subsampling=0)


def detail(im, mask, margin=18):
    y, x = np.where(mask)
    return im.crop(
        (
            max(0, x.min() - margin),
            max(0, y.min() - margin),
            min(im.width, x.max() + margin + 1),
            min(im.height, y.max() + margin + 1),
        )
    )


def figures(models, views, drive, corridor, out):
    out.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(photo())
    obs = np.load(models / "observations.npz")
    marked = im.copy()
    for i in [1, 2]:
        pts = obs["hold" + str(i)]
        for x, y in pts:
            ImageDraw.Draw(marked).ellipse((x - 3, y - 3, x + 3, y + 3), fill="#5fffc4")
    raw = Image.open(models / "casting.png").convert("RGB")
    n = raw.width
    yy, xx = np.indices((n, n))
    disc = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) <= n / 2
    tex = np.array(raw)
    mask = obs["observed_top"]
    tex[~mask & disc] = np.rint(0.45 * tex[~mask & disc] + 0.55 * np.array([237, 163, 51]))
    tex[~disc] = [244, 244, 241]
    plate(
        [
            [
                (im, "2012 年华盛顿第四街 · Payton Chung"),
                (Image.fromarray(tex), "旋转校准后的盖面，橙色为补全区"),
            ],
            [
                (marked.crop((270, 2120, 1350, 2592)), "近盖局部 · 绿色为留出边缘点"),
                (marked.crop((790, 917, 1120, 1040)), "远盖局部 · 反光和视角限制了可辨细节"),
            ],
        ],
        out / "field.jpg",
        400,
        269,
    )
    rows = [[], [], []]
    for state, label in zip(S["states"], ["盖好", "移开约 25 cm", "盖板缺失"]):
        rows[0].append((Image.open(views / state / (KEY + "-opposite.jpg")), label + " · 高斯近景"))
        for j, side in enumerate(["front", "back"], 1):
            d = drive / state / side / KEY
            im = Image.open(d / "after.png")
            part = np.asarray(Image.open(d / "part.png"))
            rows[j].append(
                (detail(im, part > 1, 14), ("前摄" if j == 1 else "后摄") + " · " + label)
            )
    plate(rows, out / "states.jpg", 300, 155)
    rows = [[], [], []]
    for side, label in [("front", "前摄"), ("back", "后摄")]:
        d = corridor / side
        im = Image.open(d / "both.jpg")
        rows[0].append((im, label + " · 同一帧内的两只凸起井盖"))
        packet = np.load(d / "both.npz")
        x, y = packet["crop_xy"]
        ids = packet["instance"]
        h, w = ids.shape
        crop = im.crop((x, y, x + w, y + h))
        for idx in [1, 2]:
            rows[idx].append(
                (
                    detail(crop, (ids == idx) & (packet["part"] > 1), 12),
                    label + (" · 第一只井盖" if idx == 1 else " · 第二只井盖"),
                )
            )
    plate(rows, out / "corridor.jpg", 470, 180)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "views", "drive", "corridor", "out"):
        p.add_argument(name, type=Path)
    figures(**vars(p.parse_args()))
