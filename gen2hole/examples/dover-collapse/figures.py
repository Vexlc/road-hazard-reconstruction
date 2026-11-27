"""Compare field exposures, independent source projections, and registered driving crops."""

import argparse
import json
from pathlib import Path
from subprocess import check_output
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from build import Surface
from cameras import HERE, photo, SIZE, calibrate


def plate(rows, out, width=600, height=280):
    gap = 12
    sheet = Image.new("RGB", (width * 2 + 3 * gap, (height + 40) * len(rows) + gap), "#f4f3f0")
    pen = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(
        check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True), 18
    )
    for j, row in enumerate(rows):
        for i, (im, title) in enumerate(row):
            im = ImageOps.contain(im.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
            x = gap + i * (width + gap)
            y = gap + j * (height + 40)
            sheet.paste(im, (x + (width - im.width) // 2, y + (height - im.height) // 2))
            pen.text((x, y + height + 3), title, font=font, fill="#242526")
    sheet.save(out, quality=93, subsampling=0)


def figures(models, views, drive, out):
    out.mkdir(parents=True, exist_ok=True)
    folder = models / "model/series/assets/pothole/P590"
    mesh = Surface(folder / "asset.glb")
    obs = Surface(folder / "observed.glb")
    off = np.asarray(json.loads((folder / "metadata.json").read_text())["local_origin_xyz_m"])
    cams, _ = calibrate()
    rows = [[], [], []]
    for j, cam in enumerate(cams):
        im = Image.fromarray(photo(j))
        rows[0].append((im, ["现场 A · 从路中线望向对侧", "现场 B · 从对侧回看断开的标线"][j]))
        gx, gy = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
        q = np.stack([gx, gy], -1)
        r = cam.rays(q)
        r /= np.linalg.norm(r, axis=-1)[..., None]
        eye = np.broadcast_to(cam.eye - off, r.shape)
        t, part, _ = mesh.cast(eye, r)
        d, _, _ = obs.cast(eye, r)
        support = (
            (part == 2)
            & np.isfinite(d)
            & (abs(np.where(np.isfinite(d), d, 0) - np.where(np.isfinite(t), t, 0)) < 0.07)
        )
        rgb = np.asarray(im.resize(tuple(SIZE))).copy()
        rgb[support] = 0.66 * rgb[support] + 0.34 * np.array([34, 223, 172])
        contours, _ = cv2.findContours(
            (part > 1).astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, (255, 154, 53), 1, cv2.LINE_AA)
        y, x = np.where(part > 1)
        region = (
            max(0, x.min() - 32),
            max(0, y.min() - 24),
            min(SIZE[0], x.max() + 33),
            min(SIZE[1], y.max() + 25),
        )
        rows[1].append(
            (Image.fromarray(rgb).crop(region), "橙线为模型首交轮廓，绿色为可见的双视角支撑区")
        )
        rows[2].append(
            (
                Image.open(views / ("P590-" + ["near", "opposite"][j] + ".jpg")),
                "独立视角渲染 · 原生高斯模型",
            )
        )
    plate(rows, out / "field.jpg", 540, 230)
    rows = [[], [], []]
    for side, name in [("front", "前摄"), ("back", "后摄")]:
        d = drive / side / "P590"
        rec = json.loads((d / "placement.json").read_text())
        rows[0].append(
            (
                Image.open(d / "drive.jpg"),
                f"PandaSet 109 {name} · 第 {rec['best_frame']} 帧 · 合成全景",
            )
        )
        y, x = np.where(np.asarray(Image.open(d / "part.png")) > 1)
        b = (max(0, x.min() - 25), max(0, y.min() - 32), x.max() + 26, y.max() + 33)
        for k, n, caption in [(1, "before", "原始车道"), (2, "after", "同一时刻加入下沉面板")]:
            rows[k].append((Image.open(d / (n + ".png")).crop(b), name + " · " + caption))
    plate(rows, out / "road.jpg", 580, 185)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "views", "drive", "out"):
        p.add_argument(k, type=Path)
    figures(**vars(p.parse_args()))
