"""Show matched crops of the two cone instances on physical cameras and the Cycles oracle."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"paper": "#f4f4ef", "ink": "#263832"}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def text(canvas, xy, string, size):
    ImageDraw.Draw(canvas).text(xy, string, font=ImageFont.truetype(FONT, size, index=2), fill=COLORS["ink"])


def inside(canvas, img, xy, size):
    img = ImageOps.contain(img, size, Image.Resampling.LANCZOS)
    canvas.paste(img, (xy[0] + (size[0] - img.width) // 2, xy[1] + (size[1] - img.height) // 2))


def rect(mask, width, height, margin):
    yy, xx = np.nonzero(mask)
    return (
        max(0, xx.min() - margin),
        max(0, yy.min() - margin),
        min(width, xx.max() + margin + 1),
        min(height, yy.max() + margin + 1),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/traffic-cones")
    args = p.parse_args()
    root = args.views
    canvas = Image.new("RGB", (1200, 818), COLORS["paper"])
    text(canvas, (19, 5), "同一组交通锥在实车道路中的位置与外观", 30)
    for j, label in enumerate(["迁入全幅", "原始局部", "迁入同帧"]):
        text(canvas, (18 + j * 400, 58), label, 20)
    for i, (site, sensor, title) in enumerate(
        [
            ("sun", "image_03", "KITTI 阳光路段 · 右侧鱼眼"),
            ("urban", "image_03", "KITTI 背阴路段 · 右侧鱼眼"),
            ("audi", "rear_center", "Audi 驶过后 · 车尾鱼眼"),
        ]
    ):
        p = root / site / "C236-C237" / sensor
        row = json.loads((root / site / "index.json").read_text())["C236-C237"]["views"][sensor]
        top = 98 + i * 239
        before = Image.open(p / "before.png")
        after = Image.open(p / "after.png")
        mask = np.array(Image.open(p / "part.png")) > 0
        box = rect(mask | (np.load(p / "shadow.npy") > 0.13), *before.size, 16)
        full = Image.open(p / "full.jpg")
        frame = np.array(box) + np.tile(row["crop_xy"], 2)
        ImageDraw.Draw(full).rectangle(tuple(frame), outline="#f6db6d", width=3)
        for j, img in enumerate([full, before.crop(box), after.crop(box)]):
            inside(canvas, img, (8 + j * 400, top), (384, 185))
        text(canvas, (16, top + 188), f"{title}  ·  原始曝光 {row['frame']}", 19)
    canvas.save(root / "real-cameras.jpg", quality=94)
    sheet = Image.new("RGB", (1200, 770), COLORS["paper"])
    text(sheet, (18, 5), "220° 等距鱼眼 · Blender Cycles 与光线迁入", 30)
    for j, label in enumerate(["原宿主", "Cycles 原生加入两件", "同一曝光的光线迁入"]):
        text(sheet, (18 + 400 * j, 56), label, 20)
    for i, (cam, title) in enumerate(
        [("center", "中央视野"), ("left-edge", "鱼眼边缘，含光轴后方"), ("return-view", "驶过后的反向机位")]
    ):
        base = root / "cycles" / cam
        mask = np.load(base / "native/id.npy") >= 3
        mask |= np.load(base / "host/id.npy") == 2
        box = rect(mask, 512, 512, 20)
        y = 92 + 226 * i
        for j, path in enumerate(
            [base / "host/rgb.png", base / "native/rgb.png", root / "cycles-check" / cam / "after.png"]
        ):
            inside(sheet, Image.open(path).crop(box), (10 + 400 * j, y), (380, 171))
        text(sheet, (18, y + 177), title, 20)
    sheet.save(root / "cycles.jpg", quality=94)


if __name__ == "__main__":
    main()
