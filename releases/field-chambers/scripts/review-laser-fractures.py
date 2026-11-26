"""Assemble measured intensity, untextured mesh and a two-camera driving comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PAPER = (242, 240, 234)
INK = (33, 42, 38)
MUTED = (76, 90, 80)
GREEN = (41, 89, 66)


def font(size, bold=False):
    path = Path("/usr/share/fonts/opentype/noto") / (
        "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    )
    return ImageFont.truetype(str(path), size, index=2)


def inset(sheet, img, box):
    x, y, w, h = box
    scaled = ImageOps.contain(
        img.convert("RGB"), (w, h), method=Image.Resampling.LANCZOS
    )
    sheet.paste(scaled, (x + (w - scaled.width) // 2, y + (h - scaled.height) // 2))


def road_crop(view):
    parts = np.load(view / "rays.npz")["part"]
    ys, xs = np.where(parts == 2)
    x, y = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    w = max(118, int(np.ptp(xs) * 1.4), int(np.ptp(ys) * 2.6))
    img = Image.open(view / "after.png")
    w = min(w, img.width, int(img.height * 1.85))
    h = round(w / 1.85)
    left = np.clip(x - w // 2, 0, img.width - w)
    top = np.clip(y - h // 2, 0, img.height - h)
    return img.crop((left, top, left + w, top + h))


def review(run, views, dest):
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "gen2hole/examples/laser-fractures/survey.json"
        ).read_text()
    )
    sheet = Image.new("RGB", (1420, 1520), PAPER)
    draw = ImageDraw.Draw(sheet)
    draw.text((36, 19), "激光巡检记录中的七处路面裂缝", font=font(37, True), fill=INK)
    draw.text(
        (38, 82),
        "FIND 原始强度与高程重建  /  PandaSet 109 同一世界位置的前后视合成",
        font=font(21),
        fill=MUTED,
    )
    cols = [(170, 272), (460, 331), (814, 280), (1114, 277)]
    for (x, w), title in zip(
        cols,
        [
            "激光回波强度",
            "无贴图网格 · 低角度光",
            "行车前视 · 局部放大",
            "驶过后回看 · 局部放大",
        ],
    ):
        draw.text((x, 144), title, font=font(19, True), fill=GREEN)
    labels = [
        "横穿防滑槽",
        "骨料旁折转",
        "宽斜裂",
        "标线边三岔缝",
        "并行双缝",
        "眼形裂口",
        "主缝与细支缝",
    ]
    for i, (rec, label) in enumerate(zip(spec["states"], labels)):
        key = rec["id"]
        top = 188 + i * 178
        draw.line((33, top - 8, 1387, top - 8), fill=(189, 193, 183))
        draw.text((36, top + 42), key, font=font(28, True), fill=INK)
        draw.text((36, top + 90), label, font=font(17), fill=MUTED)
        surface = run / "series/assets/pothole" / key / "surface.npz"
        img = Image.open(run / "series/references" / f"{key}.png").convert("RGB")
        with np.load(surface) as f:
            valid = np.uint8(f["observed_footprint"] > 0.5) * 255
        img = Image.composite(
            img, Image.new("RGB", img.size, PAPER), Image.fromarray(valid)
        )
        items = [
            img,
            Image.open(run / "evaluation" / key / "clay.jpg"),
            *(road_crop(views / c / key) for c in ("front", "back")),
        ]
        for (x, w), im in zip(cols, items):
            inset(sheet, im, (x, top, w, 158))
    draw.text(
        (36, 1454),
        "左列保持原始采样图比例。行车图取自实车合成帧，各自以裂缝为中心裁切。",
        font=font(18),
        fill=MUTED,
    )
    draw.text(
        (36, 1483),
        "扫描数据  Zhou、Canchila、Song  /  CC BY 4.0       行车背景  PandaSet  /  Scale AI、Hesai",
        font=font(16),
        fill=MUTED,
    )
    dest.parent.mkdir(exist_ok=True, parents=True)
    sheet.save(dest, quality=93, subsampling=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("run", "views", "dest"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    review(a.run, a.views, a.dest)
