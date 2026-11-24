"""Make a matched physical-camera plate and chronological front/rear replays from saved checkpoints."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
INK = "#263633"
PAPER = "#f4f4ee"


def label(draw, xy, text, size, color=INK):
    draw.text(xy, text, fill=color, font=ImageFont.truetype(FONT, size, index=2))


def panel(canvas, source, rect):
    box = rect[2] - rect[0], rect[3] - rect[1]
    picture = ImageOps.contain(source, box, Image.Resampling.LANCZOS)
    canvas.paste(picture, (rect[0] + (box[0] - picture.width) // 2, rect[1] + (box[1] - picture.height) // 2))


def physical(root, out):
    cases = [
        ("hydrant", "H234", "image_03", "消防栓 · 日照"),
        ("hydrant", "H235", "image_03", "消防栓 · 风化涂层"),
        ("barrier", "B232", "image_03", "棋盘涂装隔离墩 · 侧鱼眼"),
        ("audi", "B233", "rear_center", "吊环隔离墩 · 车尾鱼眼"),
    ]
    canvas = Image.new("RGB", (1180, 625), PAPER)
    draw = ImageDraw.Draw(canvas)
    label(draw, (20, 9), "实车同一帧里的原路面与迁入结果", 30)
    import json

    for i, (site, key, camera, title) in enumerate(cases):
        path = root / site / key / camera
        frame = json.loads((root / site / "index.json").read_text())[key]["views"][camera]["frame"]
        mask = np.asarray(Image.open(path / "part.png")) > 0
        shadow = np.load(path / "shadow.npy")
        y, x = np.nonzero(mask | (shadow > 0.08))
        margin = max(12, int(0.21 * max(np.ptp(x), np.ptp(y))))
        crop = (
            max(0, x.min() - margin),
            max(0, y.min() - margin),
            min(mask.shape[1], x.max() + margin + 1),
            min(mask.shape[0], y.max() + margin + 1),
        )
        left = 12 + 584 * (i % 2)
        top = 74 + 271 * (i // 2)
        label(draw, (left + 4, top), f"{key}  {title}", 20)
        for j, name in enumerate(["before", "after"]):
            rect = (left + j * 284, top + 37, left + j * 284 + 280, top + 224)
            with Image.open(path / (name + ".png")) as img:
                cropped = img.crop(crop)
                panel(canvas, cropped, rect)
            label(
                draw,
                (left + 8 + j * 284, top + 224),
                ("原始曝光" if j == 0 else "迁入同帧") + f"  {frame}",
                16,
                "#51675e",
            )
    canvas.save(out, quality=93)


def joint(root):
    base = root / "joint"
    dest = root / "joint-front-rear.mp4"
    tmp = dest.with_suffix(".tmp.mp4")
    with imageio.get_writer(
        str(tmp),
        fps=6,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        output_params=["-threads", "3", "-movflags", "+faststart"],
    ) as writer:
        for pair in ["H234-B232", "H235-B233"]:
            names = sorted(p.name for p in (base / "original/front").glob("*.png"))
            if len(names) != 48:
                raise ValueError("Expected all 48 recorded rig poses")
            layout = Image.new("RGB", (1200, 614), PAPER)
            draw = ImageDraw.Draw(layout)
            label(draw, (16, 4), "KITTI-360 0000   /   " + pair, 27)
            label(draw, (16, 48), "原场景", 22)
            label(draw, (612, 48), "同机位 · 同时刻 · 迁入两件", 22)
            label(draw, (16, 77), "前视相机", 16)
            label(draw, (16, 257), "虚拟后视相机", 16)
            for group, name in enumerate(names):
                frame = layout.copy()
                for j, host in enumerate(["original", pair]):
                    left = 8 + j * 600
                    for camera, top, height in [("front", 102, 150), ("rear", 283, 322)]:
                        with Image.open(base / host / camera / name) as img:
                            panel(frame, img, (left, top, left + 580, top + height))
                label(ImageDraw.Draw(frame), (1005, 13), f"轨迹帧 {group + 1:02}/48", 21)
                writer.append_data(np.asarray(frame))
    tmp.replace(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/roadside-props")
    parser.add_argument("--joint-only", action="store_true", help="Compose reloaded checkpoints without physical-camera plates")
    args = parser.parse_args()
    if not args.joint_only:
        physical(args.views, args.views / "physical.jpg")
    joint(args.views)
