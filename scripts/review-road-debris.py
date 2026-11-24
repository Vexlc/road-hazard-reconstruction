"""Join both debris placements and export a still comparing their registered driving cameras."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

video = import_module("review-round-covers").video
ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"R242-R243": "旧轮胎 / 钢轮圈", "R244-R245": "侧倒的周转筐 / 水泥袋"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/road-debris")
    parser.add_argument("--plate", type=Path)
    args = parser.parse_args()
    for site, name in [
        ("sun", "kitti-sun-front-side"),
        ("urban", "kitti-urban-front-side"),
        ("audi", "audi-front-rear"),
    ]:
        video(args.views, site, args.views / (name + ".mp4"), PAIRS)
    if not args.plate:
        return
    page = Image.new("RGB", (1380, 586), "#24282b")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22, index=2)
    for j, title in enumerate(["KITTI 日照路段 · 前视", "Audi 左前鱼眼", "Audi 驶过后的车尾鱼眼"]):
        draw.text((j * 460 + 13, 8), title, font=font, fill="white")
    for i, (pair, title) in enumerate(PAIRS.items()):
        top = 45 + i * 270
        draw.text((13, top), pair + "   " + title, font=font, fill="#dad8d0")
        for j, (site, cam) in enumerate(
            [("sun", "image_00"), ("audi", "front_left"), ("audi", "rear_center")]
        ):
            path = args.views / pair / site / pair / cam
            ys, xs = np.nonzero(np.array(Image.open(path / "part.png")))
            with Image.open(path / "after.png") as frame:
                box = (
                    max(0, xs.min() - 22),
                    max(0, ys.min() - 20),
                    min(frame.width, xs.max() + 23),
                    min(frame.height, ys.max() + 21),
                )
                detail = ImageOps.contain(
                    frame.convert("RGB").crop(box), (432, 219), Image.Resampling.LANCZOS
                )
                page.paste(
                    detail, (j * 460 + (460 - detail.width) // 2, top + 36 + (219 - detail.height) // 2)
                )
    args.plate.parent.mkdir(parents=True, exist_ok=True)
    page.save(args.plate, quality=92)


if __name__ == "__main__":
    main()
