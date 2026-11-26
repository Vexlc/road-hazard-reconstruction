"""Compose field, stereo evidence and driving plates from the original photographs and output."""

import argparse
from pathlib import Path
from subprocess import check_output
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from camera import HERE


def figures(views, drive, out):
    out.mkdir(parents=True, exist_ok=True)
    face = check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True)
    font = ImageFont.truetype(face, 22)

    def paste(board, im, box, title):
        x, y, w, h = box
        im = ImageOps.contain(im.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
        board.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))
        ImageDraw.Draw(board).text((x + 2, y + h + 5), title, font=font, fill="#252d2e")

    board = Image.new("RGB", (1180, 873), "#f4f4f0")
    for i, (file, title) in enumerate(
        [
            (HERE / "references/site.jpg", "纽约第二大道现场 · David Shankbone"),
            (views / "field-render.jpg", "按现场机位渲染 · 灰色为裁片外"),
            (views / "P584-near.jpg", "从侧前方看坑沿遮挡与倒锥"),
            (views / "P584-opposite.jpg", "反方向 · 黑袋与折叠纸板"),
        ]
    ):
        x = 12 + (i % 2) * 585
        y = 8 + (i // 2) * 435
        paste(board, Image.open(file), (x, y, 565, 388), title)
    board.save(out / "field.jpg", quality=94)
    board = Image.new("RGB", (1180, 594), "#f4f4f0")
    for i, (side, name) in enumerate([("front", "前摄"), ("back", "后摄")]):
        folder = drive / side / "P584"
        im = Image.open(folder / "after.png")
        part = np.asarray(Image.open(folder / "part.png"))
        y, x = np.where(part > 1)
        cx = (x.min() + x.max()) / 2
        cy = (y.min() + y.max()) / 2
        w = min(im.width, round((x.max() - x.min() + 1) * 1.22))
        h = min(im.height, max(round(w * 0.32), int(np.ptp(y) * 1.4)))
        x0 = int(np.clip(cx - w / 2, 0, im.width - w))
        y0 = int(np.clip(cy - h / 2, 0, im.height - h))
        paste(board, im, (12 + i * 585, 6, 565, 315), name + "合成帧")
        paste(
            board,
            im.crop((x0, y0, x0 + w, y0 + h)),
            (12 + i * 585, 366, 565, 177),
            name + "坑口局部",
        )
    board.save(out / "road.jpg", quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "drive", "out"):
        p.add_argument(name, type=Path)
    figures(**vars(p.parse_args()))
