"""Lay out field comparisons and host views directly from the verified renders."""

import argparse
from pathlib import Path
from subprocess import check_output
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def figures(views, drive, out):
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(
        check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True), 22
    )

    def panel(canvas, img, box, title):
        x, y, w, h = box
        canvas.paste(ImageOps.contain(img, (w, h), Image.Resampling.LANCZOS), (x, y))
        ImageDraw.Draw(canvas).text((x, y + h + 5), title, font=font, fill="#22292b")

    source = Image.open(views / "field.jpg")
    extracted = Image.open(views / "same-pose.jpg")
    board = Image.new("RGB", (1180, 810), "#f5f5f2")
    panel(board, source, (18, 16, 353, 510), "原照局部")
    panel(board, extracted, (390, 16, 353, 510), "同一机位的网格")
    panel(
        board,
        Image.open(views / "approach.jpg").crop((184, 66, 601, 565)),
        (762, 16, 398, 510),
        "斜前方合成视角",
    )
    for x, img in [(18, source), (606, extracted)]:
        crop = img.crop((62, 730, 594, 941))
        panel(
            board,
            crop,
            (x, 574, 554, 191),
            "现场井沿、底座和入井绳索" if x == 18 else "同一局部的重建",
        )
    board.save(out / "field.jpg", quality=94)
    road = Image.new("RGB", (1180, 786), "#f5f5f2")
    for x, side, label in [(18, "front", "前摄"), (607, "back", "后摄")]:
        loc = drive / side / "M582"
        im = Image.open(loc / "after.png")
        part = np.array(Image.open(loc / "part.png"))
        y0, x0 = np.nonzero(part == 176)
        ymin, ymax = int(y0.min()), int(y0.max())
        xmin, xmax = int(x0.min()), int(x0.max())
        crop = im.crop(
            (
                max(0, xmin - 100),
                max(0, ymin - 20),
                min(im.width, xmax + 110),
                min(im.height, ymax + 24),
            )
        )
        panel(road, im, (x, 15, 550, 380), label + "合成帧")
        panel(road, crop, (x, 450, 550, 287), label + "井口与警示柱")
    road.save(out / "road.jpg", quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "drive", "out"):
        p.add_argument(name, type=Path)
    figures(**vars(p.parse_args()))
