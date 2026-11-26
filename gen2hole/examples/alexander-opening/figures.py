"""Compose field, stereo evidence and driving plates from the original photographs and output."""

import argparse
from pathlib import Path
from subprocess import check_output
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from geometry import HERE, SIZE


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
            (HERE / "references/detail.jpg", "近景原片 · Bart Everson"),
            (HERE / "references/street.jpg", "8 秒后的街道原片 · Bart Everson"),
            (views / "detail-render.jpg", "近景相机回投 · 盖板与敞井"),
            (views / "street-render.jpg", "街道相机回投 · 灰色为裁片外"),
        ]
    ):
        x = 12 + (i % 2) * 585
        y = 8 + (i // 2) * 435
        paste(board, Image.open(file), (x, y, 565, 388), title)
    board.save(out / "field.jpg", quality=94)
    evidence = Image.new("RGB", (1180, 450), "#f4f4f0")
    with np.load(views.parent / "stereo.npz") as estimates:
        for i, (name, box) in enumerate(
            [("detail", [338, 97, 560, 320]), ("street", [235, 244, 380, 375])]
        ):
            source = Image.open(HERE / "references" / (name + ".jpg")).resize(
                tuple(SIZE), Image.Resampling.LANCZOS
            )
            pixels = np.array(source)
            valid = estimates["valid" + str(i)]
            y, x = np.where(valid)
            x0, y0 = estimates["offset" + str(i)]
            pixels[y + y0, x + x0] = [68, 238, 216]
            px = np.rint(np.array(box) * np.tile(SIZE / [613, 460], 2)).astype(int)
            paste(
                evidence,
                Image.fromarray(pixels).crop(tuple(px)),
                (12 + i * 585, 10, 565, 390),
                ("近照" if i == 0 else "远照") + " · 绿色为通过双向深度检查的像素",
            )
    evidence.save(out / "stereo.jpg", quality=94)
    board = Image.new("RGB", (1180, 594), "#f4f4f0")
    for i, (side, name) in enumerate([("front", "前摄"), ("back", "后摄")]):
        folder = drive / side / "M585"
        im = Image.open(folder / "after.png")
        part = np.asarray(Image.open(folder / "part.png"))
        y, x = np.where(np.isin(part, [5, 185, 186, 187, 189, 190]))
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
            name + "井盖与井口",
        )
    board.save(out / "road.jpg", quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "drive", "out"):
        p.add_argument(name, type=Path)
    figures(**vars(p.parse_args()))
