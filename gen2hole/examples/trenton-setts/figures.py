"""Compose field, stereo evidence and driving plates from the original photographs and output."""

import argparse
from pathlib import Path
from subprocess import check_output
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from cameras import HERE


def figures(models, views, drive, out):
    out.mkdir(parents=True, exist_ok=True)
    face = check_output(["fc-match", "-f", "%{file}", "sans:lang=zh"], text=True)
    font = ImageFont.truetype(face, 22)

    def paste(board, im, box, title):
        x, y, w, h = box
        im = ImageOps.contain(im.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
        board.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))
        ImageDraw.Draw(board).text((x + 2, y + h + 5), title, font=font, fill="#252d2e")

    board = Image.new("RGB", (1180, 745), "#f4f4f0")
    for i, n in enumerate(("detail", "street")):
        paste(
            board,
            Image.open(HERE / "references" / (n + ".jpg")),
            (12 + i * 585, 8, 565, 300),
            ("近处机位 · 10:45:00", "远处机位 · 10:44:49")[i],
        )
        paste(
            board,
            Image.open(views / (n + "-outline.jpg")),
            (12 + i * 585, 349, 565, 152),
            ("近处原照与几何轮廓", "几何轮廓投影到远处原照")[i],
        )
    field = models / "model/series/assets/pothole/P583"
    with np.load(field / "stereo.npz") as f:
        lab = f["labels"]
        v = f["accepted"]
        image = np.zeros((*lab.shape, 3), "u1") + [236, 237, 231]
        image[lab == 1] = [155, 159, 155]
        image[lab > 1] = [225, 175, 98]
        image[(lab > 1) & v] = [26, 120, 119]
    paste(
        board,
        Image.fromarray(image.astype("u1")),
        (12, 540, 565, 158),
        "绿色为双向匹配支持的露石单元，橙色为补全",
    )
    paste(
        board,
        Image.open(views / "vehicle-height.jpg"),
        (597, 540, 565, 158),
        "从斜前方查看可迁移网格",
    )
    board.save(out / "field.jpg", quality=94)
    board = Image.new("RGB", (1180, 594), "#f4f4f0")
    for i, (side, name) in enumerate([("front", "前摄"), ("back", "后摄")]):
        folder = drive / side / "P583"
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
            name + "露石路面局部",
        )
    board.save(out / "road.jpg", quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "views", "drive", "out"):
        p.add_argument(name, type=Path)
    figures(**vars(p.parse_args()))
