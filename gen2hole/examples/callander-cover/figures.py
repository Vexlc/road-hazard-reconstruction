"""Create source-aligned comparisons and vehicle-camera panels from the rebuilt model."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from camera import HERE, load


def tile(canvas, rgb, xy, size, title):
    im = rgb.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    x, y = xy
    canvas.paste(im, (x + (size[0] - im.width) // 2, y + 24))
    ImageDraw.Draw(canvas).text((x + 5, y + 5), title, fill="white")


def make(output, dest):
    dest.mkdir(parents=True, exist_ok=True)
    s, _ = load()
    photo = Image.open(HERE / "references/1673294.jpg").convert("RGB")
    render = Image.open(output / "views/source-render.png").convert("RGB")
    board = Image.new("RGB", (1000, 578), (31, 34, 36))
    box = (171, 118, 472, 295)
    for i, (im, title) in enumerate(
        ((photo, "Field photograph | Richard Webb"), (render, "Mesh | original camera"))
    ):
        tile(
            board,
            im.crop(box).resize((490, 287), Image.Resampling.LANCZOS),
            (500 * i, 0),
            (490, 287),
            title,
        )
    with np.load(output / "views/source-rays.npz") as rays:
        parts = rays["first_part"]
    pal = np.array(
        [
            [33, 35, 36],
            [105, 108, 108],
            [211, 147, 70],
            [0, 0, 0],
            [0, 0, 0],
            [79, 178, 166],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [107, 151, 217],
        ],
        "u1",
    )
    tile(
        board,
        Image.fromarray(pal[parts]).crop(box).resize((410, 241), Image.Resampling.NEAREST),
        (10, 312),
        (475, 240),
        "First return | iron face, frame, broken asphalt",
    )
    detail = photo.crop((335, 220, 397, 254)).resize((412, 226), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(detail)
    for pair in s["heldout_front_edge_top_and_toe_px"]:
        for j, (u, v) in enumerate(pair):
            x = (u - 335 + 0.5) * 412 / 62
            y = (v - 220 + 0.5) * 226 / 34
            draw.line((x - 7, y, x + 7, y), fill=(85, 238, 220) if j else (255, 206, 72), width=2)
    tile(
        board,
        detail,
        (501, 312),
        (490, 240),
        "Held-out edge | upper lip (yellow), lower trace (cyan)",
    )
    board.save(dest / "field.jpg", quality=94)
    board = Image.new("RGB", (1000, 565), (31, 34, 36))
    for j, side in enumerate(("front", "back")):
        path = output / "drive" / side / "A580"
        im = Image.open(path / "after.png").convert("RGB")
        raw = np.array(Image.open(path / "part.png"))
        m = raw > 0
        if m.ndim == 3:
            m = m.any(axis=-1)
        y, x = np.where(m)
        width = max(178, int(max(np.ptp(x) + 1, 1.75 * (np.ptp(y) + 1)) * 1.8))
        height = int(width / 1.75)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        x0 = int(np.clip(cx - width / 2, 0, im.width - width))
        y0 = int(np.clip(cy - height / 2, 0, im.height - height))
        detail = im.crop((x0, y0, x0 + width, y0 + height)).resize(
            (round(244 * width / height), 244), Image.Resampling.LANCZOS
        )
        crop = im.copy()
        ImageDraw.Draw(crop).rectangle(
            (x0, y0, x0 + width, y0 + height), outline=(233, 177, 79), width=3
        )
        tile(
            board,
            crop,
            (0, 278 * j),
            (497, 249),
            side + " camera | recorded road, installed cover",
        )
        tile(board, detail, (500, 278 * j), (490, 249), "Enlarged installation")
    board.save(dest / "road.jpg", quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("dest", type=Path)
    make(**vars(p.parse_args()))
