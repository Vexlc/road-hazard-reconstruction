"""Write field, driving and footprint comparison figures from the released artifacts."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from camera import HERE, load


def panel(canvas, image, pos, size, title):
    x, y = pos
    w, h = size
    ratio = min(w / image.width, h / image.height)
    image = image.resize(
        (round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS
    )
    canvas.paste(image, (x + (w - image.width) // 2, y + 25))
    ImageDraw.Draw(canvas).text((x + 6, y + 6), title, fill="white")


def figures(views, drive, out):
    out.mkdir(parents=True, exist_ok=True)
    s, _ = load()
    src = Image.open(HERE / "references/2559905.jpg").convert("RGB")
    image = Image.new("RGB", (1070, 590), (32, 36, 38))
    bounds = (119, 247, 503, 418)
    annotated = src.copy()
    draw = ImageDraw.Draw(annotated)
    for row, color in zip(s["sheets"], ["#ffcf67", "#63dccc", "#f2a1c0"]):
        draw.line([tuple(p) for p in row["crest"]], fill=color, width=1)
    panel(image, src, (2, 0), (416, 322), "Gervis Place | Chris Downer, 18 August 2011")
    panel(
        image,
        annotated.crop(bounds),
        (425, 0),
        (636, 322),
        "Three traced crests, separate from the striped insert",
    )
    panel(
        image,
        Image.open(views / "source-view.png").crop(bounds),
        (2, 337),
        (525, 225),
        "Inspection GLB | source camera",
    )
    panel(
        image,
        Image.open(views / "transfer-view.png").crop(bounds),
        (536, 337),
        (525, 225),
        "Transfer GLB | same geometry, grain remapped in road coordinates",
    )
    image.save(out / "field.jpg", quality=94)
    image = Image.new("RGB", (1100, 590), (32, 36, 38))
    for j, side in enumerate(("front", "back")):
        folder = drive / side / "P581"
        im = Image.open(folder / "after.png")
        labels = np.asarray(Image.open(folder / "part.png"))
        y, x = np.where(labels > 0)
        w = int(max(x.max() - x.min() + 1, 2.18 * (y.max() - y.min() + 1)) * 1.15)
        w = min(w, im.width)
        h = min(int(w / 2.18), im.height)
        x0 = int(np.clip((x.min() + x.max() - w) / 2, 0, im.width - w))
        y0 = int(np.clip((y.min() + y.max() - h) / 2, 0, im.height - h))
        panel(image, im, (550 * j, 0), (545, 300), side + " | PandaSet 109")
        panel(
            image,
            im.crop((x0, y0, x0 + w, y0 + h)),
            (550 * j, 318),
            (545, 244),
            "Vehicle view | enlarged road detail",
        )
    image.save(out / "road.jpg", quality=94)
    with np.load(views / "sections.npz") as z:
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True)
        for ax, rec in zip(axes, s["sheets"]):
            k = rec["name"]
            t = z[k + "_distance_m"]
            h = z[k + "_z_m"] * 100
            part = z[k + "_part"]
            for val, col in [(1, "#787a78"), (13, "#b4502a"), (175, "#287f85")]:
                ax.plot(t, np.where(part == val, h, np.nan), color=col, lw=2.6)
            ax.axvline(0, color="#494c4d", ls=":", lw=1)
            ax.set_title(k.replace("_", " "), loc="left")
            ax.set_ylabel("Height (cm)")
            ax.set_ylim(-1, 24)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Distance from the crest toward the return to road grade (m)")
        fig.savefig(out / "sections.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("views", "drive", "out"):
        p.add_argument(k, type=Path)
    figures(**vars(p.parse_args()))
