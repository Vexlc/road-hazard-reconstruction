"""Show source photographs, original host exposures and the paired defects at matching camera coordinates."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

helpers = import_module("review-traffic-cones")
ROOT = helpers.ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/measured-pavement/physical.jpg")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/measured-pavement")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    a = p.parse_args()
    page = Image.new("RGB", (1200, 884), helpers.COLORS["paper"])
    helpers.text(page, (18, 6), "两处实测破损在道路中的位置和外观", 30)
    labels = ["P238 · 露出龟裂底层的片状剥落", "P239 · 露出粗集料的宽碗坑"]
    for j, k in enumerate(["P238", "P239"]):
        im = Image.open(a.assets / k / "series/references" / f"{k}.png")
        helpers.inside(page, im, (j * 600 + 4, 50), (355, 155))
        helpers.text(page, (j * 600 + 330, 100), labels[j].replace(" · ", "\n"), 20)
    for j, label in enumerate(["迁入后的实车全幅", "同一曝光的原路面", "双坑迁入后"]):
        helpers.text(page, (12 + j * 400, 211), label, 20)
    for i, (site, cam, label) in enumerate(
        [
            ("kitti", "image_03", "KITTI 城区右侧鱼眼"),
            ("suburb", "image_00", "KITTI 郊区前视"),
            ("audi", "rear_center", "Audi 驶过后，实车尾部鱼眼"),
        ]
    ):
        root = a.views / site / "P238-P239" / cam
        index = json.loads((a.views / site / "index.json").read_text())["P238-P239"]["views"][cam]
        before = Image.open(root / "before.png")
        after = Image.open(root / "after.png")
        box = helpers.rect(np.asarray(Image.open(root / "part.png")) > 0, *before.size, 20)
        full = Image.open(root / "full.jpg")
        ImageDraw.Draw(full).rectangle(
            tuple(np.array(box) + np.tile(index["crop_xy"], 2)), outline="#f6db6d", width=3
        )
        top = 244 + i * 210
        for j, im in enumerate([full, before.crop(box), after.crop(box)]):
            helpers.inside(page, im, (j * 400 + 6, top), (388, 164))
        helpers.text(page, (16, top + 168), label + "  ·  曝光 " + str(index["frame"]), 20)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    page.save(a.out, quality=94)


if __name__ == "__main__":
    main()
