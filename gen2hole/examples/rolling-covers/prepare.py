"""Recover the original scan channels and isolate the ten dark vent mouths on cover 006."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


ARCHIVES = {
    "003": "65862f21a505337daf9e77edd86afc3e9066d0118e56a5acecdc9419054d1e18",
    "006": "dce260d2c2c778a763069dd8d9aac72796f89fb95cb2da7bde4d6d9768ef5a55",
}


def prepare(archives, out):
    for key, digest in ARCHIVES.items():
        stem = f"ManholeCover{key}_1K-PNG"
        path = archives / (stem + ".zip")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wrong source scan: {path}")
        dest = out / key
        dest.mkdir(parents=True, exist_ok=True)
        with ZipFile(path) as archive:
            for name in ("Color", "Displacement", "Opacity", "NormalGL", "Roughness", "Metalness"):
                (dest / (name + ".png")).write_bytes(archive.read(f"{stem}_{name}.png"))
        grey = cv2.imread(str(dest / "Color.png"), cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(grey)
        if key == "006":
            depth = cv2.imread(str(dest / "Displacement.png"), cv2.IMREAD_GRAYSCALE)
            y, x = np.indices(grey.shape)
            dark = (grey < 30) & (depth < 115) & (np.hypot(x - 508, y - 510) < 395)
            contours, _ = cv2.findContours(dark.astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vents = [c for c in contours if 1200 < cv2.contourArea(c) < 3500]
            if len(vents) != 10:
                raise ValueError("Expected eight outer vents and two central openings")
            # Each individual mouth is convex. Remove isolated scan noise at its lip without
            # filling the iron bridge between the two central openings.
            for vent in vents:
                cv2.fillConvexPoly(mask, cv2.convexHull(vent), 255)
        if not cv2.imwrite(str(dest / "openings.png"), mask):
            raise OSError("Failed to write scan mask")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prepare(args.archives, args.out)
