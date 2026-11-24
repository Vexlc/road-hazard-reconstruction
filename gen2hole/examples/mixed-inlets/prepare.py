"""Restore the registered 1K scan and trace the visibly clear parts of the sixteen vent mouths."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


def prepare(archive, out):
    if (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        != "1433692acff927eb227c7200139cf49caca5a11cfe4808614b70ecfd696400df"
    ):
        raise ValueError("Expected the original 1K ManholeCover004 PNG archive")
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as files:
        for role in ("Color", "Displacement", "Metalness", "NormalGL", "Opacity", "Roughness"):
            (out / (role + ".png")).write_bytes(files.read(f"ManholeCover004_1K-PNG_{role}.png"))
    grey = cv2.imread(str(out / "Color.png"), cv2.IMREAD_GRAYSCALE)
    y, x = np.indices(grey.shape)
    r = np.hypot(x - 527, y - 495)
    dark = ((r > 172) & (r < 246) & (grey < 52)).astype("u1")
    loops, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mouths = [c for c in loops if 200 < cv2.contourArea(c) < 900]
    if len(mouths) != 17:
        raise ValueError("Expected fifteen clear mouths and two visible gaps across the sixteenth")
    mask = np.zeros_like(grey)
    cv2.drawContours(mask, mouths, -1, 255, cv2.FILLED)
    if not cv2.imwrite(str(out / "openings.png"), mask):
        raise OSError("Unable to write vent mask")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "004")
    args = parser.parse_args()
    prepare(args.archive, args.out)
