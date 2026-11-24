"""Restore registered scan channels and trace the visible drain slots from pinned ambientCG packages."""

import argparse
import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    "005": "3cf173c8d898bb050f875cc609b10d30a38231a686dbda57a4130cea7d8a8e82",
    "008": "72bc358ed55496023f90fc1d8a70bc5d5eb891d1f616882cc17b6f7ca986e803",
}


def slot_mask(color, relief):
    y, x = np.indices(color.shape)
    radius = np.hypot(x - 507, y - 511)
    # The central logo and the two rectangular lifting sockets have dark recesses too.
    annuli = ((radius > 80) & (radius < 140)) | ((radius > 279) & (radius < 353))
    candidate = np.uint8(annuli & (relief < 76) & (color < 58))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slots = [c for c in contours if cv2.contourArea(c) > 410]
    if len(slots) != 24:
        raise ValueError(f"Expected two rings with 24 visible slot mouths, found {len(slots)}")
    mask = np.zeros_like(color)
    cv2.drawContours(mask, slots, -1, 255, cv2.FILLED)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, help="directory holding the original ManholeCover005/008 ZIPs")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    for key, digest in ARCHIVES.items():
        prefix = "ManholeCover" + key + "_1K-PNG"
        path = args.archives / (prefix + ".zip")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wrong original archive: {path}")
        dest = args.out / key
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for channel in ("Color", "Displacement", "Opacity", "Metalness", "Roughness", "NormalGL"):
                (dest / (channel + ".png")).write_bytes(archive.read(f"{prefix}_{channel}.png"))
        color = cv2.imread(str(dest / "Color.png"), cv2.IMREAD_GRAYSCALE)
        relief = cv2.imread(str(dest / "Displacement.png"), cv2.IMREAD_GRAYSCALE)
        mask = slot_mask(color, relief) if key == "005" else np.zeros_like(color)
        if not cv2.imwrite(str(dest / "openings.png"), mask):
            raise OSError("Failed to save registered slot mask")


if __name__ == "__main__":
    main()
