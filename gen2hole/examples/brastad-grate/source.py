"""Regenerate both lossless working images from W.carter's checksummed original JPEG."""

import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from casting import S


def extract(original, out):
    if hashlib.sha256(original.read_bytes()).hexdigest() != S["original_sha256"]:
        raise ValueError("Original photograph differs from survey")
    im = cv2.imread(str(original))
    x, y, w, h = S["crop_xywh"]
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "casting.png"), im[y : y + h, x : x + w])
    w, h = S["rust_rectified_size_px"]
    pose = cv2.getPerspectiveTransform(
        np.float32(S["rust_patch_original_quad_px"]),
        np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
    )
    tile = cv2.warpPerspective(im, pose, (w, h))
    x, y, w, h = S["rust_rectified_crop_xywh"]
    cv2.imwrite(str(out / "oxidation.png"), tile[y : y + h, x : x + w])
    for name, key in [("casting", "crop_sha256"), ("oxidation", "rust_patch_sha256")]:
        if hashlib.sha256((out / (name + ".png")).read_bytes()).hexdigest() != S[key]:
            raise ValueError("Working image differs: " + name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "out"):
        p.add_argument(name, type=Path)
    extract(**vars(p.parse_args()))
