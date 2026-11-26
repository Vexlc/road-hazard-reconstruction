"""Regenerate the mutual RootSIFT correspondences and fixed training split."""

import argparse
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def extract(out):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(585)
    keys, descriptors = [], []
    for name in ("detail", "street"):
        im = cv2.imread(str(HERE / "references" / (name + ".jpg")))
        im = cv2.resize(im, (1600, 1200), interpolation=cv2.INTER_AREA)
        lum = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lum = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(12, 10)).apply(lum)
        points, features = cv2.SIFT_create(
            nfeatures=15000, contrastThreshold=0.013
        ).detectAndCompute(lum, None)
        keys.append(np.array([p.pt for p in points]))
        descriptors.append(np.sqrt(features / (features.sum(1)[:, None] + 0.000001)))
    matcher = cv2.BFMatcher()

    def candidates(a, b):
        return {
            m.queryIdx: m.trainIdx
            for m, n in matcher.knnMatch(a, b, k=2)
            if m.distance < 0.82 * n.distance
        }

    forward, reverse = candidates(*descriptors), candidates(*descriptors[::-1])
    pairs = np.array([(keys[0][i], keys[1][j]) for i, j in forward.items() if reverse.get(j) == i])
    pairs = pairs[np.lexsort((pairs[:, 0, 1], pairs[:, 0, 0]))]
    train = np.arange(len(pairs)) % 5 != 1
    np.savez_compressed(out, pairs=pairs.astype("f4"), train=train)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    extract(p.parse_args().out)
