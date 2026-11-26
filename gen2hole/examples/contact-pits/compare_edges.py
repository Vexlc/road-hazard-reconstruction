"""Compare a feathered render against a hard-edge render at identical camera exposures."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def image(folder, name):
    return np.asarray(Image.open(folder / name))


def compare(baseline, feathered, output):
    rows = {}
    keys = [
        s["id"] for s in json.loads(Path(__file__).with_name("survey.json").read_text())["states"]
    ]
    for side in ("back", "front"):
        for key in keys:
            name = side + "/" + key
            a, b = baseline / name, feathered / name
            frames = [json.loads((p / "placement.json").read_text())["best_frame"] for p in (a, b)]
            labels = image(a, "part.png")
            before = image(b, "before.png")
            if (
                frames[0] != frames[1]
                or not np.array_equal(labels, image(b, "part.png"))
                or not np.array_equal(before, image(a, "before.png"))
            ):
                raise ValueError("Different camera frame, source image or support mask for " + name)
            mask = labels > 0
            if mask.ndim == 3:
                mask = mask.any(axis=-1)
            dist = cv2.distanceTransform(mask.astype("u1"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            edge, interior = mask & (dist <= 2), dist >= 6
            pair = [
                np.abs(image(p, "after.png").astype(float) - before).mean(axis=-1) for p in (a, b)
            ]
            rows[name] = dict(
                best_frame=frames[1],
                edge_px=int(edge.sum()),
                interior_px=int(interior.sum()),
                edge_mean_abs_rgb=[float(t[edge].mean()) for t in pair],
                interior_mean_abs_rgb=(
                    [float(t[interior].mean()) for t in pair] if interior.any() else None
                ),
            )
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "feathered", "output"):
        parser.add_argument(name, type=Path)
    compare(**vars(parser.parse_args()))
