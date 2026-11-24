"""Refine a camera-rig hood outline from recurring edges in multiple moving-road exposures."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect(paths, spec, step, below, guard, smooth_arc):
    stack = []
    for filename in paths:
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unreadable exposure: " + str(filename))
        h, w = img.shape
        grid = np.float64(spec["grid"])
        coarse = np.array(spec["edge"]) * [w / grid[0], h / grid[1]]
        x = np.unique(np.r_[np.arange(0, w, step), w - 1])
        pred = np.interp(x, *coarse.T)
        # Compare vertical edge strength at the same physical sensor coordinates across multiple exposures.
        y0 = max(0, int(pred.min()) - 10)
        y1 = min(h, int(pred.max()) + below + 8)
        crop = cv2.GaussianBlur(img[y0:y1], (0, 0), 1.0)
        gradient = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3))
        gradient = cv2.blur(gradient, (step, 1))
        stack.append(gradient[:, x])
    score = np.median(stack, axis=0)
    yy = np.arange(y0, y1)
    allowed = (yy[:, None] >= pred - 5) & (yy[:, None] <= pred + below)
    unit = np.percentile(score[allowed], 90)
    if unit < 5:
        raise ValueError("Too little repeated image-edge evidence to fit this rig")
    layer = np.where(allowed, np.minimum(score / unit, 4), -1e6)
    # Dynamic programming on contiguous sensor columns. Penalize large jumps away from the coarse local slope.
    best = layer[:, 0].copy()
    back = np.empty_like(layer, dtype=np.int32)
    for j in range(1, len(x)):
        delta = yy[:, None] - yy[None, :]
        trend = pred[j] - pred[j - 1]
        penalty = np.where(abs(delta) <= max(2, 0.45 * (x[j] - x[j - 1])), 0.12 * abs(delta - trend), 1e6)
        candidates = best[None, :] - penalty
        back[:, j] = np.argmax(candidates, axis=1)
        best = candidates.max(axis=1) + layer[:, j]
    row = np.empty(len(x), int)
    row[-1] = np.argmax(best)
    for j in range(len(x) - 1, 0, -1):
        row[j - 1] = back[row[j], j]
    edge = yy[row]
    strength = score[row, np.arange(len(x))]
    consensus = None
    if smooth_arc:
        # The top of this hood is one smooth arc. Reflections can create a second persistent edge.
        coord = x / w
        rng = np.random.default_rng(236)
        inliers = np.zeros(len(x), bool)
        for _ in range(300):
            selected = rng.choice(len(x), 3, replace=False)
            trial = np.polyval(np.polyfit(coord[selected], edge[selected], 2), coord)
            keep = abs(edge - trial) <= 2
            if keep.sum() > inliers.sum():
                inliers = keep
        if inliers.mean() < 0.7 or np.ptp(coord[inliers]) < 0.65:
            raise ValueError("A single smooth hood arc is not supported across this image")
        curve = np.polyval(np.polyfit(coord[inliers], edge[inliers], 2), coord)
        consensus = float(inliers.mean())
        edge = np.floor(np.minimum(edge, curve)).astype(int)
    if np.median(strength) < unit:
        raise ValueError("Ambiguous recurring rig boundary")
    return dict(grid=[w, h], edge=np.stack([x, edge - guard], 1).tolist()), dict(
        training_exposures=len(paths),
        guard_px=guard,
        smooth_arc_inlier_fraction=consensus,
        median_gradient=float(np.median(strength)),
        median_refinement_px=float(np.median(edge - pred)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--sensor", required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--search-below", type=int, default=72)
    p.add_argument(
        "--smooth-arc", action="store_true", help="Fit one smooth hood arc when supported by recurring edges"
    )
    p.add_argument("--guard", type=int, default=4)
    args = p.parse_args()
    if min(args.every, args.search_below) < 1 or args.guard < 0:
        p.error("Expected positive intervals and a nonnegative guard")
    all_frames = sorted(args.images.glob("*.png"))
    training = all_frames[:: args.every]
    if len(training) < 5:
        p.error("At least five different moving-road exposures are needed")
    data = json.loads(args.seed.read_text())
    edge, score = detect(
        training, data[args.scene][args.sensor], 8, args.search_below, args.guard, args.smooth_arc
    )
    result = {args.scene: {args.sensor: edge}, "fit": dict(**score, frames=[p.name for p in training])}
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
