"""Optionally regenerate the image correspondences with SuperPoint and LightGlue."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

HERE = Path(__file__).resolve().parent
CROPS = [
    [(0, 0, 1024, 683), (0, 0, 1024, 683)],
    [(312, 146, 805, 415), (0, 145, 925, 590)],
    [(350, 150, 645, 285), (0, 140, 550, 365)],
    [(460, 160, 810, 350), (50, 180, 905, 455)],
]
STATIC = [
    [
        [416, 0],
        [690, 0],
        [840, 166],
        [800, 240],
        [685, 269],
        [626, 276],
        [557, 269],
        [528, 264],
        [509, 248],
        [470, 236],
        [424, 223],
        [414, 179],
    ],
    [
        [0, 0],
        [1023, 0],
        [1023, 282],
        [888, 308],
        [818, 351],
        [749, 351],
        [656, 410],
        [580, 400],
        [533, 370],
        [433, 348],
        [310, 335],
        [227, 327],
        [133, 308],
        [69, 311],
        [0, 310],
    ],
]


@torch.inference_mode()
def extract(output):
    detector = SuperPoint(max_num_keypoints=4096).eval().cuda()
    matcher = (
        LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()
    )
    images = [
        cv2.imread(str(HERE / "references" / (name + ".jpg")), 0) for name in ("west", "northwest")
    ]
    candidates = []
    for boxes in CROPS:
        features = []
        for im, (x, y, u, v) in zip(images, boxes):
            crop = torch.from_numpy(im[None, None, y:v, x:u].copy()).float().cuda() / 255
            features.append(detector.extract(crop, resize=1000))
        matched = rbd(matcher({"image0": features[0], "image1": features[1]}))
        points = [
            rbd(feat)["keypoints"].cpu().numpy() + box[:2] for feat, box in zip(features, boxes)
        ]
        for (i, j), score in zip(matched["matches"].cpu().numpy(), matched["scores"].cpu().numpy()):
            candidates.append(np.r_[points[0][i], points[1][j], score])
    unique = []
    for row in sorted(candidates, key=lambda x: -x[4]):
        if not any(
            np.linalg.norm(row[:2] - old[:2]) < 3 or np.linalg.norm(row[2:4] - old[2:4]) < 3
            for old in unique
        ):
            unique.append(row)
    pts = np.asarray(unique)
    _, inliers = cv2.findFundamentalMat(pts[:, :2], pts[:, 2:4], cv2.USAC_MAGSAC, 1.7, 0.999, 30000)
    rows = []
    for p in pts[inliers.ravel() > 0]:
        if all(
            cv2.pointPolygonTest(np.int32(region), tuple(xy), False) >= 0
            for region, xy in zip(STATIC, [p[:2], p[2:4]])
        ):
            rows.append(
                dict(
                    west=p[:2].round(2).tolist(),
                    northwest=p[2:4].round(2).tolist(),
                    confidence=round(float(p[4]), 3),
                )
            )
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    extract(parser.parse_args().output)
