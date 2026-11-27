from pathlib import Path
import cv2
import numpy as np
import torch
import argparse
from cameras import HERE

parser = argparse.ArgumentParser(description="Extract the opposing USGS view correspondences")
parser.add_argument("output", type=Path)
output = parser.parse_args().output
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

torch.set_grad_enabled(False)
cv2.setNumThreads(2)
scale = np.float32([2256 / 902, 1496 / 598])
corners = [
    np.float32([[255, 257], [658, 262], [730, 389], [101, 395]]),
    np.float32([[688, 390], [173, 499], [94, 378], [484, 338]]),
]
chart = np.float32([[250, 185], [1350, 185], [1350, 965], [250, 965]])
H = []
imgs = []
for j in range(2):
    im = cv2.imread(str(HERE / "references" / ["carriageway.jpg", "reverse.jpg"][j]), 0)
    h = cv2.getPerspectiveTransform(corners[j] * scale, chart)
    H.append(h)
    a = cv2.warpPerspective(im, h, (1630, 1230))
    imgs.append(cv2.createCLAHE(clipLimit=2.25, tileGridSize=(20, 16)).apply(a))
ext = SuperPoint(max_num_keypoints=2700).eval().cuda()
net = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()
rows = []
for x, y, r, b in [
    [10, 10, 1620, 1220],
    [90, 220, 1070, 1170],
    [600, 100, 1610, 1190],
    [90, 90, 1590, 740],
    [250, 540, 1440, 1190],
]:
    feat = [
        ext.extract(
            torch.from_numpy(im[y:b, x:r].copy()[None, None]).cuda().float() / 255, resize=1200
        )
        for im in imgs
    ]
    m = rbd(net({"image0": feat[0], "image1": feat[1]}))
    ids = m["matches"].cpu().numpy()
    pts = [rbd(f)["keypoints"].cpu().numpy()[ids[:, j]] + [x, y] for j, f in enumerate(feat)]
    rows.extend(np.column_stack([*pts, m["scores"].cpu().numpy()]))
data = []
for row in sorted(rows, key=lambda r: -r[-1]):
    if not any(max(abs(row[:2] - q[:2])) < 4 or max(abs(row[2:4] - q[2:4])) < 4 for q in data):
        data.append(row)
a = np.asarray(data)
px = [
    cv2.perspectiveTransform(a[:, i * 2 : i * 2 + 2].copy()[:, None, :], np.linalg.inv(h))[:, 0]
    for i, h in enumerate(H)
]
p, q = px
hold = np.arange(len(a)) % 5 == 3
F, _ = cv2.findFundamentalMat(p[~hold], q[~hold], cv2.USAC_MAGSAC, 2.0, 0.99999, 60000)


def err(F, p, q):
    a = np.c_[p, np.ones(len(p))]
    b = np.c_[q, np.ones(len(q))]
    l = a @ F.T
    r = b @ F
    return abs((b * l).sum(1)) / np.sqrt((l[:, :2] ** 2 + r[:, :2] ** 2).sum(1))


err = err(F, p, q)
valid = err < 1.5
np.savez_compressed(output, native0=p, native1=q, hold=hold, error=err, score=a[:, -1], F=F)
