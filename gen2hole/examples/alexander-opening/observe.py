"""Locate casting silhouettes from a traced ellipse and native image gradients."""

import cv2
import numpy as np
from pathlib import Path
import json, hashlib

HERE = Path(__file__).resolve().parent
SIZE = np.array([1600, 1200])
# Seed coordinates on 613 x 460 overview, clockwise from top.
SEEDS = [
    dict(
        rim=[
            [424, 100],
            [486, 110],
            [535, 146],
            [555, 197],
            [554, 245],
            [530, 286],
            [474, 312],
            [427, 309],
            [377, 284],
            [347, 244],
            [337, 202],
            [348, 153],
            [381, 115],
        ],
        bore=[
            [426, 121],
            [483, 129],
            [519, 155],
            [534, 190],
            [537, 225],
            [516, 267],
            [480, 291],
            [441, 297],
            [395, 282],
            [365, 252],
            [352, 210],
            [361, 169],
            [385, 138],
        ],
        lid=[
            [177, 142],
            [218, 146],
            [262, 166],
            [286, 198],
            [298, 238],
            [290, 277],
            [263, 317],
            [222, 344],
            [173, 356],
            [135, 348],
            [100, 322],
            [78, 285],
            [76, 249],
            [89, 208],
            [116, 174],
            [148, 152],
        ],
    ),
    dict(
        rim=[
            [278, 249],
            [309, 251],
            [338, 267],
            [359, 293],
            [370, 326],
            [367, 346],
            [353, 363],
            [328, 370],
            [297, 367],
            [271, 350],
            [251, 326],
            [241, 299],
            [243, 274],
            [256, 256],
        ],
        bore=[
            [280, 261],
            [309, 264],
            [332, 277],
            [351, 299],
            [360, 324],
            [357, 345],
            [342, 357],
            [321, 360],
            [295, 354],
            [275, 342],
            [257, 319],
            [249, 294],
            [251, 275],
            [264, 264],
        ],
        lid=[
            [119, 260],
            [153, 259],
            [184, 269],
            [211, 288],
            [225, 311],
            [230, 340],
            [218, 369],
            [193, 390],
            [164, 402],
            [130, 402],
            [99, 392],
            [78, 374],
            [63, 348],
            [60, 324],
            [70, 295],
            [89, 275],
        ],
    ),
]


def contours(im, seed):
    ellipse = cv2.fitEllipse(np.float32(seed))
    (cx, cy), (w, h), degrees = ellipse
    t = np.arange(120) * 2 * np.pi / 120
    a = np.deg2rad(degrees)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    sec = np.stack((np.cos(t), np.sin(t)), 1)
    start = sec * [w / 2, h / 2] @ R + [cx, cy]
    normal = sec / [w / 2, h / 2] @ R
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    step = np.arange(-4, 4.01, 0.125)
    points = start[:, None, :] + normal[:, None, :] * step[None, :, None]
    px = (points + 0.5) * im.shape[1] / 613 - 0.5
    colors = cv2.remap(im, *px.transpose(2, 0, 1).astype("f4"), cv2.INTER_LINEAR).astype(float)
    delta = np.linalg.norm(np.gradient(colors, axis=1), axis=2)
    # Regularized radial displacement with a continuity penalty across neighbours.
    cost = -delta + 0.19 * abs(step)[None, :] ** 2
    c = cost[0].copy()
    hist = []
    trans = 0.85 * (step[:, None] - step[None, :]) ** 2
    for row in cost[1:]:
        comb = c[:, None] + trans
        hist.append(np.argmin(comb, axis=0))
        c = row + comb.min(0)
    j = int(c.argmin())
    ids = [j]
    for prev in hist[::-1]:
        j = prev[j]
        ids.append(int(j))
    q = points[np.arange(len(t)), ids[::-1]]
    return (q + 0.5) * SIZE / [613, 460] - 0.5


if __name__ == "__main__":
    objs = {}
    for i, name in enumerate(("detail", "street")):
        p = HERE / "references" / (name + ".jpg")
        im = cv2.GaussianBlur(cv2.imread(str(p)), (5, 5), 1.0)
        objs[name] = dict(
            source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            outlines={k: contours(im, seed).round(3).tolist() for k, seed in SEEDS[i].items()},
        )
    objs.update(
        working_image_size=SIZE.tolist(),
        assumed_outer_diameter_m=0.64,
        reference_author="Bart Everson",
        image_license="CC BY 2.0",
        detail_url="https://commons.wikimedia.org/wiki/File:SEWER_-_Open_Manhole_-_New_Orleans_September_2020.jpg",
        street_url="https://commons.wikimedia.org/wiki/File:Open_Manhole_-_New_Orleans_September_2020.jpg",
    )
    (HERE / "survey.json").write_text(json.dumps(objs, indent=2) + "\n")
