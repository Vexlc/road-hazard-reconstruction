"""Optional SuperPoint / LightGlue regeneration of both photographs' native feature coordinates."""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from cameras import Camera, photograph

PITCH = 0.0028
WIDTH, HEIGHT = 2160, 1850
WINDOWS = [
    (0, 0, WIDTH, HEIGHT),
    (70, 760, 1500, 1580),
    (350, 270, 2100, 1390),
    (440, 670, 1800, 1480),
    (330, 0, 2160, 1090),
]


def native(c, pixel):
    p = np.asarray(pixel)
    world = np.stack(
        (p[..., 0] * PITCH - 0.85, 3.40 - p[..., 1] * PITCH, np.zeros(p.shape[:-1])), axis=-1
    )
    return c.project(world)


@torch.inference_mode()
def extract(out):
    cams = [Camera(n) for n in ("detail", "street")]
    yy, xx = np.indices((HEIGHT, WIDTH))
    pixels = np.stack((xx, yy), axis=-1)
    images = []
    for name, c in zip(("detail", "street"), cams):
        uv = native(c, pixels)
        g = cv2.cvtColor(photograph(name), cv2.COLOR_BGR2GRAY)
        im = cv2.remap(g, uv[..., 0].astype("f4"), uv[..., 1].astype("f4"), cv2.INTER_LINEAR)
        images.append(cv2.createCLAHE(clipLimit=2.5, tileGridSize=(25, 20)).apply(im))
    detector = SuperPoint(max_num_keypoints=3072).eval().cuda()
    matcher = (
        LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()
    )
    pairs = []
    for x, y, r, b in WINDOWS:
        feats = []
        for im in images:
            patch = torch.from_numpy(im[y:b, x:r].copy()[None, None]).cuda().float() / 255
            feats.append(detector.extract(patch, resize=1400))
        m = rbd(matcher({"image0": feats[0], "image1": feats[1]}))
        ids = m["matches"].cpu().numpy()
        pts = [rbd(f)["keypoints"].cpu().numpy()[ids[:, k]] + [x, y] for k, f in enumerate(feats)]
        pairs.extend(np.column_stack([*pts, m["scores"].cpu().numpy()]))
    keep = []
    for row in sorted(pairs, key=lambda v: -v[-1]):
        if not any(max(abs(row[:2] - r[:2])) < 3 or max(abs(row[2:4] - r[2:4])) < 3 for r in keep):
            keep.append(row)
    data = np.asarray(keep)
    a = native(cams[0], data[:, :2])
    b = native(cams[1], data[:, 2:4])
    np.savez_compressed(out, pairs=np.c_[a, b, data[:, 4]].astype("f4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    extract(parser.parse_args().out)
