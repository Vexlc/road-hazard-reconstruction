"""Re-extract vertical plane-sweep patches using the fitted metric cameras."""

import argparse
from pathlib import Path
import cv2
import numpy as np
from camera import Survey
from sources import photos


def run(folder, out):
    photos(folder)
    out.mkdir(parents=True, exist_ok=True)
    a, b, c = Survey().fit()

    def sweep(ref, tag, path):
        im0 = cv2.imread(str(folder / "near.jpg"), 0).astype("f4")
        image = cv2.imread(str(path), 0).astype("f4")
        region = a.at(np.array([[280, 0], [480, 0], [480, 443], [280, 443]]) * 1021 / 720, 2)
        pix = ref.project(region)
        lo = np.floor(pix.min(0)).astype(int) - 14
        hi = np.ceil(pix.max(0)).astype(int) + 14
        yy, xx = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
        coords = np.stack([xx, yy], -1)
        rays = ref.rays(coords.reshape(-1, 2))
        shape = xx.shape
        wide = image[yy, xx]
        near = cv2.GaussianBlur(im0, (0, 0), 1.6 if tag == "ab" else 0.75)

        def smooth(x):
            return cv2.GaussianBlur(x, (9, 9), 1.65)

        wm = smooth(wide)
        var = np.maximum(smooth(wide * wide) - wm * wm, 0)
        heights = np.arange(-5.0, 8.01, 0.08)
        cc = []
        for z in heights:
            xyz = ref.eye + rays * ((z - ref.eye[2]) / rays[:, 2])[:, None]
            uv = a.project(xyz).reshape(*shape, 2).astype("f4")
            tex = cv2.remap(near, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
            tm = smooth(tex)
            cc.append(
                (smooth(wide * tex) - wm * tm)
                / np.sqrt((var + 3) * np.maximum(smooth(tex * tex) - tm * tm + 3, 3))
            )
        cost = np.stack(cc)
        k = np.argmax(cost, axis=0)
        i, j = np.indices(shape)
        z = heights[k]
        peak = cost[k, i, j]
        margin = peak - np.max(np.where(abs(heights[:, None, None] - z) > 1.15, cost, -2), axis=0)
        sharp = (
            cost[np.minimum(k + 4, len(heights) - 1), i, j]
            + cost[np.maximum(k - 4, 0), i, j]
            - 2 * peak
        )
        raw = (
            (k > 5)
            & (k < len(heights) - 6)
            & (peak > 0.75)
            & (margin > 0.08)
            & (var > 38)
            & (sharp < -0.004)
        )
        xyz = ref.eye + rays * ((z.ravel() - ref.eye[2]) / rays[:, 2])[:, None]
        xyz = xyz.reshape(*shape, 3)
        uv = a.project(xyz)
        inside = (
            (uv[..., 0] > 285 * 1021 / 720)
            & (uv[..., 0] < 469 * 1021 / 720)
            & (uv[..., 1] > 0)
            & (uv[..., 1] < 440 * 1021 / 720)
        )
        take = raw & inside
        # Retain the highest score in each 3-by-3 native pixel cell.
        rr, cc = np.where(take)
        score = peak[take] + margin[take]
        keys = (rr // 3) * (1 + shape[1] // 3) + cc // 3
        keep = np.lexsort((-score, keys))
        keep = keep[np.r_[True, np.diff(keys[keep]) != 0]]
        rr, cc = rr[keep], cc[keep]
        np.savez_compressed(
            out / (tag + "-patches.npz"),
            xyz=xyz[rr, cc],
            u=uv[rr, cc],
            v=coords[rr, cc],
            peak=peak[rr, cc],
            margin=margin[rr, cc],
        )
        print(tag, len(rr), flush=True)

    sweep(b, "ab", folder / "wide.jpg")
    sweep(c, "ac", folder / "third.jpg")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    run(**vars(p.parse_args()))
