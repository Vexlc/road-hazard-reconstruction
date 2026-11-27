"""Re-extract the three sparse correspondence tables from local hash-checked JPEGs."""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from sources import photos


def match(folder, out):
    photos(folder)
    out.mkdir(parents=True, exist_ok=True)
    photos_gray = [
        cv2.imread(str(folder / (name + ".jpg")), 0) for name in ("near", "wide", "third")
    ]
    near, far = photos_gray[:2]
    torch.set_grad_enabled(False)
    corners = np.float32([[299, 0], [424, 0], [455, 505], [334, 505]])
    wide = np.float32([[449, 273], [466, 265], [575, 380], [552, 395]])
    warp = cv2.getPerspectiveTransform(wide * far.shape[1] / 720, corners * near.shape[1] / 720)
    rev = cv2.warpPerspective(far, warp, near.shape[::-1])

    def enhance(im):
        return cv2.createCLAHE(2.4, (12, 10)).apply(im)

    ext = SuperPoint(max_num_keypoints=2400).eval().cuda()
    lg = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()
    allrows = []
    for j, (x, y, w, h) in enumerate(
        [
            (0, 0, near.shape[1], near.shape[0]),
            (300, 12, 460, 685),
            (0, 50, 780, 590),
            (378, 200, 510, 510),
        ]
    ):
        for inv in (False, True):
            imgs = [enhance(im[y : y + h, x : x + w]) for im in [near, rev]]
            if inv:
                imgs[0] = imgs[0][::-1, ::-1]
            feat = [
                ext.extract(
                    torch.from_numpy(im.copy()[None, None]).cuda().float() / 255, resize=930
                )
                for im in imgs
            ]
            pred = rbd(lg({"image0": feat[0], "image1": feat[1]}))
            idx = pred["matches"].cpu().numpy()
            uv = [rbd(k)["keypoints"].cpu().numpy()[idx[:, i]] for i, k in enumerate(feat)]
            if inv:
                uv[0] = np.array(imgs[0].shape[::-1]) - 1 - uv[0]
            a, b = [q + [x, y] for q in uv]
            if len(b):
                b = cv2.perspectiveTransform(b[:, None, :], np.linalg.inv(warp))[:, 0]
                allrows.extend(np.column_stack([a, b, pred["scores"].cpu().numpy()]))
    sel = []
    for row in sorted(allrows, key=lambda row: -row[-1]):
        if not any(
            np.linalg.norm(row[:2] - q[:2]) < 3 or np.linalg.norm(row[2:4] - q[2:4]) < 1.5
            for q in sel
        ):
            sel.append(row)
    a = np.asarray(sel)
    hold = np.arange(len(a)) % 5 == 2
    u, v = a[:, :2].copy(), a[:, 2:4].copy()
    F, _ = cv2.findFundamentalMat(u[~hold], v[~hold], cv2.USAC_MAGSAC, 1.7, 0.9999, 50000)
    aa = np.c_[u, np.ones(len(u))]
    bb = np.c_[v, np.ones(len(u))]
    g = aa @ F.T
    j = bb @ F
    sampson = abs((g * bb).sum(1)) / np.sqrt((g[:, :2] ** 2 + j[:, :2] ** 2).sum(1))

    np.savez_compressed(out / "ab.npz", u=u, v=v, hold=hold, score=a[:, -1], error=sampson)
    ext = SuperPoint(max_num_keypoints=2800).eval().cuda()
    net = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()

    def fit(i, j, pts, points, regions, name):
        orig, src = photos_gray[i], photos_gray[j]

        def run(H):
            warped = cv2.warpPerspective(src, H, orig.shape[::-1])
            rows = []
            for box in regions:
                x, y, w, h = np.rint(np.array(box) * orig.shape[1] / 720).astype(int)
                ft = []
                for im in (orig, warped):
                    roi = cv2.createCLAHE(2.5, (14, 14)).apply(im[y : y + h, x : x + w])
                    ft.append(
                        ext.extract(
                            torch.from_numpy(roi[None, None]).cuda().float() / 255, resize=1080
                        )
                    )
                result = rbd(net({"image0": ft[0], "image1": ft[1]}))
                ids = result["matches"].cpu().numpy()
                uv = [
                    rbd(f)["keypoints"].cpu().numpy()[ids[:, k]] + [x, y] for k, f in enumerate(ft)
                ]
                if len(ids):
                    uv[1] = cv2.perspectiveTransform(uv[1][:, None, :], np.linalg.inv(H))[:, 0]
                    rows.extend(np.column_stack([*uv, result["scores"].cpu().numpy()]))
            return rows

        H = cv2.getPerspectiveTransform(
            np.float32(points) * src.shape[1] / 720, np.float32(pts) * orig.shape[1] / 720
        )
        rows = run(H)
        r = np.array(rows)
        H, _ = cv2.findHomography(r[:, 2:4], r[:, :2], cv2.USAC_MAGSAC, 4, maxIters=30000)
        rows += run(H)
        q = []
        # Occupancy hashing bounds duplicate lookup at high match density.
        from collections import defaultdict

        cells = [defaultdict(list) for _ in (0, 1)]
        for row in sorted(rows, key=lambda r: -r[-1]):
            dup = False
            for k in (0, 1):
                uv = row[2 * k : 2 * k + 2]
                x, y = np.floor(uv / 3).astype(int)
                if any(
                    np.linalg.norm((xy - uv)) < 2.5
                    for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    for xy in cells[k][x + dx, y + dy]
                ):
                    dup = True
                    break
            if dup:
                continue
            q.append(row)
            for k in (0, 1):
                uv = row[2 * k : 2 * k + 2]
                cells[k][tuple(np.floor(uv / 3).astype(int))].append(uv)
        q = np.array(q)
        hold = np.arange(len(q)) % 5 == 2
        u, v = q[:, :2].copy(), q[:, 2:4].copy()
        F, _ = cv2.findFundamentalMat(u[~hold], v[~hold], cv2.USAC_MAGSAC, 1.5, 0.9999, 55000)
        aa = np.c_[u, np.ones(len(u))]
        bb = np.c_[v, np.ones(len(v))]
        l = aa @ F.T
        r = bb @ F
        err = abs((l * bb).sum(1)) / np.sqrt((l[:, :2] ** 2 + r[:, :2] ** 2).sum(1))
        np.savez_compressed(out / (name + ".npz"), u=u, v=v, score=q[:, 4], hold=hold, error=err)
        print(name, len(q), (err < 1).sum(), flush=True)

    fit(
        0,
        2,
        [[298, 0], [422, 0], [450, 505], [341, 505]],
        [[339, 154], [388, 156], [353, 387], [308, 384]],
        [
            (0, 0, 720, 505),
            (240, 0, 264, 502),
            (271, 135, 220, 275),
            (68, 88, 390, 370),
            (390, 50, 320, 420),
        ],
        "ac",
    )
    fit(
        2,
        1,
        [[333, 322], [365, 161], [494, 42], [74, 120]],
        [[529, 364], [464, 275], [472, 145], [170, 268]],
        [
            (0, 0, 720, 480),
            (170, 120, 335, 230),
            (210, 150, 230, 190),
            (0, 0, 510, 198),
            (253, 0, 400, 440),
        ],
        "cb",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    match(**vars(p.parse_args()))
