"""Bidirectional plane sweep restricted to pixels visible through both apertures."""

import argparse
from pathlib import Path
import cv2
import numpy as np
from geometry import HERE, SIZE, fit


def gray(n):
    im = cv2.resize(
        cv2.imread(str(HERE / "references" / (n + ".jpg"))),
        tuple(SIZE),
        interpolation=cv2.INTER_AREA,
    )
    # Log luminance separates dark masonry features even when the photographed rim is bright.
    lum = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (np.log1p(lum.astype("f4")) * 47).astype("f4")


def sweep(c0, c1, a, b, poly, other):
    mask = np.zeros(a.shape, "u1")
    other_mask = mask.copy()
    cv2.fillPoly(mask, [np.int32(np.rint(poly))], 1)
    cv2.fillPoly(other_mask, [np.int32(np.rint(other))], 1)
    left, top, w, h = cv2.boundingRect(np.float32(poly))
    x0, y0 = left - 7, top - 7
    yy, xx = np.indices((h + 14, w + 14))
    px = np.stack([xx + x0, yy + y0], -1)
    ray = c0.rays(px)
    a = a[y0 : y0 + h + 14, x0 : x0 + w + 14]
    mask = mask[y0 : y0 + h + 14, x0 : x0 + w + 14]
    levels = np.arange(-0.035, -1.42, -0.0075)
    frames = np.full((len(levels), *xx.shape), -2, "f2")

    def blur(v):
        return cv2.blur(v, (11, 11))

    am = blur(a)
    var = np.maximum(0, blur(a * a) - am * am)
    for i, z in enumerate(levels):
        w = c0.eye + ray * ((z - c0.eye[2]) / ray[..., 2])[..., None]
        q = c1.project(w).astype("f4")
        bb = cv2.remap(b, q[..., 0], q[..., 1], cv2.INTER_LINEAR)
        om = cv2.remap(other_mask, q[..., 0], q[..., 1], cv2.INTER_NEAREST)
        bm = blur(bb)
        den = np.sqrt(var * np.maximum(0, blur(bb * bb) - bm * bm))
        val = (blur(a * bb) - am * bm) / np.maximum(den, 0.01)
        frames[i] = np.where((blur((mask * om).astype("f4")) > 0.94) & (den > 9), val, -2)
    best = np.argmax(frames, axis=0)
    peak = frames[best, yy, xx].astype("f4")
    for i in range(-6, 7):
        frames[np.clip(best + i, 0, len(levels) - 1), yy, xx] = -2
    second = frames.max(0).astype("f4")
    return dict(
        z=levels[best].astype("f4"),
        valid=(peak > 0.72) & (peak - second > 0.075) & (best > 0) & (best < len(levels) - 1),
        score=peak,
        offset=np.array([x0, y0]),
        px=px,
    )


def solve(out):
    cams, reg = fit()
    im = [gray(s) for s in ("detail", "street")]
    fields = []
    for i in (0, 1):
        j = 1 - i
        fields.append(
            sweep(
                cams[i], cams[j], im[i], im[j], reg["traces"][i]["bore"], reg["traces"][j]["bore"]
            )
        )
    data = {}
    for i, f in enumerate(fields):
        g = fields[1 - i]
        v = cams[i].plane(f["px"], f["z"])
        uv = np.int32(np.rint(cams[1 - i].project(v) - g["offset"]))
        x, y = uv.transpose(2, 0, 1)
        inside = (x >= 0) & (y >= 0) & (x < g["z"].shape[1]) & (y < g["z"].shape[0])
        x = x.clip(0, g["z"].shape[1] - 1)
        y = y.clip(0, g["z"].shape[0] - 1)
        valid = f["valid"] & inside & g["valid"][y, x] & (abs(f["z"] - g["z"][y, x]) <= 0.023)
        print(i, int(valid.sum()), flush=True)
        data.update(
            {
                f"{n}{i}": value
                for n, value in dict(
                    z=f["z"], valid=valid, score=f["score"], offset=f["offset"]
                ).items()
            }
        )
    np.savez_compressed(out, **data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    solve(p.parse_args().out)
