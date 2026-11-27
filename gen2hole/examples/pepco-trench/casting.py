"""Recover source-supported cover colour and complete the clipped tread by phase-aligned rotation."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from camera import S, photo


def project(matrix, xy):
    p = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], -1) @ matrix.T
    return p[..., :2] / p[..., 2:]


def warp(img, uv):
    return cv2.remap(
        img,
        uv[..., 0].astype("f4"),
        uv[..., 1].astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def rectified(calibration):
    img = photo()
    n = 641
    r = S["nominal_near_diameter_m"] / 2
    y, x = np.indices((n, n))
    u = (x + 0.5) / n - 0.5
    v = 0.5 - (y + 0.5) / n
    xy = np.stack([u, v], -1) * 2 * r
    uv = project(np.asarray(calibration["homography"]), xy + calibration["near_xz_m"])
    sampled = warp(img, uv)
    source = ((uv >= 1) & (uv < (np.array(img.shape[1::-1]) - 2))).all(2)
    dist = np.hypot(u, v) * 2
    # The central tread is periodic. Align the rotated donor to the observed tread;
    # the smooth outside rim must continue about the geometric centre.
    gray = cv2.cvtColor(sampled, cv2.COLOR_RGB2GRAY).astype("f4")
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 9)
    shared = source & source[::-1, ::-1] & (dist < 0.75)
    yy, xx = np.where(shared)
    yy, xx = yy[::9], xx[::9]
    train = xx < n * 0.55

    def score(dx, dy, mask):
        a, b = xx[mask], yy[mask]
        c, d = n - 1 - a + dx, n - 1 - b + dy
        keep = source[d, c]
        t = hp[b[keep], a[keep]]
        z = hp[d[keep], c[keep]]
        return float(np.corrcoef(t, z)[0, 1])

    offsets = [(dx, dy) for dx in range(-24, 25, 2) for dy in range(-24, 25, 2)]
    dx, dy = max(offsets, key=lambda p: score(*p, train))
    blend = np.clip((0.95 - dist) / 0.11, 0, 1)
    donor = np.stack([n - 1 - x + dx * blend, n - 1 - y + dy * blend], -1)
    alternate = warp(sampled, donor)
    inside = dist <= 1
    if np.any(inside & ~source & (warp(source.astype("f4"), donor) < 0.99)):
        raise ValueError("Missing donor on casting")
    # Blend on the original/rotated boundary only. Source-only outputs retain raw RGB.
    weight = np.clip(distance_transform_edt(source) / 8.0, 0, 1)[..., None]
    atlas = np.uint8(np.rint(sampled * weight + alternate * (1 - weight)))
    # Neutral adaptation is only applied on the transferable model. Preserve luminance
    # differences and scratches; use adjacent asphalt to estimate the blue shade cast.
    patch = img[2080:2160, 1550:1690]
    road = np.median(patch, axis=(0, 1))
    gain = np.mean(road) / road
    atlas = np.uint8(np.clip(np.round(atlas * gain), 0, 255))
    return (
        atlas,
        source & inside,
        dict(
            optical_support_fraction=float((source & inside).sum() / inside.sum()),
            phase_offset_px=[dx, dy],
            heldout_texture_correlation=score(dx, dy, ~train),
            unregistered_correlation=score(0, 0, ~train),
            neutral_gain=gain.tolist(),
        ),
    )


def visible_face(calibration, local_xy):
    uv = project(np.asarray(calibration["homography"]), local_xy + calibration["near_xz_m"])
    return ((uv >= 1) & (uv < [1934, 2590])).all(-1)
