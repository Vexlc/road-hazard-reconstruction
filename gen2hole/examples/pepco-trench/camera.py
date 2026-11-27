"""Register two circular covers to one road plane in the oriented Washington exposure."""

import hashlib, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "survey.json").read_text())
PRINCIPAL = np.array(S["principal_px"])


def photo():
    path = HERE / "references/field.jpg"
    if hashlib.sha256(path.read_bytes()).hexdigest() != S["sha256"]:
        raise ValueError(path)
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def measure(gray, path, band, blur):
    h = gray.shape[0]
    initial = cv2.fitEllipse(np.float32(path))
    (cx, cy), (a, b), angle = initial
    angle = np.deg2rad(angle)
    basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    t = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    unit = np.c_[np.cos(t), np.sin(t)]
    perimeter = (unit * [a / 2, b / 2]) @ basis.T + [cx, cy]
    normal = unit / [a / 2, b / 2] @ basis.T
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    delta = np.arange(-band, band + 0.5, 0.5)
    xy = perimeter[:, None, :] + normal[:, None, :] * delta[None, :, None]
    sample = cv2.remap(
        cv2.GaussianBlur(gray, (0, 0), blur),
        xy[:, :, 0].astype("f4"),
        xy[:, :, 1].astype("f4"),
        cv2.INTER_LINEAR,
    )
    # The far lid is distinctly lighter than road. Its metal-to-road transition is negative.
    derivative = np.gradient(sample, axis=1) * 2
    strength = abs(derivative) if band > 5 else -derivative
    best = strength.argmax(1)
    usable = (
        (best > 0)
        & (best < len(delta) - 1)
        & (perimeter[:, 1] < h - band - 5)
        & (strength[np.arange(len(best)), best] > max(1, np.percentile(strength, 65)))
    )
    pts = xy[np.arange(len(best)), best]
    val = usable & (np.arange(len(t)) % 7 == 2)
    train = usable & ~val

    def signed(q, p):
        x, y, la, lb, rot = q
        T = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        P = (p - [x, y]) @ T
        axes = np.exp([la, lb])
        return ((P**2 / axes**2).sum(1) - 1) / (2 * np.linalg.norm(P / axes**2, axis=1))

    q0 = np.r_[cx, cy, np.log([a / 2, b / 2]), angle]
    q = least_squares(signed, q0, args=(pts[train],), loss="soft_l1", f_scale=1.1).x
    assert np.isfinite(q).all() and val.sum() >= 10
    return (
        q,
        pts,
        train,
        val,
        dict(
            center_px=q[:2].tolist(),
            diameter_axes_px=(2 * np.exp(q[2:4])).tolist(),
            angle_deg=np.rad2deg(q[-1]),
            training=int(train.sum()),
            heldout=int(val.sum()),
            heldout_median_px=float(np.median(abs(signed(q, pts[val])))),
            heldout_p90_px=float(np.percentile(abs(signed(q, pts[val])), 90)),
        ),
    )


def solution(q, radius=0.36):
    f, tilt, roll, H, x1, z1, x2, z2, r2 = q
    n = np.array([np.sin(roll) * np.cos(tilt), -np.cos(roll) * np.cos(tilt), -np.sin(tilt)])
    ex = np.array([1.0, 0, 0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ez = np.cross(n, ex)
    K = np.array([[f, 0, PRINCIPAL[0]], [0, f, PRINCIPAL[1]], [0, 0, 1]])
    M = K @ np.column_stack([ex, ez, -H * n])
    inv = np.linalg.inv(M)
    out = []
    for cx, cz, r in [(x1, z1, radius), (x2, z2, r2)]:
        C = np.array([[1, 0, -cx], [0, 1, -cz], [-cx, -cz, cx**2 + cz**2 - r**2]])
        out.append(inv.T @ C @ inv)
    return out, M


def residual(c, pt):
    a = np.c_[pt, np.ones(len(pt))]
    g = a @ c
    return (g * a).sum(1) / np.maximum(2 * np.linalg.norm(g[:, :2], axis=1), 1e-16)


def solve(prior, obs, summary):
    def error(q):
        cs, _ = solution(q)
        return np.r_[
            residual(cs[0], obs["train1"]) / 2,
            residual(cs[1], obs["train2"]) / 0.8,
            (q[0] - prior) / 500,
        ]

    bounds = (
        [600, -0.05, -0.25, 0.6, -2, 0.2, -4, 2, 0.15],
        [3600, 0.65, 0.25, 3, 2, 6, 4, 40, 0.8],
    )
    trials = []
    for initf in [1800, 2400, 2900]:
        q0 = np.array([initf, 0.22, 0.02, 1.3, -0.12, 1.6, -0.15, 11.0, 0.36])
        _, M = solution(q0)
        back = np.linalg.inv(M)
        pts = [back @ np.r_[row["center_px"], 1.0] for row in summary]
        for i, pos in enumerate(pts):
            q0[4 + 2 * i : 6 + 2 * i] = pos[:2] / pos[2]
        q0 = np.clip(q0, np.array(bounds[0]) + 0.001, np.array(bounds[1]) - 0.001)
        trial = least_squares(error, q0, bounds=bounds, x_scale="jac", loss="soft_l1", max_nfev=600)
        trials.append(trial)
    q = min(trials, key=lambda trial: trial.cost).x
    c, M = solution(q)
    e = [abs(residual(a, obs["hold" + str(i + 1)])) for i, a in enumerate(c)]
    return dict(
        focal_px=q[0],
        pitch_deg=np.rad2deg(q[1]),
        roll_deg=np.rad2deg(q[2]),
        height_m=q[3],
        near_xz_m=q[4:6].tolist(),
        far_xz_m=q[6:8].tolist(),
        far_diameter_m=2 * q[8],
        nominal_near_diameter_m=0.72,
        centre_separation_m=float(np.linalg.norm(q[4:6] - q[6:8])),
        hold_median_px=[float(np.median(t)) for t in e],
        hold_p90_px=[float(np.percentile(t, 90)) for t in e],
        homography=M.tolist(),
    )


def calibrate():
    gray = cv2.cvtColor(photo(), cv2.COLOR_RGB2GRAY).astype("f4")
    obs = {}
    summary = []
    for i, (key, band, blur) in enumerate(
        [("manual_near_px", 12, 2), ("manual_far_px", 3.5, 1.0)], 1
    ):
        _, pts, train, hold, stats = measure(gray, np.asarray(S[key]), band, blur)
        obs["train" + str(i)] = pts[train]
        obs["hold" + str(i)] = pts[hold]
        summary.append(stats)
    result = solve(2200, obs, summary)
    return result, obs, summary
