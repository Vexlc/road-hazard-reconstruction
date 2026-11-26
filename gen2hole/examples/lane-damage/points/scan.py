"""Read a PointPSSN survey and establish a local frame using labelled surrounding road."""

from io import BytesIO
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import cv2


def load(z, split, index):
    xyz = np.loadtxt(BytesIO(z.read(f"dataset/Data/{split}/data_{index}.txt")))
    labels = np.loadtxt(BytesIO(z.read(f"dataset/Label/{split}/label_{index}.txt")))
    if (
        xyz.shape != (5000, 3)
        or labels.shape != (5000,)
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("Unexpected PointPSSN point or label layout")
    return xyz, labels.astype(bool)


def frame(xyz, mask, train):
    road = xyz[~mask & train]
    good = np.ones(len(road), bool)
    for _ in range(6):
        cloud = road[good]
        origin = cloud.mean(0)
        _, axes = np.linalg.eigh(np.cov((cloud - origin).T))
        up = axes[:, 0]
        residual = abs((road - origin) @ up)
        good = residual < max(0.002, 3.5 * np.median(residual))
    if np.median((xyz[mask & train] - origin) @ up) > 0:
        up = -up
    defect = xyz[mask & train]
    center = defect.mean(0)
    center -= ((center - origin) @ up) * up
    tangent = defect - center - np.outer((defect - center) @ up, up)
    _, axes = np.linalg.eigh(np.cov(tangent.T))
    along = axes[:, -1]
    if along[np.argmax(abs(along))] < 0:
        along = -along
    across = np.cross(up, along)
    basis = np.column_stack([along, across, up])
    return center, basis


def classify(raw_xy, raw_class, q, k=7):
    d, ix = cKDTree(raw_xy).query(q, k=k)
    sigma = np.maximum(d[:, -1:] * 0.44, 0.0015)
    weights = np.exp(-0.5 * (d / sigma) ** 2)
    votes = (weights * raw_class[ix]).sum(1) / weights.sum(1)
    return votes, d[:, 0]


def summary(xyz, mask):
    c, b = frame(xyz, mask, np.ones(len(mask), bool))
    v = (xyz - c) @ b
    lo, hi = np.percentile(v[mask, :2], [0.3, 99.7], axis=0)
    size = hi - lo
    x, y = np.meshgrid(
        np.linspace(lo[0] - 0.04, hi[0] + 0.04, 65),
        np.linspace(lo[1] - 0.04, hi[1] + 0.04, 65),
    )
    q = np.c_[x.ravel(), y.ravel()]
    votes, d = classify(v[:, :2], mask, q)
    cover = (d < 0.018).reshape(x.shape)
    fg = (votes.reshape(x.shape) > 0.5) & cover
    road_sigma = np.median(abs(v[~mask, 2]))
    h = np.percentile(-v[mask, 2], 90)
    cnt, _ = cv2.findContours(np.uint8(fg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(c) for c in cnt)
    big = max(cnt, key=cv2.contourArea)
    conv = cv2.contourArea(cv2.convexHull(big))
    rim = (cv2.dilate(fg.astype("u1"), np.ones((7, 7), "u1")) > 0) & ~fg
    return dict(
        size=size.tolist(),
        depth=float(h),
        sigma=float(road_sigma),
        fraction=float(mask.mean()),
        n=int(mask.sum()),
        up=b[:, 2].tolist(),
        rim_support=float(cover[rim].mean()),
        fill=float(cv2.contourArea(big) / max(conv, 1)),
        major_share=float(cv2.contourArea(big) / max(area, 1)),
        shape=fg.astype("u1"),
        v=v,
    )


def local_height(points, query, k=16):
    """Fit local planes with a robust loss so isolated deep returns do not leave spikes."""
    distances, idx = cKDTree(points[:, :2]).query(query, k=k)
    xy = points[idx, :2] - query[:, None, :]
    design = np.concatenate([xy, np.ones((*xy.shape[:2], 1))], axis=2)
    z = points[idx, 2]
    scale = np.maximum(0.001, distances[:, -1:] * 0.48)
    weight = np.exp(-0.5 * (distances / scale) ** 2)
    base = weight.copy()
    for _ in range(3):
        matrix = design.transpose(0, 2, 1) @ (weight[:, :, None] * design)
        matrix[:, 0, 0] += 0.00001
        matrix[:, 1, 1] += 0.00001
        rhs = (design * (weight * z)[:, :, None]).sum(1)
        fit = np.linalg.solve(matrix, rhs[..., None])[..., 0]
        residual = abs(z - np.einsum("nki,ni->nk", design, fit))
        width = np.maximum(0.0015, 1.8 * np.median(residual, axis=1, keepdims=True))
        weight = base * np.minimum(1, width / np.maximum(residual, 1e-9))
    return fit[:, 2], distances[:, 0]


def refine_votes(xy, labels, query, height, votes, k, fraction):
    score = votes.copy()
    boundary = (votes > 0.2) & (votes < 0.8)
    if fraction and boundary.any():
        q = query[boundary]
        bed, _ = local_height(xy[labels], q, k)
        road, _ = local_height(xy[~labels], q, k)
        depth_fraction = np.clip(
            (road - height[boundary]) / np.maximum(road - bed, 0.008), 0, 1
        )
        score[boundary] = (1 - fraction) * votes[boundary] + fraction * depth_fraction
    return score


def choose_fit(xyz, label, fold):
    """Cross-validate neighbourhood and rim evidence on three training-only spatial folds."""
    scores = {}
    for f in (1, 4, 8):
        train = (fold != 0) & (fold != f)
        valid = fold == f
        origin, basis = frame(xyz, label, train)
        p = (xyz - origin) @ basis
        t = p[train]
        q = p[valid, :2]
        truth = label[valid]
        votes, _ = classify(t[:, :2], label[train], q, k=9)
        for k in (5, 8, 12, 16, 24):
            h, _ = local_height(t, q, k)
            for fraction in (0.0, 0.25):
                pred = refine_votes(t, label[train], q, h, votes, k, fraction) > 0.5
                s = scores.setdefault((k, fraction), np.zeros(4))
                s += [
                    abs(h[truth] - p[valid, 2][truth]).sum(),
                    truth.sum(),
                    (pred & truth).sum(),
                    (pred | truth).sum(),
                ]

    def metric(item):
        s = scores[item]
        return round(s[0] / s[1], 8), -s[2] / s[3]

    k, fraction = min(scores, key=metric)
    return (
        k,
        fraction,
        {
            "neighbours": k,
            "rim_height_weight": fraction,
            "folds": [1, 4, 8],
            "inner_mae_mm": float(
                1000 * scores[k, fraction][0] / scores[k, fraction][1]
            ),
            "inner_iou": float(scores[k, fraction][2] / scores[k, fraction][3]),
        },
    )


def reconstruct(xyz, mask, pitch, apron_m=0.072):
    # Spatially contiguous holdouts are fixed in raw scanner coordinates before fitting the road plane.
    block = np.floor(xyz[:, [0, 2]] / 0.028).astype("i8")
    fold = (block[:, 0] + 5 * block[:, 1]) % 11
    hold = fold == 0
    k, fraction, fit = choose_fit(xyz, mask, fold)
    c, b = frame(xyz, mask, ~hold)
    points = (xyz - c) @ b
    valid = points[mask & ~hold, :2]
    low = valid.min(0) - apron_m
    high = valid.max(0) + apron_m
    origin = (low + high) * 0.5
    c = c + b[:, :2] @ origin
    points[:, :2] -= origin
    radius = np.ceil((high - low) * 0.5 / pitch).astype(int)
    x = np.arange(-radius[0], radius[0] + 1) * pitch
    y = np.arange(radius[1], -radius[1] - 1, -1) * pitch
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx, yy], -1)
    query = xy.reshape(-1, 2)
    trained = points[~hold]
    value, dist = local_height(trained, query, k)
    votes, _ = classify(trained[:, :2], mask[~hold], query, k=9)
    votes = refine_votes(trained, mask[~hold], query, value, votes, k, fraction)
    nearest, _ = cKDTree(trained[:, :2]).query(trained[:, :2], k=2)
    limit = max(0.022, min(0.05, np.percentile(nearest[:, -1], 90) * 4))
    # Only accept supported defect observations. Sparse margins converge to the surrounding grade.
    support = (dist < limit).reshape(xx.shape)
    soft = np.clip((limit - dist) / (0.40 * limit), 0, 1)
    fade = np.clip(
        (0.071 - distance_transform_edt(~(votes.reshape(xx.shape) > 0.5)) * pitch)
        / 0.045,
        0,
        1,
    )
    z = (value * soft).reshape(xx.shape) * fade
    z = z.astype("f4")
    return dict(
        xy=xy,
        z=z,
        mask=(votes.reshape(xx.shape) > 0.5) & support,
        support=support,
        confidence=votes.reshape(xx.shape),
        origin=c,
        basis=b,
        points=points,
        hold=hold,
        spatial_fold=fold,
        limit=limit,
        fit=fit,
    )
