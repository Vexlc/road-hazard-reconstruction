"""Fuse metric road-aligned heights, segmentation and color from registered fitting frames."""

import warnings

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from pit_captures import project, transform


def plane(samples):
    pts = np.concatenate(samples)
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        active = pts[keep]
        if len(active) < 100:
            raise ValueError('Too little measured pavement to establish the local road plane')
        center = active.mean(0)
        delta = active - center
        normal = np.linalg.eigh(delta.T @ delta)[1][:, 0]
        residual = (pts - center) @ normal
        keep = abs(residual) < max(0.003, 2.8 * 1.4826 * np.median(abs(residual[keep])))
    if normal @ center > 0:
        normal = -normal
    return center, normal, float(np.sqrt(np.mean(residual[keep] ** 2)))


def depth_samples(view, pose):
    ys, xs = np.nonzero(view.mask)
    pad = max(45, int(max(np.ptp(xs), np.ptp(ys)) * 0.65))
    y0, y1 = max(0, ys.min() - pad), min(view.mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(view.mask.shape[1], xs.max() + pad + 1)
    step = max(1, int(np.ceil(np.sqrt((y1 - y0) * (x1 - x0) / 450000))))
    v, u = np.mgrid[y0:y1:step, x0:x1:step]
    keep = view.depth[v, u] > 0
    u, v = u[keep], v[keep]
    xyz = transform(view.xyz(np.c_[u, v], view.depth[v, u]), pose)
    guard = cv2.dilate(view.mask.astype('u1'), np.ones((19, 19), 'u1'))
    return xyz, view.mask[v, u], guard[v, u] == 0


def local_frame(samples):
    center, normal, rms = plane([xyz[road][::4] for xyz, _, road in samples])
    axis = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
    axis /= np.linalg.norm(axis)
    basis = np.column_stack([axis, np.cross(normal, axis), normal])
    bounds = []
    for xyz, mask, _ in samples:
        local = (xyz[mask] - center) @ basis
        bounds.append(np.percentile(local[:, :2], [0.2, 99.8], axis=0))
    lo = np.min([b[0] for b in bounds], axis=0)
    hi = np.max([b[1] for b in bounds], axis=0)
    return center + basis[:, :2] @ ((hi + lo) / 2), basis, hi - lo, rms


def sample_height(xyz, center, basis, grid, pitch, limit):
    points = (xyz - center) @ basis
    extent = np.max(abs(grid), axis=(0, 1)) + 0.018
    inside = (np.abs(points[:, :2]) < extent).all(1) & (abs(points[:, 2]) < limit)
    local = points[inside]
    if len(local) < 80:
        raise ValueError('Too few aligned surface samples inside the patch')
    distance, idx = cKDTree(local[:, :2]).query(grid, k=5, workers=2)
    z = local[idx, 2]
    med = np.median(z, axis=2)
    w = (abs(z - med[:, :, None]) < 0.014) / (distance**2 + (0.5 * pitch) ** 2)
    value = (z * w).sum(2) / w.sum(2)
    valid = distance[:, :, 0] < max(0.006, 2.2 * pitch)
    return np.where(valid, value, np.nan)


def samples_in_image(view, world, pose):
    local = transform(world, np.linalg.inv(pose))
    uv = project(local, view.K).astype('f4')
    h, w = view.mask.shape
    valid = (
        (local[:, :, 2] > 0)
        & (uv[:, :, 0] >= 0)
        & (uv[:, :, 0] < w - 1)
        & (uv[:, :, 1] >= 0)
        & (uv[:, :, 1] < h - 1)
    )
    color = cv2.remap(view.rgb, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.remap(view.mask.astype('u1'), uv, None, cv2.INTER_NEAREST)
    depth = cv2.remap(view.depth, uv, None, cv2.INTER_NEAREST)
    valid &= (depth > 0) & (abs(depth - local[:, :, 2]) < 0.05)
    return color, mask, valid, local[:, :, 2]


def road_texture(color, observed, mask):
    """Quilt nearby photographed intact road at its native rectified texel scale."""
    safe = observed & ~ndimage.binary_dilation(mask, iterations=4)
    clearance = ndimage.distance_transform_edt(np.pad(safe, 1))[1:-1, 1:-1]
    radius = int(min(21, clearance.max() / 1.6))
    if radius < 2:
        raise ValueError('No intact pavement texture around the photographed pit')
    centers = np.column_stack(np.nonzero(clearance > radius * 1.45))
    rng = np.random.default_rng(4201)
    n, m = mask.shape
    span = 2 * radius + 1
    tile = np.hanning(span + 2)[1:-1]
    window = tile[:, None] * tile[None, :]
    result = np.zeros((n + 2 * span, m + 2 * span, 3))
    weights = np.zeros(result.shape[:2])
    for y in range(0, n + span, radius + 1):
        for x in range(0, m + span, radius + 1):
            # A local donor preserves slower illumination and asphalt color changes.
            choices = centers[rng.integers(0, len(centers), 32)]
            cy, cx = choices[np.argmin(np.sum((choices - [y, x]) ** 2, axis=1))]
            patch = color[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
            result[y : y + span, x : x + span] += patch * window[:, :, None]
            weights[y : y + span, x : x + span] += window
    return np.uint8(
        result[radius : radius + n, radius : radius + m]
        / weights[radius : radius + n, radius : radius + m, None]
    )


def fuse(views, poses, spacing=0.0025):
    samples = [depth_samples(v, poses[v.key]) for v in views]
    center, basis, size, rms = local_frame(samples)
    border = np.clip(size.max() * 0.16, 0.055, 0.14)
    width = float(size.max() + 2 * border)
    n = int(np.clip(np.ceil(width / spacing) + 1, 145, 511))
    x = np.linspace(-width / 2, width / 2, n)
    y = -x
    pitch = width / (n - 1)
    grid = np.stack(np.meshgrid(x, y), axis=-1)
    height_limit = max(
        0.16,
        2 * max(np.percentile(abs((xyz[mask] - center) @ basis[:, 2]), 98) for xyz, mask, _ in samples),
    )
    heights = np.stack(
        [sample_height(xyz, center, basis, grid, pitch, height_limit) for xyz, _, _ in samples]
    )
    count = np.isfinite(heights).sum(0)
    supported = count > 0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        height = np.nanmedian(heights, axis=0)
    nearest = ndimage.distance_transform_edt(~supported, return_distances=False, return_indices=True)
    height = ndimage.median_filter(height[tuple(nearest)], 3)
    local = np.dstack([grid, height])
    world = local @ basis.T + center
    weight = np.zeros_like(height)
    rgb = np.zeros((*height.shape, 3))
    vote = np.zeros_like(height)
    for i, view in enumerate(views):
        color, mask, valid, z = samples_in_image(view, world, poses[view.key])
        # Prefer the closer, front-facing photograph and let independent agreeing
        # surfaces share color. A view cannot vote on a cell outside its observed depth.
        cam = np.linalg.inv(poses[view.key])
        normal = basis[:, 2] @ cam[:3, :3].T
        facing = np.clip(-normal[2], 0.12, 1.0)
        w = valid * np.isfinite(heights[i]) * facing * (view.K[0, 0] / np.maximum(z, 0.1)) ** 2
        weight += w
        vote += mask * w
        rgb += color * w[:, :, None]
    observed = weight > 0
    mask = (vote >= 0.5 * weight) & observed
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    labels, num = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = np.isin(labels, np.flatnonzero(sizes >= max(16, 0.002 * sizes.max())))
    if mask.sum() < 100:
        raise ValueError('Fitting frames do not provide a resolved labelled target')
    color = np.uint8(np.clip(rgb / np.maximum(weight[:, :, None], 1e-12), 0, 255))
    nearest = ndimage.distance_transform_edt(~observed, return_distances=False, return_indices=True)
    color = color[tuple(nearest)]
    fill = road_texture(color, observed, mask)
    edge_weight = np.minimum(1.0, ndimage.distance_transform_edt(observed) / 4.0)
    edge_weight[mask] = 1.0
    color = np.uint8(np.round(color * edge_weight[:, :, None] + fill * (1 - edge_weight[:, :, None])))

    def smooth(t):
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)

    gap = ndimage.distance_transform_edt(~supported) * pitch
    to_pit = ndimage.distance_transform_edt(~mask) * pitch
    edge = np.minimum(width / 2 - abs(grid[:, :, 0]), width / 2 - abs(grid[:, :, 1]))
    alpha = smooth((border - to_pit) / (0.55 * border)) * smooth(edge / 0.025) * smooth(1 - gap / 0.018)
    height = height * alpha
    yy, xx = np.nonzero(mask)
    stats = dict(
        width_m=float(np.ptp(x[xx])),
        length_m=float(np.ptp(y[yy])),
        depth_p95_m=float(np.percentile(-height[mask], 95)),
        max_depth_m=float((-height[mask]).max()),
        pit_components=int(ndimage.label(mask)[1]),
        road_fit_rms_m=rms,
        grid_pitch_m=pitch,
        measured_pit_fraction=float(supported[mask].mean()),
        multi_view_pit_fraction=float((count[mask] >= 2).mean()),
        local_origin_in_anchor_m=center.tolist(),
        local_axes_in_anchor_camera=basis.tolist(),
    )
    return (
        dict(
            height=height.astype('f4'),
            defect_mask=mask,
            water_mask=np.zeros_like(mask),
            alpha=alpha.astype('f4'),
            view_count=count.astype('u1'),
            color_observed=observed,
            width=width,
            length=width,
        ),
        color,
        stats,
    )


def photo_texture(views, poses, field, base, frame):
    """Sample raw photographs at their projected pixel density on the recovered surface."""
    width = float(field['width'])
    center = np.asarray(frame['local_origin_in_anchor_m'])
    basis = np.asarray(frame['local_axes_in_anchor_camera'])
    focal_density = []
    for view in views:
        z = transform(center, np.linalg.inv(poses[view.key]))[2]
        focal_density.append(view.K[0, 0] / max(z, 0.1))
    n = int(np.clip(np.ceil(width * max(focal_density)), 384, 1280))
    axis = np.linspace(-width / 2, width / 2, n)
    gx, gy = np.meshgrid(axis, -axis)
    shape = field['height'].shape
    indices = np.stack(
        [(0.5 + gx / width) * (shape[1] - 1), (0.5 - gy / width) * (shape[0] - 1)], -1
    ).astype('f4')

    def sample(a):
        return cv2.remap(a, indices, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    heights = sample(field['height'])
    colors = sample(base)
    dzdy, dzdx = np.gradient(field['height'], -width / (shape[0] - 1), width / (shape[1] - 1))
    nx, ny = -sample(dzdx), -sample(dzdy)
    normals = np.stack([nx, ny, np.ones_like(nx)], -1) @ basis.T
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    covered = np.zeros((n, n), bool)
    winners = np.zeros((n, n), np.uint16)
    for y in range(0, n, 96):
        region = np.s_[y : min(y + 96, n)]
        xyz = np.stack((gx[region], gy[region], heights[region]), -1) @ basis.T + center
        weighted = np.zeros_like(xyz)
        total = np.zeros(xyz.shape[:2])
        previous = np.zeros_like(total)
        for i, view in enumerate(views):
            pose = poses[view.key]
            local = transform(xyz, np.linalg.inv(pose))
            uv = project(local, view.K).astype('f4')
            truth = cv2.remap(view.depth, uv, None, cv2.INTER_NEAREST)
            im = cv2.remap(view.rgb, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            z = local[:, :, 2]
            error = np.abs(truth - z)
            valid = (z > 0) & (truth > 0) & (error < 0.045)
            to_eye = pose[:3, 3] - xyz
            cos = np.maximum(0, (normals[region] * to_eye).sum(2) / np.linalg.norm(to_eye, axis=2))
            rate = (view.K[0, 0] / max(focal_density)) / np.maximum(z, 0.1)
            quality = valid * cos * rate**2 * np.exp(-0.5 * (error / 0.012) ** 2)
            w = quality**3
            weighted += im * w[:, :, None]
            total += w
            better = w > previous
            winners[region][better] = i
            previous = np.maximum(w, previous)
        good = total > 1.0e-8
        colors[region][good] = np.uint8(np.clip(np.rint(weighted[good] / total[good, None]), 0, 255))
        covered[region] = good
    pit = cv2.remap(field['defect_mask'].astype('u1'), indices, None, cv2.INTER_NEAREST) > 0
    stats = dict(
        size_px=n,
        projected_texture_pitch_m=width / (n - 1),
        observed_pit_fraction=float(covered[pit].mean()),
        dominant_pit_texels={
            v.key: int(np.sum(covered & pit & (winners == i))) for i, v in enumerate(views)
        },
    )
    return colors, stats
