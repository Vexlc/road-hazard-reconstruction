"""Recover a metric road surface by sweeping two RGB photographs across candidate heights."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from fuse_pit import road_texture
from pit_captures import pinhole, polygons


@dataclass
class Photo:
    key: str
    rgb: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    sources: dict


def read_pair(spec, source, catalog, base):
    keys = spec['fit']
    if (
        len(keys) != 2
        or len(set(keys)) != 2
        or any(not isinstance(k, str) or not k or Path(k).name != k or k in ('.', '..') for k in keys)
    ):
        raise ValueError('Two distinct RGB frame names are required')
    photos = []

    def add(key, data, files, params, polygon):
        photo = Image.open(BytesIO(data['rgb'])).convert('RGB')
        if polygon:
            mask = polygons(data['mask'], photo.size)
        else:
            raw_mask = np.array(Image.open(BytesIO(data['mask'])).convert('L'))
            mask = raw_mask > (int(raw_mask.max()) / 2)
        if mask.shape != photo.size[::-1] or mask.sum() < 32 or not (~mask).any():
            raise ValueError('Mark the target on a registered mask for ' + key)
        k = pinhole(params)
        if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError('Invalid RGB camera intrinsics')
        photos.append(Photo(key, np.array(photo), mask, k, files))

    if spec['source'] == 'PothRGBD':
        index = json.loads(catalog.read_text())
        with ZipFile(source / 'rgbd/images.zip') as rgb, ZipFile(source / 'rgbd/labels.zip') as labels:
            for key in keys:
                row = index['rgbd_pothole_' + key]
                files = {k: row[k] for k in ('rgb', 'mask')}
                data = {
                    role: z.read(files[role]['filename']) for role, z in [('rgb', rgb), ('mask', labels)]
                }
                if any(sha256(b).hexdigest() != files[role]['sha256'] for role, b in data.items()):
                    raise ValueError('RGB source changed ' + key)
                add(key, data, files, spec['fx_fy_cx_cy'], True)
    elif spec['source'] == 'FanStereo':
        directory = source / 'stereo' / spec['dataset']
        for key in keys:
            paths = {
                'rgb': directory / 'rgb' / (key + '.png'),
                'mask': directory / 'label' / (key + '.png'),
            }
            data = {k: p.read_bytes() for k, p in paths.items()}
            records = {
                k: dict(filename=str(paths[k].relative_to(source)), sha256=sha256(v).hexdigest())
                for k, v in data.items()
            }
            add(key, data, records, spec['fx_fy_cx_cy'], False)
    elif spec['source'] == 'files':
        for key in keys:
            frame = spec['frames'][key]
            data = {k: (base / frame[k]).read_bytes() for k in ('rgb', 'mask')}
            records = {k: dict(filename=frame[k], sha256=sha256(v).hexdigest()) for k, v in data.items()}
            add(key, data, records, frame.get('fx_fy_cx_cy', spec.get('fx_fy_cx_cy')), False)
    else:
        raise ValueError('Use two PothRGBD, FanStereo or local RGB images')
    return photos


def camera_rays(uv, k):
    return np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T


def fit_road(first, second, ruler):
    detector = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.012)
    matcher = cv2.BFMatcher()
    detections = [
        detector.detectAndCompute(cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY), None)
        for view in [first, second]
    ]
    if any(d is None or len(d) < 30 for _, d in detections):
        raise ValueError('Too few road texture features')

    def mutual_candidates(a, b):
        return {
            m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(a, b, k=2) if m.distance < 0.79 * n.distance
        }

    d, e = [item[1] for item in detections]
    forward, reverse = mutual_candidates(d, e), mutual_candidates(e, d)
    matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    if len(matches) < 30:
        raise ValueError('Insufficient matching road landmarks')
    pts = []
    outside = np.ones(len(matches), bool)
    for frame, indices, (kp, _) in zip((first, second), np.array(matches).T, detections):
        uv = np.float32([kp[i].pt for i in indices])
        pts.append(uv)
        x, y = np.clip(uv.round().astype(int), 0, np.array(frame.mask.shape[::-1]) - 1).T
        outside &= cv2.dilate(frame.mask.astype('u1'), np.ones((11, 11), 'u1'))[y, x] == 0
    p, q = pts
    if outside.sum() < 22 or (~outside).sum() < 10:
        raise ValueError('Need both intact road and target matches')
    H, inliers = cv2.findHomography(p[outside], q[outside], cv2.USAC_MAGSAC, 1.5)
    if H is None or inliers.sum() < 22:
        raise ValueError('Cannot register the surrounding pavement')
    nr1 = camera_rays(p, first.K)
    calibrated = np.linalg.solve(second.K, H @ first.K)
    _, rotations, offsets, normals = cv2.decomposeHomographyMat(calibrated, np.eye(3))
    candidates = []
    for R, t, n in zip(rotations, offsets, normals):
        t = t.ravel()
        n = n.ravel()
        if np.linalg.norm(t) < 0.025 or (nr1 @ n > 0.12).mean() < 0.80:
            continue
        F = np.linalg.inv(second.K).T @ np.cross(np.eye(3), t).T @ R @ np.linalg.inv(first.K)
        lines = np.c_[p, np.ones(len(p))] @ F.T
        residual = abs((lines * np.c_[q, np.ones(len(q))]).sum(1)) / np.maximum(
            1e-10, np.linalg.norm(lines[:, :2], axis=1)
        )
        # Off-plane correspondences distinguish the two possible plane decompositions.
        candidates.append((float(np.minimum(residual[~outside], 4).mean()), R, t, n))
    if not candidates:
        raise ValueError('No forward-facing plane with sufficient camera translation')
    candidates.sort(key=lambda item: item[0])
    error, R, t, n = candidates[0]
    if error > 2.5 or (len(candidates) > 1 and candidates[1][0] - error < 0.04):
        raise ValueError('Camera motion is ambiguous at the target')
    endpoints = np.asarray(ruler['pixels'], float)
    length = float(ruler['length_m'])
    if (
        endpoints.shape != (2, 2)
        or not np.isfinite(endpoints).all()
        or not np.isfinite(length)
        or length <= 0
        or (endpoints < 0).any()
        or (endpoints > np.array(first.mask.shape[::-1]) - 1).any()
        or np.linalg.norm(endpoints[1] - endpoints[0]) < 15
    ):
        raise ValueError('Supply the separation of two distinct road pixels in metres')
    x, y = endpoints.round().astype(int).T
    if first.mask[y, x].any():
        raise ValueError('The scale reference must lie on the intact road')
    ray = camera_rays(endpoints, first.K)
    den = ray @ n
    if min(den) <= 0.05:
        raise ValueError('Reference pixels do not see the road plane')
    distance = length / np.linalg.norm(np.diff(ray / den[:, None], axis=0))
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('Invalid scale reference')
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t * distance
    return (
        pose,
        n,
        distance,
        dict(
            mutual_features=len(p),
            road_inliers=int(inliers.sum()),
            target_region_features=int((~outside).sum()),
            clipped_epipolar_error_px=error,
            candidate_errors_px=[v[0] for v in candidates],
            baseline_m=float(np.linalg.norm(pose[:3, 3])),
        ),
    )


def plane_grid(photo, normal, distance):
    y, x = np.nonzero(photo.mask)
    rays = camera_rays(np.c_[x, y], photo.K)
    den = rays @ normal
    if min(den) <= 0.05:
        raise ValueError('Target crosses the road horizon')
    cloud = rays * (distance / den[:, None])
    up = -normal
    axis = np.array([1.0, 0.0, 0.0]) - up * up[0]
    axis /= np.linalg.norm(axis)
    basis = np.stack([axis, np.cross(up, axis), up], axis=1)
    uv = cloud @ basis[:, :2]
    lo, hi = uv.min(0), uv.max(0)
    size = max(hi - lo)
    border = np.clip(0.19 * size, 0.06, 0.18)
    width = size + 2 * border
    center = basis[:, :2] @ ((lo + hi) / 2) + normal * distance
    count = int(np.clip(np.ceil(width / 0.0025) + 1, 149, 443))
    x = np.linspace(-width / 2, width / 2, count)
    grid = np.stack(np.meshgrid(x, -x), -1)
    return center, basis, (grid @ basis[:, :2].T + center).astype('f4'), width, border


def coordinates(view, plane, up, pose):
    # Homogeneous image coordinates are affine in the candidate relief for every grid cell.
    base = ((plane @ pose[:3, :3].T + pose[:3, 3]) @ view.K.T).astype('f4')
    slope = (up @ pose[:3, :3].T @ view.K.T).astype('f4')
    return base, slope


def warp(coords, height):
    base, slope = coords
    p = base + slope * height if np.ndim(height) == 0 else base + slope * height[:, :, None]
    return np.ascontiguousarray(p[:, :, :2] / np.maximum(0.01, p[:, :, 2:3]), dtype='f4')


def sample(img, uv, method=cv2.INTER_LINEAR):
    return cv2.remap(img, uv, None, method, borderMode=cv2.BORDER_REPLICATE)


def in_frame(uv, view, margin=2):
    h, w = view.mask.shape
    return (
        (uv[:, :, 0] >= margin)
        & (uv[:, :, 0] < w - 1 - margin)
        & (uv[:, :, 1] >= margin)
        & (uv[:, :, 1] < h - 1 - margin)
    )


def matching_photo(photo):
    h, w = photo.mask.shape
    factor = min(1.0, 960 / max(h, w))
    if factor == 1:
        return photo
    size = (round(w * factor), round(h * factor))
    scale = np.diag([size[0] / w, size[1] / h, 1.0])
    scale[:2, 2] = 0.5 * (np.diag(scale)[:2] - 1)
    return Photo(
        photo.key,
        cv2.resize(photo.rgb, size, interpolation=cv2.INTER_AREA),
        cv2.resize(photo.mask.astype('u1'), size, interpolation=cv2.INTER_NEAREST) > 0,
        scale @ photo.K,
        photo.sources,
    )


def match_in_image(first, second, pose, normal, distance, depth_span):
    if not np.isfinite(depth_span) or depth_span <= 0.01:
        raise ValueError('Search depth must exceed one centimetre')
    h, w = first.mask.shape
    y, x = np.indices((h, w), dtype='f4')
    ray = camera_rays(np.c_[x.ravel(), y.ravel()], first.K).reshape(h, w, 3).astype('f4')
    den = (ray @ normal).astype('f4')
    # Multiplying by the road-ray denominator eliminates the division for every candidate height.
    rotated = (ray @ pose[:3, :3].T @ second.K.T).astype('f4')
    translated = den[:, :, None] * (second.K @ pose[:3, 3]).astype('f4')
    gray0, gray1 = [cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY).astype('f4') / 255 for v in (first, second)]

    def avg(a):
        return cv2.boxFilter(a, -1, (9, 9))

    heights = np.linspace(-depth_span, min(0.04, depth_span * 0.3), 112, dtype='f4')
    step = float(heights[1] - heights[0])
    spread = max(3, round(0.015 / step))
    out = [np.zeros((h, w), 'f4') for _ in range(3)]
    for y0 in range(0, h, 72):
        bottom = min(y0 + 72, h)
        start = max(y0 - 4, 0)
        stop = min(bottom + 4, h)
        region = np.s_[start:stop]
        take = np.s_[y0 - start : bottom - start]
        gray = gray0[region]
        ma = avg(gray)
        va = np.maximum(avg(gray * gray) - ma * ma, 0)
        costs = []
        for level in heights:
            p = rotated[region] * (distance - level) + translated[region]
            uv = np.ascontiguousarray(p[:, :, :2] / np.maximum(p[:, :, 2:3], 1e-5))
            b = sample(gray1, uv)
            mb = avg(b)
            vb = np.maximum(avg(b * b) - mb * mb, 0)
            corr = (avg(gray * b) - ma * mb) / np.sqrt(np.maximum(va * vb, 1.0e-10))
            valid = in_frame(uv, second, 4) & (den[region] > 0.05) & (np.minimum(va, vb) > 0.00008)
            costs.append(np.where(valid, 1 - np.clip(corr, -1, 1), 2)[take])
        c = np.stack(costs)
        idx = c.argmin(0)

        def at(i):
            return np.take_along_axis(c, np.clip(i, 0, len(c) - 1)[None], axis=0)[0]

        best = at(idx)
        left, right = at(idx - 1), at(idx + 1)
        sub = np.clip(0.5 * (left - right) / np.maximum(left + right - 2 * best, 1e-6), -0.5, 0.5)
        z = heights[idx] + step * sub
        other = np.full_like(best, 2.0)
        for i, one in enumerate(c):
            np.minimum(other, one, out=other, where=abs(idx - i) >= spread)
        out[0][y0:bottom] = z
        out[1][y0:bottom] = np.where((idx > 0) & (idx < len(c) - 1), 1 - best, -1)
        out[2][y0:bottom] = other - best
    height, correlation, gap = out
    mapped = rotated * (distance - height[:, :, None]) + translated
    uv = np.ascontiguousarray(mapped[:, :, :2] / np.maximum(mapped[:, :, 2:3], 1e-5))
    valid = (correlation > 0.68) & (gap > 0.008) & (abs(cv2.medianBlur(height, 5) - height) < 0.018)
    return height, valid, uv, ray


def relief_grid(photos, pose, normal, distance, plane, center, basis, width, depth_span):
    left, right = map(matching_photo, photos)
    front, good, uv, ray = match_in_image(left, right, pose, normal, distance, depth_span)
    n2 = pose[:3, :3] @ normal
    back, reverse_good, _, _ = match_in_image(
        right, left, np.linalg.inv(pose), n2, distance + pose[:3, 3] @ n2, depth_span
    )
    agreement = abs(front - sample(back, uv, cv2.INTER_NEAREST)) < 0.012
    source_direct = good & agreement & (sample(reverse_good.astype('u1'), uv, cv2.INTER_NEAREST) > 0)
    # A match near an occluding rim can land in intact pavement from the other viewpoint.
    expanded = cv2.dilate(right.mask.astype('u1'), np.ones((5, 5), 'u1'))
    source_direct &= left.mask & (sample(expanded, uv, cv2.INTER_NEAREST) > 0)
    height = np.zeros_like(front)
    labels, count = ndimage.label(left.mask)
    fractions = []
    for component in range(1, count + 1):
        region = labels == component
        if region.sum() < 16:
            continue
        usable = source_direct & region
        fraction = float(usable.sum() / region.sum())
        fractions.append(fraction)
        if fraction < 0.18 or usable.sum() < 24:
            raise ValueError('Target lacks reciprocal RGB matches')
        nearest = ndimage.distance_transform_edt(~usable, return_distances=False, return_indices=True)
        filled = front[tuple(nearest)]
        for _ in range(32):
            filled = np.where(usable, front, cv2.GaussianBlur(filled, (5, 5), 1))
        height[region] = filled[region]
    # Recover points in the first camera coordinates, including the intact exterior on its datum.
    denominator = ray @ normal
    points = ray * ((distance - height) / np.maximum(denominator, 0.01))[:, :, None]
    local = (points - center) @ basis
    grid = (plane - center) @ basis
    valid = (denominator > 0.05) & (abs(local[:, :, :2]) < width * 0.56).all(-1)
    tree = cKDTree(local[:, :, :2][valid])
    separation, idx = tree.query(grid[:, :, :2].reshape(-1, 2), k=4, workers=2)
    # Local interpolation over projected samples closes occluded walls between the visible rim and floor.
    weight = 1 / (separation**2 + (width / len(plane) * 0.3) ** 2)
    weight /= weight.sum(1)[:, None]

    def blend(a):
        return np.sum(a[valid][idx] * weight, axis=1).reshape(plane.shape[:2])

    z = blend(height)
    # At a steep wall several source pixels project to the same ground cell. Keep the
    # connected footprint of each photographed pit instead of isolated nearest-neighbour votes.
    projected = warp(coordinates(left, plane, basis[:, 2], np.eye(4)), z)
    inside = in_frame(projected, left, 0)
    outline = np.zeros(z.shape, bool)
    for component in range(1, count + 1):
        source_region = labels == component
        if source_region.sum() < 16:
            continue
        votes = (blend(source_region) >= 0.5) & inside
        connected = ndimage.binary_closing(votes, iterations=2) & inside
        regions, number = ndimage.label(connected, np.ones((3, 3), bool))
        if not number:
            raise ValueError('A labelled region did not project to the road grid')
        largest = np.argmax(np.bincount(regions.ravel())[1:]) + 1
        outline |= regions == largest
    conf = (blend(source_direct) > 0.5) & (separation[:, 0].reshape(z.shape) < 2.5 * width / len(plane))
    z[~outline] = 0
    return (
        z,
        outline,
        conf,
        source_direct,
        dict(
            reciprocal_tolerance_m=0.012,
            per_component_matched_fraction=fractions,
            matched_pixels=int(source_direct.sum()),
            target_pixels=int(left.mask.sum()),
            matching_size_px=list(left.mask.shape[::-1]),
        ),
    )


def build_pair(spec, source, catalog, base, bundle):
    photos = read_pair(spec, source, catalog, base)
    first, second = photos
    transform, normal, distance, alignment = fit_road(first, second, spec['ruler'])
    center, basis, plane, width, border = plane_grid(first, normal, distance)
    coords = [coordinates(v, plane, basis[:, 2], pose) for v, pose in zip(photos, [np.eye(4), transform])]
    height, outline, conf, source_direct, support = relief_grid(
        photos,
        transform,
        normal,
        distance,
        plane,
        center,
        basis,
        width,
        float(spec.get('search_depth_m', 0.22)),
    )
    maps = [warp(c, height) for c in coords]
    masks = [
        (sample(v.mask.astype('u1'), uv, cv2.INTER_NEAREST) > 0) & in_frame(uv, v, 0)
        for v, uv in zip(photos, maps)
    ]
    overlap = (masks[0] & masks[1]).sum() / max(1, (masks[0] | masks[1]).sum())
    supported = float(conf[outline].mean())
    pitch = width / (len(height) - 1)
    if overlap < 0.40 or supported < 0.25:
        raise ValueError(
            f'Insufficient two-photo evidence: mask IoU {overlap:.3f}, matched fraction {supported:.3f}'
        )
    observed = in_frame(maps[0], first, 0)
    color = sample(first.rgb, maps[0])
    grain = road_texture(color, observed, outline)
    border_fade = np.clip(ndimage.distance_transform_edt(np.pad(observed, 1))[1:-1, 1:-1] / 7, 0, 1)
    color = np.uint8(np.rint(color * border_fade[:, :, None] + grain * (1 - border_fade[:, :, None])))
    to_target = ndimage.distance_transform_edt(~outline) * pitch
    alpha = np.clip((border - to_target) / (0.55 * border), 0, 1)
    edge = np.minimum.outer(
        np.minimum(np.arange(len(height)), np.arange(len(height))[::-1]),
        np.minimum(np.arange(len(height)), np.arange(len(height))[::-1]),
    )
    alpha *= np.clip(edge * pitch / 0.025, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    height *= alpha
    iy, ix = np.where(outline)
    stats = dict(
        width_m=float(np.ptp(ix) * pitch),
        length_m=float(np.ptp(iy) * pitch),
        depth_p95_m=float(np.percentile(-height[outline], 95)),
        max_depth_m=float(-height[outline].min()),
        grid_pitch_m=pitch,
        local_origin_in_anchor_m=center.tolist(),
        local_axes_in_anchor_camera=basis.tolist(),
        rgb_supported_target_fraction=supported,
        second_mask_iou=float(overlap),
        pit_components=int(ndimage.label(outline, np.ones((3, 3), bool))[1]),
    )
    if stats['depth_p95_m'] <= 0.003:
        raise ValueError('Target lacks resolved concavity')
    root = bundle / 'calibration'
    refs = bundle / 'references'
    key = spec['id']
    field = dict(
        height=height.astype('f4'),
        defect_mask=outline,
        alpha=alpha.astype('f4'),
        water_mask=np.zeros_like(outline),
        width=width,
        length=width,
        rgb_supported=conf,
    )
    np.savez_compressed(root / (key + '.npz'), **field)
    Image.fromarray(source_direct.astype('u1') * 255).save(root / (key + '_correspondence.png'))
    for index, photo in enumerate(photos, 1):
        name = key if index == 1 else key + '_second'
        Image.fromarray(photo.rgb).save(refs / (name + '.png'))
        Image.fromarray(photo.mask.astype('u1') * 255).save(root / (name + '_outline.png'))
    Image.fromarray(grain).save(refs / (key + '_pavement.jpg'), quality=95)
    # Reproject the full-resolution source onto the recovered GLB surface.
    tex_n = int(np.clip(width * first.K[0, 0] / max(0.1, center[2]), 448, 1280))
    tex_height = cv2.resize(height, (tex_n, tex_n), interpolation=cv2.INTER_LINEAR)
    world = cv2.resize(plane, (tex_n, tex_n), interpolation=cv2.INTER_LINEAR)
    tex_map = warp(coordinates(first, world, basis[:, 2], np.eye(4)), tex_height)
    weight = cv2.resize(border_fade, (tex_n, tex_n), interpolation=cv2.INTER_LINEAR)[:, :, None]
    texture = np.uint8(
        np.rint(weight * sample(first.rgb, tex_map) + (1 - weight) * cv2.resize(grain, (tex_n, tex_n)))
    )
    Image.fromarray(texture).save(refs / (key + '_ortho.png'))
    record = dict(
        spec=spec,
        inputs={
            p.key: dict(files=p.sources, K=p.K.tolist(), size_px=list(p.rgb.shape[1::-1])) for p in photos
        },
        poses_view_to_anchor={first.key: np.eye(4).tolist(), second.key: np.linalg.inv(transform).tolist()},
        alignment=alignment,
        support=support,
        reference_road_plane=dict(normal_away_from_camera=normal.tolist(), distance_m=distance),
        geometry=stats,
        scale=spec['ruler'],
        texture_source=first.key,
    )

    def save(path, obj):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    save(root / (key + '_capture.json'), record)
    source_image = key + '.png'
    save(
        root / (key + '.json'),
        dict(
            id=key,
            label=spec['label'],
            category='pothole',
            state='pair_reconstruction',
            family=key,
            recipe_version='rgb-parallax-v1',
            source_image='references/' + source_image,
            source_sha256=sha256((refs / source_image).read_bytes()).hexdigest(),
            geometry_source='Two RGB views, road homography decomposition and dense plane-sweep matching',
            scale_kind='Calibrated pinhole cameras and one known distance on the undamaged road',
            measurement_record='calibration/' + key + '_capture.json',
            seed=307101 + int(key[1:]),
            **stats,
        ),
    )
    print(
        key,
        'two-photo support',
        round(supported, 3),
        'mask overlap',
        round(overlap, 3),
        'P95 mm',
        round(stats['depth_p95_m'] * 1000, 1),
        flush=True,
    )
