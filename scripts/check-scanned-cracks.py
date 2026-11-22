"""Compare delivered crack meshes with withheld displacement texels and registered source normals."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from scanned_crack_regions import MIN_AREA_M2, enclosed_regions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import Surface, glb_triangles, read_json, save_json, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def angular(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.einsum('ij,ij->i', a, b).clip(-1, 1)))


def errors(a, b):
    err = np.abs(a - b) * 1000
    return dict(
        pixels=len(err),
        median_mm=float(np.median(err)),
        p95_mm=float(np.percentile(err, 95)),
        rmse_mm=float(np.sqrt(np.mean(err**2))),
    )


def cast(mesh, meta, row, col):
    pitch = meta['source_pitch_m']
    half = meta['intervals'] / 2
    origins = np.column_stack([(col - half) * pitch, (half - row) * pitch, np.full(len(row), 0.3)])
    d, part, normal = mesh.cast(origins, np.broadcast_to([0.0, 0.0, -1.0], origins.shape))
    require(np.isfinite(d).all(), 'Missing surface in the core of ' + meta['id'])
    return 0.3 - d, part, normal


def source_height(meta, raw, rr, cc):
    c, r = meta['origin_col_row']
    a, b, k = meta['road_plane_pixel']
    s = meta['material_scale']
    col, row = cc + c, rr + r
    authored = (raw[row, col].astype(float) / 65535 - s['midlevel']) * s['scale_m'] - (a * col + b * row + k)
    return authored * meta.get('normal_scale_fit', {}).get('factor', 1.0)


def source_path(meta, inputs, role):
    record = meta['source_files'][role]
    path = inputs / meta['material'] / record['url'].split('/')[-1]
    require(sha256(path) == record['sha256'], 'Source hash mismatch ' + str(path))
    return path


def measure_cells(mask, pitch):
    _, areas = enclosed_regions(mask, pitch)
    return dict(min_area_m2=MIN_AREA_M2, count=len(areas), areas_m2=areas.tolist())


def control(samples, mask, pitch):
    # Give the V-shaped control the same observed intact shoulders and fit its
    # slope and depth cap using only the supplied geometry vertices.
    shoulder = ~mask
    radius = 0.027 / pitch
    mass = ndi.gaussian_filter(shoulder.astype(float), radius)
    z = ndi.gaussian_filter(samples * shoulder, radius) / np.maximum(mass, 1e-8)
    half = np.maximum(0, ndi.distance_transform_edt(mask) - 0.5) * pitch
    depth = (z - samples)[mask]
    widths = half[mask]
    fit = least_squares(
        lambda a: 1000 * (np.minimum(widths * a[0], a[1]) - depth),
        [0.6, 0.013],
        bounds=([0, 0], [15, 0.15]),
        loss='soft_l1',
    )
    return z, z - np.minimum(half * fit.x[0], fit.x[1]), fit.x.tolist()


def bilinear(grid, rr, cc, step):
    return ndi.map_coordinates(grid, [rr / step, cc / step], order=1, mode='nearest')


def evaluate(run, inputs, meta):
    key = meta['id']
    b = run / 'series'
    folder = b / 'assets/pothole' / key
    dest = run / 'evaluation' / key
    dest.mkdir(parents=True, exist_ok=True)
    n = meta['intervals']
    s = meta['step']
    pitch = meta['source_pitch_m']
    raw = cv2.imread(str(source_path(meta, inputs, 'Displacement')), cv2.IMREAD_UNCHANGED)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    normals = cv2.imread(str(source_path(meta, inputs, 'nor_gl')), cv2.IMREAD_UNCHANGED)
    require(normals.shape == (*raw.shape, 3), 'Expected a registered tangent normal map')
    color = np.array(Image.open(source_path(meta, inputs, 'Diffuse')).convert('RGB'))
    c, r = meta['origin_col_row']
    color = color[r : r + n + 1, c : c + n + 1]
    require(np.array_equal(color, np.array(Image.open(b / meta['source_image']))), 'RGB crop moved')
    primitive = list(glb_triangles(folder / 'asset.glb', names=True, appearance=True))
    span = meta['metric_source_span_m']
    for name, xyz, _, uv, _ in primitive:
        expected = np.c_[0.5 + xyz[:, 0] / span, 0.5 - xyz[:, 1] / span]
        require(np.max(np.abs(expected - uv)) < 1e-6, 'Texture coordinates moved on ' + name)
    images = [p[4] for p in primitive]
    require(all(a == images[0] for a in images), 'Inconsistent source texture on road/cracks')
    decoded = np.array(Image.open(BytesIO(images[0])).convert('RGB'))
    rgb_mae = float(np.abs(decoded.astype('i2') - color).mean())
    mesh = Surface(folder / 'asset.glb')
    outline = np.array(Image.open(b / 'calibration' / (key + '_outline.png'))) > 0
    with np.load(b / 'calibration' / (key + '_samples.npz')) as f:
        samples = f['height']
    small_mask = outline[::s, ::s]
    rr, cc = np.indices(samples.shape)
    require(
        np.max(np.abs(samples - source_height(meta, raw, rr * s, cc * s))) < 1e-7,
        'Exported input differs from source',
    )
    flat, v, params = control(samples, small_mask, pitch * s)
    rr, cc = np.indices(outline.shape, dtype='i4')
    core = np.minimum.reduce([rr, cc, n - rr, n - cc]) * pitch > max(0.015, 5 * pitch * s)
    held = core & ((rr % s != 0) | (cc % s != 0))
    # Evaluate the entire withheld population, including narrow fissures and the block shoulders.
    row, col = np.where(held)
    z_chunks = []
    part_chunks = []
    angle_chunks = []
    flat_chunks = []
    factor = meta.get('normal_scale_fit', {}).get('factor', 1.0)
    plane_x, plane_y, _ = np.array(meta['road_plane_pixel']) * factor
    for i in range(0, len(row), 80000):
        rr0, cc0 = row[i : i + 80000], col[i : i + 80000]
        z, part, norm = cast(mesh, meta, rr0, cc0)
        z_chunks.append(z)
        part_chunks.append(part)
        # Undo the fitted road plane before comparing with source-space normals.
        estimate = norm.astype(float)
        estimate[:, 0] -= plane_x / pitch * estimate[:, 2]
        estimate[:, 1] += plane_y / pitch * estimate[:, 2]
        truth_norm = normals[r + rr0, c + cc0, ::-1].astype(float) * 2 - 1
        angle_chunks.append(angular(estimate, truth_norm))
        flat_chunks.append(
            angular(np.broadcast_to([-plane_x / pitch, plane_y / pitch, 1.0], estimate.shape), truth_norm)
        )
    z = np.concatenate(z_chunks)
    parts = np.concatenate(part_chunks)
    angles = np.concatenate(angle_chunks)
    ref = source_height(meta, raw, row, col)
    target = outline[row, col]
    pred = parts == 2
    flat_z = bilinear(flat, row, col, s)
    v_z = bilinear(v, row, col, s)
    gr, gc = row + r, col + c
    # The estimator uses the opposite parity and keeps its Gaussian window inside each fit block.
    fresh = (
        target
        & ((gr // 128 + gc // 128) % 2 == 1)
        & (gr % 128 >= 16)
        & (gr % 128 < 112)
        & (gc % 128 >= 16)
        & (gc % 128 < 112)
    )
    normal_holdout = (
        dict(
            pixels=int(fresh.sum()),
            median_deg=float(np.median(angles[fresh])),
            p95_deg=float(np.percentile(angles[fresh], 95)),
        )
        if fresh.any()
        else None
    )
    scores = {}
    for name, mask in [('crack', target), ('road', ~target)]:
        scores[name] = dict(
            mesh=errors(z[mask], ref[mask]),
            shoulder_control=errors(flat_z[mask], ref[mask]),
            fitted_groove=errors(v_z[mask], ref[mask]),
            normal_error_median_deg=float(np.median(angles[mask])),
            flat_normal_error_median_deg=float(np.median(np.concatenate(flat_chunks)[mask])),
        )
    errmap = np.full(outline.shape, np.nan, 'f4')
    errmap[row, col] = 1000 * (z - ref)
    # The display grid is separate from evaluation and includes all source pixels on this section.
    if meta['material'] == 'asphalt_02':
        global_row = np.full(201, 938)
        global_col = np.arange(140, 341)
    else:
        global_row = np.arange(1000, 1261)
        global_col = np.full(261, 1220)
    line_r, line_c = global_row - r, global_col - c
    inside = (line_r > 0) & (line_r < n) & (line_c > 0) & (line_c < n)
    if not inside.all():
        # A supplied custom crop can use a transverse line through its largest marked band.
        y, x = np.unravel_index(ndi.distance_transform_edt(small_mask).argmax(), small_mask.shape)
        rr0, cc0 = y * s, x * s
        line_c = np.arange(max(1, cc0 - 96), min(n, cc0 + 97))
        line_r = np.full_like(line_c, rr0)
    obs, _, _ = cast(mesh, meta, line_r, line_c)
    shoulder = np.linspace(np.median(obs[:7]), np.median(obs[-7:]), len(obs))
    section_depth = float(np.max(shoulder - obs)) * 1000
    section = np.stack(
        [line_c, line_r, source_height(meta, raw, line_r, line_c), obs, bilinear(v, line_r, line_c, s)], 1
    )
    np.savez_compressed(dest / 'comparison.npz', error_mm=errmap, section=section)
    parent_result = None
    if meta.get('parent'):
        parent = read_json(b / 'assets/pothole' / meta['parent'] / 'metadata.json')
        require(
            meta['source_files'] == parent['source_files']
            and meta['road_plane_pixel'] == parent['road_plane_pixel'],
            'Nested scans have different reference data',
        )
        pc, pr = parent['origin_col_row']
        py, px = row + r - pr, col + c - pc
        keep = target & ((px % parent['step'] != 0) | (py % parent['step'] != 0))
        outer = Surface(b / 'assets/pothole' / parent['id'] / 'asset.glb')
        rz, _, _ = cast(outer, parent, py[keep], px[keep])
        parent_result = dict(
            parent=parent['id'],
            parent_error=errors(rz, ref[keep]),
            detail_error=errors(z[keep], ref[keep]),
            glb_disagreement=errors(z[keep], rz),
            source_origin_in_parent_px=[c - pc, r - pr],
        )
    result = dict(
        glb_sha256=sha256(folder / 'asset.glb'),
        capture=meta['material'],
        source_url=meta['source_page'],
        source_texel_m=pitch,
        mesh_pitch_m=pitch * s,
        relative_vertical_factor=factor,
        material_scale_m=meta['material_scale']['scale_m'],
        effective_scale_m=factor * meta['material_scale']['scale_m'],
        holdout_crack_normals=normal_holdout,
        authored_displacement_error=errors(z[target], ref[target] / factor),
        section_depth_below_shoulders_mm=section_depth,
        section_endpoints_xy_px=[[int(line_c[0]), int(line_r[0])], [int(line_c[-1]), int(line_r[-1])]],
        withheld_source_texels=int(held.sum()),
        source_crack_vs_mesh_part_iou=float((target & pred).sum() / (target | pred).sum()),
        fitted_groove_slope_depth_m=params,
        metrics=scores,
        enclosed_islands=measure_cells(outline, pitch),
        texture_rgb_mae_0_255=rgb_mae,
        matched_detail=parent_result,
    )
    require(rgb_mae == 0, 'Exported GLB texture differs from scan')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = read_json(a.run / 'series/assets/asset_manifest.json')
    result = {}
    for meta in rows:
        result[meta['id']] = evaluate(a.run, a.inputs, meta)
        v = result[meta['id']]
        m = v['metrics']['crack']
        print(
            meta['id'],
            v['enclosed_islands']['count'],
            'enclosed islands;',
            m['mesh'],
            m['fitted_groove'],
            flush=True,
        )
    save_json(a.out, result)
