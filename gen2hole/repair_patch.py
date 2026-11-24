"""Reconstruct and retint a shallow asphalt repair without importing source-road RGB."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d


ENGINE = Path(__file__).resolve().parent
EXAMPLE = ENGINE / 'examples/transverse-repair/P202.json'


def read(path):
    return Path(path).read_text(encoding='utf8')


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def rgb_material(field, base):
    return np.uint8(np.clip(np.asarray(base) * np.exp(field['log_gain'])[..., None], 0, 255).round())


def reconstruct(photo, spec, dest):
    import cv2

    if hashlib.sha256(photo.read_bytes()).hexdigest() != spec['source_sha256']:
        raise ValueError('Reference hash differs from the traced photograph')
    with Image.open(photo) as image:
        if list(image.size) != spec['image_size']:
            raise ValueError('Reference size differs from the traced photograph')
        rgb = np.array(image.convert('RGB'))
    width, length = spec['width_m'], spec['length_m']
    cols, rows = 801, 1 + round(800 * length / width)
    pitch_x, pitch_y = width / (cols - 1), length / (rows - 1)
    quad = np.float32(spec['rectification_quad_far_left_clockwise_px'])
    top = np.float32([[0, 0], [cols-1, 0], [cols-1, rows-1], [0, rows-1]])
    homography = cv2.getPerspectiveTransform(quad, top)
    def rectify(points):
        return cv2.perspectiveTransform(np.float32(points)[None], homography)[0]
    observed = cv2.warpPerspective(rgb, homography, (cols, rows))
    luminance = np.log(np.maximum(cv2.cvtColor(observed, cv2.COLOR_RGB2GRAY), 1).astype('f4'))
    body = np.zeros((rows, cols), 'u1')
    cv2.fillPoly(body, [rectify(spec['boundary_px']).round().astype('i4')], 1)
    inside = distance_transform_edt(body, sampling=(pitch_y, pitch_x))
    support = np.clip(inside / .018, 0, 1)
    joint = rectify(spec['joint_px'])
    cols_at = np.arange(cols)
    seed_y = np.interp(cols_at, *joint.T)
    offsets = np.arange(-9, 10)[:, None]
    trial_y = np.rint(seed_y + offsets).astype(int).clip(0, rows-1)
    brightness = gaussian_filter(luminance, (2.0, 2.6))
    picked = trial_y[(brightness[trial_y, cols_at] + .00055*offsets**2).argmin(0), cols_at]
    ridge = gaussian_filter1d(picked.astype(float), 2.3)
    signed_m = (np.arange(rows)[:, None] - ridge) * pitch_y
    slow = gaussian_filter(luminance, (13, 13))
    contrast = (slow - brightness)[ridge.round().astype(int), cols_at]
    strength = np.clip(gaussian_filter1d(contrast, 2.8) / .28, .12, 1)
    ends = np.clip(np.minimum(cols_at-joint[0, 0], joint[-1, 0]-cols_at) * pitch_x / .085, 0, 1)
    sigma = spec['design_joint_sigma_m'] * (.70 + .45*strength)
    groove = ends * strength * np.exp(-.5 * (signed_m / sigma)**2)
    lip = ends * strength * np.exp(-.5 * ((signed_m+.047) / .022)**2)
    base = np.median(observed[(inside > .05) & (abs(signed_m) > .09)], axis=0)
    grain = np.clip(luminance - slow, -.32, .20)
    gain = support * (.047 + .74*grain - .20*groove - .075*np.exp(-inside/.023))
    height = support * (.0025*np.tanh(inside/.065) - spec['design_joint_depth_m']*groove
                        + spec['design_lip_height_m']*lip + .0016*np.tanh(4*gaussian_filter(grain, 3)))
    dest.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(dest/'field.npz', width=width, length=length, height=height.astype('f4'),
                        log_gain=gain.astype('f4'), alpha=support.astype('f4'), defect_mask=body.astype(bool),
                        joint=groove.astype('f4'), source_base=base)
    overlay = rgb.copy()
    cv2.polylines(overlay, [np.int32(spec['boundary_px'])], True, (40, 213, 170), 2, cv2.LINE_AA)
    fitted = np.c_[cols_at[ends > 0], ridge[ends > 0]]
    fitted = cv2.perspectiveTransform(np.float32(fitted)[None], np.linalg.inv(homography))[0]
    cv2.polylines(overlay, [fitted.round().astype('i4')], False, (255, 173, 58), 2, cv2.LINE_AA)
    Image.fromarray(overlay[755:968, 310:1210]).save(dest/'trace.jpg', quality=95)
    Image.fromarray(rgb[755:968, 310:1210]).save(dest/'reference-crop.jpg', quality=95)
    write(dest/'reconstruction.json', dict(id=spec['id'], source_sha256=spec['source_sha256'],
          scale_kind='Uncalibrated design dimensions and relief; outline and joint traced from photograph',
          grid_extent_m=[width, length], footprint_m2=float(body.sum()*pitch_x*pitch_y),
          relief_range_mm=[float(height.min()*1000), float(height.max()*1000)],
          source_to_grid=homography.tolist(), joint_search_max_px_in_grid=9))
    return base


def mesh(dest, output, base, cache=None):
    sys.path.insert(0, str(ENGINE/'recipes/series'))
    from mesh_io import Mesh, grid_faces

    with np.load(dest/'field.npz') as data:
        width, length = float(data['width']), float(data['length'])
        material = rgb_material(data, base)
        sample_rows = np.unique(np.r_[np.arange(0, material.shape[0], 2), material.shape[0]-1])
        sample_cols = np.unique(np.r_[np.arange(0, material.shape[1], 2), material.shape[1]-1])
        uv = np.stack(np.meshgrid(sample_cols/(material.shape[1]-1), 1-sample_rows/(material.shape[0]-1)), -1)
        sub = np.ix_(sample_rows, sample_cols)
        points = np.dstack(((uv[..., 0]-.5)*width, (uv[..., 1]-.5)*length, data['height'][sub]))
        faces = grid_faces(*uv.shape[:2])
        selected = data['defect_mask'][sub].ravel()[faces].all(1)
        model = Mesh()
        key = model.material('Road-relative asphalt', material, .96, 0, 0, lossless_color=True)
        model.add('Fracture lip and exposed bed', points, faces[selected], uv, key)
        output.parent.mkdir(exist_ok=True, parents=True)
        if cache:
            record = json.loads(read(dest/'reconstruction.json'))
            meta = dict(id=record['id'], label='横向修补带与断续接缝', category='pothole',
                        seed=307303, scale_kind=record['scale_kind'], source_sha256=record['source_sha256'])
            model.save(output.parent, cache, meta, {k: data[k] for k in ('width','length','height','alpha','defect_mask')})
        else:
            output.write_bytes(model.glb_bytes())


def export(dest, output, base, cfg, cache=None):
    args = [cfg['mesh_python'], str(Path(__file__).resolve()), 'mesh', str(dest), str(output), '--rgb', *map(str, base)]
    if cache:
        args.extend(['--cache', str(cache)])
    subprocess.run(args, check=True)


def retint_splats(dest, output, base):
    from plyfile import PlyData

    source = dest/'asset.ply'
    if source.resolve() == output.resolve():
        raise ValueError('Keep the source PLY; choose a separate output')
    with np.load(dest/'field.npz') as field:
        gain = np.asarray(base) / np.maximum(field['source_base'], 1)
    ply = PlyData.read(source, mmap=False)
    points = ply['vertex'].data
    for channel, ratio in enumerate(gain):
        name = f'f_dc_{channel}'
        points[name] = ratio * points[name] + (ratio-1)*.5/.28209479177387814
        for band in range(3):
            points[f'f_rest_{3*channel+band}'] *= ratio
    output.parent.mkdir(parents=True, exist_ok=True)
    ply.write(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('photo', type=Path)
    build.add_argument('dest', type=Path)
    build.add_argument('--spec', type=Path, default=EXAMPLE)
    build.add_argument('--config', type=Path, default=ENGINE/'config/local.json')
    build.add_argument('--splats', action='store_true')
    paint = commands.add_parser('mesh', help='Retint the same geometry with a local median sRGB (0–255)')
    paint.add_argument('dest', type=Path)
    paint.add_argument('output', type=Path)
    paint.add_argument('--rgb', type=float, nargs=3, required=True)
    paint.add_argument('--cache', type=Path, help=argparse.SUPPRESS)
    splats = commands.add_parser('splats', help='Retint the degree-1 SH colours without moving any splats')
    splats.add_argument('dest', type=Path)
    splats.add_argument('output', type=Path)
    splats.add_argument('--rgb', type=float, nargs=3, required=True)
    args = parser.parse_args()
    if args.command in ('mesh', 'splats'):
        if not np.all((np.array(args.rgb) >= 0) & (np.array(args.rgb) <= 255)):
            parser.error('--rgb channels must be in [0,255]')
        if args.command == 'splats':
            retint_splats(args.dest, args.output, args.rgb)
        else:
            mesh(args.dest, args.output, args.rgb, args.cache)
        return
    cfg, spec = json.loads(read(args.config)), json.loads(read(args.spec))
    base = reconstruct(args.photo, spec, args.dest)
    cache = args.dest/'mesh_cache'
    export(args.dest, args.dest/'asset.glb', base, cfg, cache)
    if args.splats:
        sys.path.insert(0, str(ENGINE/'recipes/common'))
        from export_splats import make_one
        make_one(args.dest, cache)


if __name__ == '__main__':
    main()
