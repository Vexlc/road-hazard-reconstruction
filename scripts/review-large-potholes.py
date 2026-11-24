"""Measure exported P160–P189 triangles, draw their metric planforms and render a 3D preview."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, sha256

NAMES = dict(bowled='碗状坑', wheel_rut='车辙坑', joined='连通坑', crescent='月牙坑', cluster='群坑',
             spall='浅层剥落', patch_break='补丁破损', deep_break='深层破损', slab_corner='板角破损',
             waterlogged='积水坑')
BG = '#f7f6f0'
INK = '#183c40'
SOFT = '#476469'
CM = np.array([1., 5., 15., 30., 60.])
PALETTE = np.array([[162, 203, 183], [105, 175, 162], [53, 136, 143], [35, 92, 119], [30, 49, 84]])


@lru_cache()
def font(size, bold=False):
    weight = 'Bold' if bold else 'Regular'
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-' + weight + '.ttc', size, index=2)


def label(d, xy, text, size, color=INK, bold=False, anchor='lt'):
    d.text(xy, text, font=font(size, bold), fill=color, anchor=anchor)


def depth_color(cm):
    return np.stack([np.interp(cm, CM, column) for column in PALETTE.T], axis=-1).astype('u1')


def measure(glb, threshold):
    # Clip each bed triangle at -threshold and sum its horizontal projection.
    # A triangle with exactly one vertex on one side contributes a similar corner triangle.
    area = 0.
    for name, vertices, faces in glb_triangles(glb, names=True):
        if name == 'Water surface':
            if vertices[:, 2].max() > -threshold:
                raise ValueError('Water surface crosses the measurement plane')
            continue
        tri = vertices[faces].astype('f8')
        size = .5 * np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2])
        lo, mid, hi = np.sort(tri[:, :, 2] + threshold, axis=1).T
        fraction = (hi <= 0).astype('f8')
        low = (lo <= 0) & (mid > 0)
        high = (mid <= 0) & (hi > 0)
        fraction[low] = lo[low] ** 2 / ((mid[low] - lo[low]) * (hi[low] - lo[low]))
        fraction[high] = 1 - hi[high] ** 2 / ((hi[high] - mid[high]) * (hi[high] - lo[high]))
        area += size @ fraction
    return float(area)


def footprint(root, row, box=(296, 292), px_per_m=31):
    with np.load(root / row['surface']) as field:
        z = field['height']
        pitch = float(field['width']) / (z.shape[1] - 1)
    r, c = np.where(z <= -.01)
    # Centre each silhouette while keeping a single pixel-to-metre ratio.
    yy, xx = np.indices(box[::-1])
    dx = (xx - (box[0] - 1) / 2) / px_per_m
    dy = (yy - (box[1] - 1) / 2) / px_per_m
    cm = -100 * map_coordinates(z, [dy / pitch + .5 * (r.min() + r.max()),
                                    dx / pitch + .5 * (c.min() + c.max())],
                                order=1, mode='constant')
    if np.any(cm[[0, -1]] >= 1) or np.any(cm[:, [0, -1]] >= 1):
        raise ValueError('Clipped footprint: ' + row['id'])
    rgb = np.full((*xx.shape, 3), [240, 241, 233], dtype='u1')
    grid = (np.abs(dx - np.round(dx)) < .017) | (np.abs(dy - np.round(dy)) < .017)
    rgb[grid] = [219, 225, 216]
    mask = cm >= 1
    rgb[mask] = depth_color(cm[mask])
    return Image.fromarray(rgb)


def overview(library, groups, destination):
    size = (296, 292)
    width, block, pitch = 1936, 966, 306
    header, row_height = 174, 370
    page = Image.new('RGB', (width, header + 5 * row_height + 15), BG)
    d = ImageDraw.Draw(page)
    label(d, (32, 24), '大面积路面坑洞', 49, bold=True)
    label(d, (34, 91), '30 个三维模型  /  10 种形态，每种 3 个面积档', 26)
    label(d, (34, 135), '俯视图使用同一比例，每格 1 m × 1 m。数字为凹深至少 1 cm 的水平投影面积。', 20, SOFT)
    label(d, (1256, 36), '可见表面凹深', 22, bold=True)
    ramp = np.linspace(1, 60, 560)
    page.paste(Image.fromarray(depth_color(np.tile(ramp, (15, 1)))), (1258, 74))
    for v in (1, 15, 30, 45, 60):
        x = 1258 + round((v - 1) * 559 / 59)
        label(d, (x, 100), f'{v} cm', 19, SOFT, anchor='mt')
    for n, (family, entries) in enumerate(groups.items()):
        left, top = 32 + n % 2 * block, header + n // 2 * row_height
        label(d, (left, top + 1), NAMES[family], 26, bold=True)
        for k, row in enumerate(entries):
            x = left + k * pitch
            page.paste(footprint(library, row, size), (x, top + 36))
            label(d, (x + 3, top + 335), row['id'], 21, SOFT)
            label(d, (x + size[0] - 5, top + 335), f"{row['depression_area_m2']:.2f} m²", 22,
                  bold=True, anchor='rt')
    page.save(destination, quality=94, subsampling=0)


def movie(library, groups, destination):
    import imageio.v2 as imageio
    from atlas_gaussians import load, render

    fps, frames = 20, 80
    width, height = 480, 410
    with imageio.get_writer(str(destination), fps=fps, codec='libx264', quality=8,
                            macro_block_size=2, output_params=['-threads', '3', '-movflags', '+faststart']) as writer:
        for family, entries in groups.items():
            template = Image.new('RGB', (width * 3, 550), BG)
            d = ImageDraw.Draw(template)
            label(d, (22, 14), NAMES[family], 36, bold=True)
            label(d, (width * 3 - 20, 25), '三件模型使用相同机位与米制比例', 19, SOFT, anchor='rt')
            states, reach = [], 0
            for i, row in enumerate(entries):
                states.append(load(dict(row, ply=str((library / row['ply']).resolve()))))
                with np.load(library / row['surface']) as field:
                    mask = field['defect_mask']
                    iy, ix = np.nonzero(mask)
                    pitch = float(field['width']) / (mask.shape[0] - 1)
                    r = np.hypot(ix - (mask.shape[1] - 1) / 2, iy - (mask.shape[0] - 1) / 2)
                    reach = max(reach, r.max() * pitch)
                left = i * width + 20
                label(d, (left, 490), f"{row['id']}    {row['depression_area_m2']:.2f} m²", 24, bold=True)
                label(d, (left, 524), f"{row['width_m']:.2f} × {row['length_m']:.2f} m", 17, SOFT)
            phi = math.radians(48)
            slope = math.tan(math.radians(15))
            radius = 1.12 * reach * max(math.sin(phi) + math.cos(phi) / slope, 1 / (slope * width / height) + .15)
            for frame in range(frames):
                image = template.copy()
                camera = dict(theta=2 * np.pi * frame / frames - .35, phi=phi, target=[0, 0, 0],
                              radius=radius, fov=30.)
                for i, state in enumerate(states):
                    rgb, _, _ = render(state, camera, width, height)
                    image.paste(Image.fromarray(rgb), (i * width, 64))
                writer.append_data(np.asarray(image))
            print('Rendered', family, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--library', type=Path, default=ROOT / 'outputs/large-potholes')
    cli.add_argument('--video', action='store_true')
    args = cli.parse_args()
    library = args.library.resolve()
    rows = json.loads((library / 'asset_manifest.json').read_text())
    if {r['id'] for r in rows} != {f'P{i}' for i in range(160, 190)} or len(rows) != 30:
        cli.error('Review requires the full batch of thirty models')
    groups, report = defaultdict(list), {}
    for row in sorted(rows, key=lambda r: r['id']):
        model = library / row['glb']
        area = measure(model, row['depression_threshold_m'])
        field_area = row['depression_area_m2']
        if abs(area / field_area - 1) > .012 or area < 3 or sha256(model) != row['glb_sha256']:
            raise ValueError('Mesh area or GLB digest mismatch: ' + row['id'])
        report[row['id']] = dict(mesh_area_below_1cm_m2=round(area, 5), grid_area_below_1cm_m2=field_area,
                                 glb_sha256=row['glb_sha256'], ply_sha256=row['ply_sha256'])
        groups[row['family']].append(row)
    if list(groups) != list(NAMES) or any(len(rs) != 3 for rs in groups.values()):
        raise ValueError('Unexpected family sequence or size')
    with TemporaryDirectory(dir=library) as temp:
        stage = Path(temp)
        overview(library, groups, stage / 'overview.jpg')
        (stage / 'areas.json').write_text(json.dumps(report, indent=2) + '\n')
        if args.video:
            movie(library, groups, stage / 'orbits.mp4')
        for path in stage.iterdir():
            path.replace(library / path.name)
    print('Verified', len(report), 'mesh footprints')


if __name__ == '__main__':
    main()
