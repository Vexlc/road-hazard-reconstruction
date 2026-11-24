"""Build the P160–P189 large potholes. Run with work/envs/assets/bin/python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "outputs/road_hazard_expansion/reproduction"
sys.path.insert(0, str(LEGACY))
import make_variants as mesh
from variant_catalog import PAVEMENTS

# Nominal planform aspect (length/width) and independent design depth range in metres.
FAMILIES = [
    ("bowled", 1.22, .15, .32), ("wheel_rut", 2.9, .10, .23),
    ("joined", .74, .16, .33), ("crescent", .76, .11, .23),
    ("cluster", 1.0, .13, .29), ("spall", 1.38, .055, .105),
    ("patch_break", 1.55, .10, .22), ("deep_break", .87, .24, .48),
    ("slab_corner", 1.12, .11, .26), ("waterlogged", 1.35, .15, .35),
]
SOURCE = ROOT / "work/reconstruction-inputs/pbr"
if not SOURCE.is_dir():
    SOURCE = ROOT / "work/road_hazard_sources"
GRID = 321
THRESHOLD = .01


def texture_sample(name, channel, extent, size, offset):
    # Sample the seamless scan at a fixed ground scale across all pit sizes.
    image = mesh.read_map(SOURCE, name, channel, 1024)
    axis = np.linspace(0, 2 * extent / 2.15, size, dtype='f4') * len(image)
    u, v = np.meshgrid(axis + offset[0], axis + offset[1])
    if image.ndim == 2:
        return map_coordinates(image, [v, u], order=1, mode='grid-wrap')
    return np.stack([map_coordinates(layer, [v, u], order=1, mode='grid-wrap')
                     for layer in image.transpose(2, 0, 1)], axis=-1)


def roughness(rng, sigma):
    band = gaussian_filter(rng.standard_normal((GRID, GRID)).astype('f4'), sigma)
    return np.tanh(band / max(float(band.std()), 1e-5))


def shape(config, w, l):
    rng = np.random.default_rng(config['seed'])
    extent = np.hypot(w, l) * .61 + .25
    x, y = mesh.axes(extent, GRID)
    warp = .027 * min(w, l)
    def warp_axis():
        return warp * (roughness(rng, 7) + .35 * roughness(rng, 1.8))
    wx, wy = x + warp_axis(), y + warp_axis()
    dist, pit, cracks = mesh.pit_fields(dict(config, width_m=w, length_m=l), wx, wy, rng)
    if config['family'] in {'patch_break', 'slab_corner'}:
        angle = rng.uniform(-.45, .45)
        u = (wx * np.cos(angle) + wy * np.sin(angle)) / (.5 * w)
        v = (wy * np.cos(angle) - wx * np.sin(angle)) / (.5 * l)
        dist = 1 - (np.abs(u) ** 5 + np.abs(v) ** 5) ** .2
        if config['family'] == 'slab_corner':
            dist = np.minimum(dist, .25 + .78 * u + .96 * v)
        pit = -config['depth_design_m'] * (.43 * mesh.smoothstep(dist / .06)
                                            + .57 * mesh.smoothstep(dist / .34))
        cracks *= mesh.smoothstep((.25 - np.abs(dist)) / .18)
        pit -= cracks * .005
    inside = mesh.smoothstep(dist / .12)
    pit *= 1 + .16 * roughness(rng, 14) * inside
    return extent, x, y, dist, pit, cracks


def broken_pieces(rng, x, y, interior, depth):
    """Faceted pieces seated on the bed, with tilted tops and short chipped sides."""
    z, cover = np.zeros_like(x), np.zeros_like(x)
    rows, cols = np.where(interior > .18)
    for i in rng.choice(len(rows), size=int(rng.integers(7, 20)), replace=False):
        cx, cy = x[rows[i], cols[i]], y[rows[i], cols[i]]
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(.08, .28)
        u = ((x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)) / radius
        v = ((y - cy) * np.cos(angle) - (x - cx) * np.sin(angle)) / (radius * rng.uniform(.45, .85))
        edge = 1 - np.maximum.reduce([np.abs(u), np.abs(v), .65 * u + .85 * v, -.88 * u - .57 * v])
        face = mesh.smoothstep(edge / .22)
        top = min(.085, .42 * depth, radius * .48) * np.clip(.85 + .28 * u - .15 * v, .35, 1.3)
        z = np.maximum(z, face * top)
        cover = np.maximum(cover, face)
    return z, cover


def build_one(index, output, cache):
    tier, family_slot = divmod(index, len(FAMILIES))
    family, aspect, shallow, deep = FAMILIES[family_slot]
    key = f'P{160 + index}'
    seed = 20261123 + index * 997
    rng = np.random.default_rng(seed)
    target = float(rng.uniform(*[(3.2, 4.9), (6., 8.8), (10.2, 14.5)][tier]))
    aspect *= rng.uniform(.87, 1.15)
    material = ('concrete' if family == 'slab_corner' else
                ('fine_asphalt', 'aged_asphalt', 'chip_seal')[(tier + family_slot) % 3])
    cfg = dict(id=key, category='pothole', family=family, material=material, seed=seed,
               recipe_version='large-potholes-v1', scale_kind='physical_design', size_multiplier=1.,
               target_depression_area_m2=round(target, 4), depth_design_m=round(rng.uniform(shallow, deep), 4),
               gaussian_sampling_density_m2=16000, gaussian_sampling_min=120000, gaussian_sampling_max=680000)
    # Estimate area from the generated shape before resizing horizontally. Depth stays in metres.
    width = np.sqrt(target / (.76 * aspect))
    extent, _, _, _, trial, _ = shape(cfg, width, width * aspect)
    estimate = (trial <= -THRESHOLD).sum() * (2 * extent / (GRID - 1)) ** 2
    width *= np.sqrt(target / estimate)
    extent, x, y, dist, pit, cracks = shape(cfg, width, width * aspect)
    step = 2 * extent / (GRID - 1)
    source, tint, amp = PAVEMENTS[material]
    offset = rng.integers(0, 1024, 2)
    micro = texture_sample(source, 'Displacement', extent, GRID, offset)
    aggregate = texture_sample('Concrete049', 'Displacement', extent, GRID, offset[::-1])
    inside = mesh.smoothstep(dist / .11)
    height = pit + ((1 - inside) * (micro - .5) * amp + inside * (aggregate - .5) * .021)
    chips = np.zeros_like(x)
    if family in {'joined', 'cluster', 'patch_break', 'deep_break', 'slab_corner', 'waterlogged'}:
        rubble, chips = broken_pieces(rng, x, y, dist, cfg['depth_design_m'])
        height += np.minimum(rubble, .55 * np.maximum(-height, 0))
        chips *= inside
    height *= mesh.smoothstep((extent - np.hypot(x, y)) / .17)
    height = height.astype('f4')
    field = mesh.local_field(x, y, height, extent)
    asset = mesh.Asset(key, cache)
    size = 1536
    def resize(a):
        return np.asarray(Image.fromarray(a.astype('f4')).resize((size, size), Image.Resampling.BILINEAR))
    pavement = texture_sample(source, 'Color', extent, size, offset) * np.asarray(tint, 'f4')
    gravel = texture_sample('Concrete049', 'Color', extent, size, offset[::-1])
    # Leave the matrix and exposed stone distinct from the surrounding pavement.
    gravel *= np.asarray([.82, .79, .73], 'f4')
    exposed_road = resize(.24 + .18 * roughness(rng, 10))[..., None]
    gravel = gravel * (1 - exposed_road) + pavement * exposed_road
    blend = resize(mesh.smoothstep((dist + .012) / .095))[..., None]
    tex = pavement * (1 - blend) + gravel * blend
    tex *= (1 - .37 * resize(cracks) - .19 * resize(mesh.smoothstep(-pit / cfg['depth_design_m'])))[..., None]
    caps = resize(chips)[..., None]
    tex = tex * (1 - caps) + (.76 * pavement + .24 * gravel) * caps
    asset.texture('Road and exposed aggregate', tex)
    xyz = np.stack([x, y, height], -1)
    uv = np.stack([x / (2 * extent) + .5, y / (2 * extent) + .5], -1)
    faces = mesh.faces_grid(GRID, GRID)
    asset.add('Road surface, lip and pit floor', xyz, faces, uv, 'Road and exposed aggregate', 1)
    if family == 'waterlogged':
        level = -.52 * cfg['depth_design_m']
        water_faces = faces[(height.ravel()[faces] < level).all(axis=1)]
        water = xyz.copy()
        water[..., 2] = level
        asset.texture('Muddy standing water', .065 * gravel + [.085, .074, .051], roughness=.20)
        asset.add('Water surface', water, water_faces, uv, 'Muddy standing water')
        wet = np.zeros_like(height, bool)
        wet.ravel()[np.unique(water_faces)] = True
        field.update(water_mask=wet, bed_height=height)
        field['height'] = np.where(wet, np.float32(level), height)
        cfg['water_level_m'] = level
    defect = field['height'] <= -THRESHOLD
    if np.any(field['alpha'][defect] < .999):
        raise ValueError(f'{key} reaches the edge of the asset')
    rows, cols = np.nonzero(defect)
    area = float(defect.sum() * step ** 2)
    if abs(area / target - 1) > .06:
        raise ValueError(f'{key}: depression area {area:.3f} vs target {target:.3f}')
    cfg.update(width_m=round(float(np.ptp(cols) + 1) * step, 4),
               length_m=round(float(np.ptp(rows) + 1) * step, 4),
               depression_threshold_m=THRESHOLD, depression_area_m2=round(area, 4),
               max_bed_depth_m=round(float(-height.min()), 4),
               max_surface_depth_m=round(float(-field['height'].min()), 4))
    field['defect_mask'] = defect
    dest = output / 'pothole' / key
    asset.save(dest, cfg, field)
    cfg['material_sources'] = sorted({source, 'Concrete049'})
    (dest / 'metadata.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(f"{key} {family:12} {area:5.2f} m²", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/large-potholes')
    parser.add_argument('--ids', nargs='+', choices=[f'P{i}' for i in range(160, 190)],
                        help='Build a subset in a new directory')
    args = parser.parse_args()
    dest = args.out.resolve()
    if dest.exists():
        parser.error(f'Choose a new output directory: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    ids = set(args.ids) if args.ids else {f'P{i}' for i in range(160, 190)}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
               ('work/asset_pipeline', 'work/road_hazard_pipeline')), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    env.setdefault('CUDA_HOME', '/usr/local/cuda')
    python = ROOT / 'work/envs/streetcrafter/bin/python'
    with TemporaryDirectory(prefix='.large-pits-', dir=dest.parent) as temp:
        stage, cache = Path(temp) / 'library', Path(temp) / 'meshes'
        for index in range(30):
            if f'P{160 + index}' in ids:
                build_one(index, stage, cache)
        subprocess.run([python, LEGACY / 'export_gaussians.py', '--out', stage, '--meshes', cache,
                        '--samples', cache], env=env, check=True)
        subprocess.run([python, LEGACY / 'verify_library.py', stage], env=env, check=True)
        stage.rename(dest)
    print(dest)


if __name__ == '__main__':
    main()
