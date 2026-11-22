"""Check native PLY previews and compare captured WebGL colour and placement with CUDA and GLB rays."""

import argparse
import hashlib
import sys
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage
import torch

from atlas_gaussians import ATLAS, ROOT, catalog, digest, load, projection, render

sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, save_json

DEPTH_COLORS = np.array([[55, 127, 113], [133, 171, 120], [238, 181, 91], [189, 82, 58]], dtype='u1')


def read_surface(path):
    scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    groups = []
    for name, vertices, triangles in glb_triangles(path, names=True):
        geometry = scene.add_triangles(o3d.core.Tensor(vertices.copy()), o3d.core.Tensor(triangles.copy()))
        if geometry != len(groups):
            raise ValueError('Unexpected geometry ID')
        prefixes = [
            ('Road', 'apron'),
            ('Fracture', 'pit_or_crack'),
            ('Cast ring', 'rim'),
            ('Chipped concrete collar', 'collar'),
            ('Lid', 'cover'),
            ('Hinge', 'hinge'),
            ('Inspection shaft', 'wall'),
            ('Shaft bottom', 'floor'),
            ('Water', 'water'),
            ('Detached pavement top', 'slab_top'),
            ('Detached pavement edges', 'slab_edge'),
            ('Detached pavement underside', 'slab_underside'),
            ('Exposed slab foundation', 'slab_base'),
            ('Subgrade bearing', 'slab_support'),
            ('Slab socket wall', 'slab_socket'),
        ]
        groups.append(next((group for prefix, group in prefixes if name.startswith(prefix)), name))
    return scene, groups


def check_manifest(rows, folder):
    manifest = json.loads((folder / 'views.json').read_text())
    if rows.keys() != manifest.keys():
        raise ValueError('Gaussian previews do not cover the current atlas')
    source = hashlib.sha256()
    images = hashlib.sha256()
    counts = {}
    points = 0
    minimum = 1.0
    for key, row in sorted(rows.items()):
        view = manifest[key]
        p = folder / (key + '.webp')
        if (
            digest(ATLAS / row['ply']) != view['ply_sha256']
            or digest(ATLAS / row['glb']) != view['glb_sha256']
            or digest(p) != view['image_sha256']
        ):
            raise ValueError('Changed source PLY or derived image for ' + key)
        with Image.open(p) as im:
            if im.size != (view['width'], view['height']):
                raise ValueError(key + ' preview dimensions')
            im.verify()
        if view['count'] != row['gaussians'] or view['foreground_pixels'] <= 0:
            raise ValueError(key + ' empty view')
        source.update((key + ':' + view['ply_sha256'] + '\n').encode())
        images.update((key + ':' + view['image_sha256'] + '\n').encode())
        counts[row['method']] = counts.get(row['method'], 0) + 1
        points += view['count']
        minimum = min(minimum, view['foreground_pixels'] / (view['width'] * view['height']))
    return dict(
        assets=len(rows),
        gaussians=points,
        methods=counts,
        ply_inventory_sha256=source.hexdigest(),
        preview_inventory_sha256=images.hexdigest(),
        minimum_fraction_alpha_above_half=minimum,
    )


def cast_glb(scene, camera, width, height):
    view, K = projection(camera, width, height)
    # Mesh vertices and PLY centres have independently been placed in the local Z-up frame.
    pose = np.linalg.inv(view)
    rr, cc = np.mgrid[:height, :width]
    camera_rays = np.stack(
        [(cc + 0.5 - K[0, 2]) / K[0, 0], (rr + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(rr)], -1
    )
    directions = camera_rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    rays = np.concatenate([np.broadcast_to(origin, directions.shape), directions], -1).astype('f4')
    hits = scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)
    z = hits['t_hit'].numpy()
    front = np.sum(hits['primitive_normals'].numpy() * directions, axis=2) < -1e-5
    return z, front, hits['geometry_ids'].numpy()


def compare(surface, gaussians, info, gaussian_file, mesh_file, evidence):
    w, h = info['width'], info['height']
    rgb, alpha, depth = render(gaussians, info['camera'], w, h)
    web = np.asarray(Image.open(gaussian_file).convert('RGB'))
    if web.shape != rgb.shape:
        raise ValueError('Wrong WebGL viewport')
    opaque = alpha > 0.85
    error = np.abs(rgb.astype(float) - web).mean(2)[opaque]
    if not error.size:
        raise ValueError('No opaque Gaussian support')
    scene, groups = surface
    z, front, geometry = cast_glb(scene, info['camera'], w, h)
    with Image.open(mesh_file) as image:
        coverage = np.asarray(image.convert('RGBA').getchannel('A').resize((w, h), Image.Resampling.BILINEAR))
    fully_covered = coverage > 245
    face = np.isfinite(z) & front
    # Antialiasing and raster boundary rounding can differ at an exact pixel edge.
    core = ndimage.binary_erosion(face, iterations=2)
    reach = ndimage.binary_dilation(np.isfinite(z), iterations=2)
    recall = float(fully_covered[core].mean()) if core.any() else None
    precision = float(reach[fully_covered].mean()) if fully_covered.any() else None
    # At silhouettes/translucent pixels the expected splat depth represents multiple layers.
    paired = opaque & core & fully_covered
    absolute = 1000 * np.abs(depth[paired] - z[paired])
    values = dict(
        width=w,
        height=h,
        opaque_pixels=int(opaque.sum()),
        rgb_mae_0_255=float(error.mean()),
        rgb_error_p95_0_255=float(np.percentile(error, 95)),
        glb_front_interior_recall=recall,
        glb_ray_support_precision=precision,
        shared_opaque_front_pixels=int(paired.sum()),
        gaussian_expected_vs_glb_depth_mm_median=float(np.median(absolute)) if absolute.size else None,
        gaussian_expected_vs_glb_depth_mm_p95=float(np.percentile(absolute, 95)) if absolute.size else None,
    )
    values['blend'] = info['blend']
    values['camera'] = info['camera']
    hit_groups = np.array(groups)[geometry[paired]]
    values['depth_by_first_mesh_part'] = {}
    for group in np.unique(hit_groups):
        diff = absolute[hit_groups == group]
        values['depth_by_first_mesh_part'][group] = dict(
            pixels=len(diff), median_mm=float(np.median(diff)), p95_mm=float(np.percentile(diff, 95))
        )
    values['depth_abs_over_50mm_fraction'] = float(np.mean(absolute > 50))
    tinted = np.full((h, w, 3), [228, 230, 220], dtype='u1')
    tinted[paired] = DEPTH_COLORS[np.digitize(absolute, [2, 10, 50])]
    Image.fromarray(tinted).save(evidence / (info['key'] + '-' + info['mode'] + '.png'))
    if values['rgb_mae_0_255'] > 2 or values['rgb_error_p95_0_255'] > 5:
        raise ValueError(f'WebGL and CUDA colours disagree for {gaussian_file}: {values}')
    if min(recall or 0, precision or 0) < 0.98:
        raise ValueError(f'Projection mismatch for {mesh_file}: {values}')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', help='Audit these captured assets and verify the full preview inventory')
    parser.add_argument('--previews', type=Path, default=ROOT / 'runs/atlas-gaussians')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'runs/atlas-gaussian-checks/geometry')
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = catalog()
    report = check_manifest(rows, args.previews)
    by_key = {}
    for path in sorted(args.captures.glob('*.json')):
        observation = json.loads(path.read_text())
        if args.ids and observation['key'] not in args.ids:
            continue
        by_key.setdefault(observation['key'], []).append((path, observation))
    if args.ids and set(args.ids) != set(by_key):
        raise ValueError('Missing requested browser captures')
    if not by_key:
        raise ValueError('No browser captures')
    args.evidence.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, shots in by_key.items():
        row = rows[key]
        state = load(row)
        scene = read_surface(ATLAS / row['glb'])
        measures = {}
        for path, obs in shots:
            if obs.get('sources') is not None and obs['sources'] != row['gaussianSources']:
                raise ValueError('Browser capture uses a different GLB or PLY for ' + key)
            if obs['count'] != len(state['means']):
                raise ValueError('Browser point count mismatch for ' + key)
            measures[obs['mode']] = compare(
                scene,
                state,
                obs,
                path.with_name(path.stem + '-gaussian.png'),
                path.with_name(path.stem + '-mesh.png'),
                args.evidence,
            )
        sources = {kind: digest(ATLAS / row[kind]) for kind in ('ply', 'glb')}
        result[key] = dict(method=row['method'], gaussians=row['gaussians'], sources=sources, views=measures)
        print(
            key,
            [
                (n, round(q['rgb_mae_0_255'], 3), round(q['gaussian_expected_vs_glb_depth_mm_median'], 2))
                for n, q in measures.items()
            ],
            flush=True,
        )
    report['browser_vs_cuda_and_glb'] = result
    save_json(args.out, report)


if __name__ == '__main__':
    main()
