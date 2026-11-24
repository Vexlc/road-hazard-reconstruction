"""Validate independent hatch leaves, native reloads, road occlusion and clipped fisheye transfers."""

import argparse
from importlib import import_module
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream'))
from mesh_transfer import RoadMesh, fit_footprint, fit_road, footprint_size, pixel_rays, resolve_samples
from road_geometry import glb_triangles, read_json, save_json, sha256, Surface

physical = import_module('render-mixed-road')
matched = import_module('check-mixed-road')
topology = import_module('check-perforated-covers').topology
IDS = ('A228', 'A229', 'M229', 'M230', 'M231')
SCENES = ('a2d2-frontleft', 'kitti360-0000', 'kitti360-0003-right')


def geometry(folder):
    path = folder / 'asset.glb'
    meta = read_json(folder / 'metadata.json')
    mesh = list(glb_triangles(path, names=True))
    left_present = meta['id'] in ('A228', 'A229', 'M230')
    right_present = meta['id'] in ('A228', 'A229', 'M229')
    leaves = {}
    for prefix, present in [('Left', left_present), ('Right', right_present)]:
        leaf = [part for part in mesh if part[0].startswith(prefix + ' lid ')]
        if bool(leaf) != present:
            raise ValueError('Wrong physical leaf present in ' + meta['id'])
        if present:
            leaves[prefix] = topology(leaf)
    beam = topology([part for part in mesh if part[0] == 'Central bearing beam'])
    surface = Surface(path)
    xx, yy = np.meshgrid(np.linspace(-0.58, 0.58, 101), np.linspace(-0.25, 0.25, 29))
    origins = np.stack([xx, yy, np.ones_like(xx)], -1)
    distance, part, _ = surface.cast(origins, np.broadcast_to([0, 0, -1.], origins.shape))
    height = 1 - distance
    report = {}
    for side, mask, present in [('left', xx < -.08, left_present), ('right', xx > .08, right_present)]:
        expect_z = (.0535 if meta['id'] == 'A229' and side == 'right' else 0) if present else -.84
        if not np.all(part[mask] == (9 if present else 7)) or abs(height[mask] - expect_z).max() > .003:
            raise ValueError('Vertical rays disagree with leaf/shaft height for ' + folder.name)
        report[side] = {'rays': int(mask.sum()), 'median_z_m': float(np.median(height[mask]))}
    dist, code, _ = surface.cast(np.array([[meta['beam_center_x_m'], 0., 1.]]), np.array([[0., 0., -1.]]))
    if code[0] != 20 or not np.isclose(1 - dist[0], -.052, atol=1e-5):
        raise ValueError('Centre seam did not expose the independently supported beam')
    return dict(leaf_solids=leaves, beam_solid=beam, surface_probes=report)


def grazing_control(mesh):
    # A die-cut sensor with an equidistant lens: the right sensor edge crosses the target
    # behind the optical plane. Compare the optimized crop with uncropped brute-force casts.
    width, height = 256, 180
    k = [[79., 0., 127.5], [0., 79., 89.5], [0., 0., 1.]]
    sensor = dict(size=[width, height], model='Fisheye', K=k, D=[0., 0., 0., 0.])
    model = physical.view_model('test', {'source_calibration': {'test': sensor}})
    window = physical.RayWindow(model)
    pose = np.array([[1,0,0,0], [0,0,1,0], [0,-1,0,1.35], [0,0,0,1]], dtype=float)
    anchor = dict(center_world=[3.8,-.8,0.], basis_columns=np.eye(3).tolist())
    host = np.full((height, width, 3), 112, 'u1')
    measured = np.full((height, width), np.inf, 'f4')
    samples = []
    for dx, dy in physical.SUBPIXELS:
        rays, valid = model[2](0, 0, width, height, dx, dy)
        samples.append(mesh.render(host, pose, rays, anchor, measured, ~valid))
    full, parts, ranges = resolve_samples(samples)
    with TemporaryDirectory() as tmp:
        photo = Path(tmp) / 'host.png'
        Image.fromarray(host).save(photo)
        clipped = physical.frame_crop(mesh, anchor, model, pose, photo, np.empty((0, 3)), window,
                                      dict(grid=[width, height], edge=[[0,height],[width-1,height]]))
    if clipped is None or 'right' not in clipped['clipped_sides'] or not parts[:, -1].any():
        raise ValueError('Lost a partially visible target on the sensor edge')
    x, y = clipped['xy']
    h, w = clipped['part'].shape
    ids = np.zeros_like(parts)
    ids[y:y+h,x:x+w] = clipped['part']
    if (not np.array_equal(ids, parts) or not np.array_equal(clipped['after'], full[y:y+h,x:x+w]) or
            not np.allclose(clipped['range'], ranges[y:y+h,x:x+w], atol=0, rtol=0, equal_nan=True)):
        raise ValueError('Tile-based rendering differs from the full-sensor reference')
    rays, valid = model[2](0, 0, width, height)
    rear = (parts > 0) & (rays[..., 2] < 0)
    uv = model[1](rays[rear])
    yy, xx = np.where(rear)
    if len(xx) < 20 or np.max(abs(uv - np.c_[xx,yy])) > .001:
        raise ValueError('Folded a rear-hemisphere fisheye ray')
    # Distances to labelled objects can be within the road tolerance or even below the entry plane.
    baseline, label, hit = mesh.render(host, pose, rays, anchor, measured, ~valid)
    close = np.where(label > 0, hit - .015, np.inf)
    hidden, hide_id, _ = mesh.render(host, pose, rays, anchor, measured, ~valid, occluder_range=close)
    if hide_id.any() or not np.array_equal(hidden, host):
        raise ValueError('Overwrote labelled foreground 15 mm closer than the asset')
    behind = np.where(label > 0, hit + .015, np.inf)
    retained, keep_id, _ = mesh.render(host, pose, rays, anchor, measured, ~valid, occluder_range=behind)
    if not np.array_equal(retained, baseline) or not np.array_equal(keep_id, label):
        raise ValueError('A farther object removed the replacement surface')
    _, valid_domain = pixel_rays(['OPENCV_FISHEYE',width,height,62.,62.,128.,90.,-.19,0,0,0], return_valid=True)
    if valid_domain[0,0] or not valid_domain[90,128]:
        raise ValueError('Folding fisheye polynomial did not mask the invalid border')
    invalid_corner = np.zeros_like(valid_domain)
    invalid_corner[:8, :8] = True
    missing_ray = rays.copy()
    missing_ray[invalid_corner] = np.nan
    masked, _, _ = mesh.render(host, pose, missing_ray, anchor, measured, invalid_corner)
    if not np.array_equal(masked, baseline):
        raise ValueError('Protected invalid lens bearings changed visible geometry')
    try:
        mesh.render(host, pose, missing_ray, anchor, measured, np.zeros_like(valid_domain))
    except ValueError:
        pass
    else:
        raise ValueError('Accepted undefined bearings on an unmasked sensor')
    return dict(hit_pixels=int((parts > 0).sum()), rear_hemisphere_pixels=int(rear.sum()),
                visible_on_right_edge=int((parts[:, -1] > 0).sum()), masked_uncalibrated_pixels=int((~valid_domain).sum()),
                exact_full_sensor_rgb_part_range=True, masked_nan_bearings=int(invalid_corner.sum()), foreground_delta_m=0.015)


def curb_controls(bundle):
    scene = 'kitti360-0003-right'
    data, cameras = physical.load_rig(ROOT/'work/streetcrafter-rig'/scene)
    scans = physical.scans(ROOT/'work/data'/scene, data, cameras)
    group = 24
    pose = np.linalg.inv(cameras[f'{group:03}00.png'])
    points = np.concatenate(scans[group-7:group+8:2])
    footprint = np.max([footprint_size(p.parent) for p in bundle.glob('assets/*/*/surface.npz')], axis=0)
    result = {}
    for offset in (.85, 1.55, 1.85):
        seed = pose[:3, 3] + offset*pose[:3, 0] - [0,0,1.6]
        anchor = fit_road(points, seed, pose[:3, 2], up_world=[0,0,1], radius_m=2.1)
        try:
            checked = fit_footprint(anchor, points, footprint)
        except ValueError as err:
            if offset == .85:
                raise
            result[str(offset)] = dict(accepted=False, reason=str(err))
        else:
            if offset != .85:
                raise ValueError(f'Accepted the surveyed curb/gutter crossing at {offset} m')
            result[str(offset)] = dict(accepted=True, **checked['footprint_support'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'runs/twin-hatch/twin/series')
    p.add_argument('--views', type=Path, default=ROOT/'outputs/twin-hatch')
    p.add_argument('--out', type=Path, default=ROOT/'runs/twin-hatch/validation.json')
    p.add_argument('--second-bundle', type=Path, help='Compare all five files per asset with a second clean build')
    args = p.parse_args()
    checks = read_json(args.bundle / 'checks.json')
    if args.second_bundle and checks != read_json(args.second_bundle / 'checks.json'):
        raise ValueError('Independent builds differ')
    data = dict(assets={}, hosts={}, independent_build_files=0)
    for key in IDS:
        (folder,) = args.bundle.glob(f'assets/*/{key}')
        hashes = checks[key]['sha256']
        for name, digest in hashes.items():
            if sha256(folder / name) != digest:
                raise ValueError(f'Changed {key}/{name}')
            if args.second_bundle:
                path = args.second_bundle / folder.relative_to(args.bundle) / name
                if sha256(path) != digest:
                    raise ValueError(f'Second build differs in {key}/{name}')
                data['independent_build_files'] += 1
        hosts = [s for s in args.bundle.glob('scenes/*/scene.json') if read_json(s)['assets'] == [key]]
        (scene_path,) = hosts
        info = read_json(scene_path)
        pl, = info['placements']
        ck = 'trained_model/iteration_7000.pth'
        if (pl['asset_ply_sha256'] != hashes['asset.ply'] or not info['actors_sky_unchanged'] or
                not info['checkpoint_reloaded'] or pl['reload_max_pixel_error'] != 0 or
                pl['size_multiplier'] != 1 or sha256(scene_path.parent / ck) != read_json(scene_path.parent/'complete.json')['files'][ck]):
            raise ValueError('Native checkpoint lost asset identity or scene state in ' + key)
        data['assets'][key] = dict(**geometry(folder), **hashes, site=info['id'], reload_pixel_error=0,
                                   depth_check=pl.get('cavity_depth_check'))
    mesh = RoadMesh(args.bundle / 'assets/missing/M230')
    data['grazing_sensor_control'] = grazing_control(mesh)
    data['kitti_0003_footprint_candidates'] = curb_controls(args.bundle)
    for scene in SCENES:
        base = args.views / scene
        data['hosts'][scene] = dict(**matched.check_frames(base, data['assets']),
                                    packets=matched.packet_controls(base, mesh, 'M230'))
        data['hosts'][scene]['M230_sensor_edge_frames'] = {
            sensor:r['image_edge_frames'] for sensor,r in read_json(base/'index.json')['M230']['views'].items()}
    save_json(args.out, data)
    print('Verified five double hatches, native checkpoints, three drives and the full-sensor grazing control.')


if __name__ == '__main__':
    main()
