"""Trace scanned pavement islands into the calibrated road camera and count closed crack outlines."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from cover_observations import compare_cases
from scanned_crack_regions import enclosed_regions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import Surface, read_json, save_json


def readback(mesh, meta, pose, shot, labels):
    l, t, r, b = shot['crop_xyxy']
    yy, xx = np.mgrid[t:b, l:r] + 0.5
    K = np.asarray(shot['K_full'])
    ray = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1)
    camera = np.asarray(shot['camera_to_world'])
    basis = np.asarray(pose['basis_columns'])
    origin = (camera[:3, 3] - pose['center_world']) @ basis
    direction = ray.reshape(-1, 3) @ camera[:3, :3].T @ basis
    distance, part, _ = mesh.cast(np.broadcast_to(origin, direction.shape), direction)
    fixed = np.isfinite(distance) & (part == 1)
    tags = np.zeros(len(distance), 'i4')
    pts = origin + direction[fixed] * distance[fixed, None]
    pitch = meta['source_pitch_m']
    half = meta['intervals'] / 2
    cc = np.rint(pts[:, 0] / pitch + half).astype(int).clip(0, labels.shape[1] - 1)
    rr = np.rint(half - pts[:, 1] / pitch).astype(int).clip(0, labels.shape[0] - 1)
    tags[fixed] = labels[rr, cc]
    return tags.reshape(xx.shape)


def audit_camera(folder, shot, tags, dest):
    with np.load(folder / shot['geometry']) as geom:
        cracks = geom['part'] == 2
        protected = geom['protected'] != 0
    real = np.array(Image.open(folder / shot['before_stem']).convert('RGB'))
    placed = np.array(Image.open(folder / shot['after_stem']).convert('RGB'))
    marked = placed.copy()
    marked[cracks] = (0.50 * marked[cracks] + 0.50 * np.array([245, 177, 63])).astype('u1')
    regions, num = ndi.label(~cracks)
    outside = set(
        np.unique(np.r_[regions[0], regions[-1], regions[:, 0], regions[:, -1], regions[protected], 0])
    )
    observed = np.bincount(tags[~protected])
    resolved = {}
    for i in range(1, num + 1):
        if i in outside:
            continue
        m = regions == i
        total = int(m.sum())
        if total < 3:
            continue
        membership = np.bincount(tags[m])
        if len(membership) < 2:
            continue
        index = 1 + int(membership[1:].argmax())
        matched = int(membership[index])
        if matched / total < 0.8 or matched / observed[index] < 0.5:
            continue
        resolved[str(index)] = dict(
            pixels=matched, region_pixels=total, projected_source_pixels=int(observed[index])
        )
        marked[m] = (0.55 * marked[m] + 0.45 * np.array([12, 211, 211])).astype('u1')
    Image.fromarray(marked).save(dest / (shot['camera'] + '.png'))
    change = np.abs(placed.astype('i2') - real).mean(axis=2)[cracks]
    return dict(
        frame=shot['frame'],
        crop_xyxy=shot['crop_xyxy'],
        visible_crack_pixels=int(cracks.sum()),
        projected_islands_at_least_3_pixels=int(np.count_nonzero(observed[1:] >= 3)),
        enclosed_islands=resolved,
        median_rgb_change_on_crack=float(np.median(change)) if len(change) else None,
    )


def check(run):
    bundle = run / 'series'
    cases = {}
    paired = {}
    for scene_path in sorted((bundle / 'scenes').glob('*/scene.json')):
        scene = read_json(scene_path)
        (pose,) = scene['placements']
        key = pose['asset_id']
        folder = bundle / 'assets/pothole' / key
        meta = read_json(folder / 'metadata.json')
        if meta.get('paired'):
            paired[key] = meta['paired']
        labels, areas = enclosed_regions(
            np.array(Image.open(bundle / 'calibration' / (key + '_outline.png'))) > 0, meta['source_pitch_m']
        )
        mesh = Surface(folder / 'asset.glb')
        case = scene['id']
        camera = run / 'sensors/camera' / case
        out = run / 'evaluation' / key / case
        out.mkdir(parents=True, exist_ok=True)
        cameras = {}
        for shot in read_json(camera / 'views.json'):
            if not shot['available']:
                continue
            tags = readback(mesh, meta, pose, shot, labels)
            cameras[shot['camera']] = audit_camera(camera, shot, tags, out)
        with np.load(run / 'sensors/cases' / (case + '.npz')) as beams:
            hits = (beams['action'] == 1) & (beams['part'] == 2)
            returns = np.bincount(beams['sensor'][hits], minlength=2).tolist()
        cases[case] = dict(
            asset=key,
            site=scene['site'],
            source_islands=len(areas),
            cameras=cameras,
            simulated_fissure_hits_by_sensor=returns,
        )
    by_site_asset = {(obs['site'], obs['asset']): case for case, obs in cases.items()}
    for case, obs in cases.items():
        base = by_site_asset.get((obs['site'], paired.get(obs['asset'])))
        if base is None:
            continue
        poses = [read_json(bundle / 'scenes' / name / 'scene.json')['placements'][0] for name in (base, case)]
        if any(
            poses[0][k] != poses[1][k]
            for k in ('center_world', 'basis_columns', 'size_multiplier', 'source_reference_frame')
        ):
            raise ValueError('Different road anchor for the two depth scales at ' + obs['site'])
        obs['scale_comparison'] = compare_cases(run, obs['site'], base, case, image_suffix='depth-scale')
    return cases


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    save_json(a.out, check(a.run))
