"""Measure when a detached piece is visible from the calibrated front and rear cameras."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from camera_geometry import camera_hits, foreground, track
from road_geometry import Surface, glb_triangles, load_inventory, read_json, sha256


def visible_rect(corners, pose, K, size):
    # The box is convex. Clip all twelve edges at the camera near plane before projection.
    xyz = (corners - pose[:3, 3]) @ pose[:3, :3]
    selected = xyz[:, 2] > 0.05
    if not selected.any():
        return None
    endpoints = [xyz[selected]]
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if i < j and selected[i] != selected[j]:
                a, b = xyz[i], xyz[j]
                endpoints.append((a + (b - a) * ((0.05 - a[2]) / (b[2] - a[2])))[None])
    projected = np.vstack(endpoints) @ K.T
    uv = projected[:, :2] / projected[:, 2, None]
    lo = np.maximum(0, np.floor(uv.min(0)).astype(int) - 1)
    hi = np.minimum(size, np.ceil(uv.max(0)).astype(int) + 1)
    return [*lo, *hi] if (hi > lo).all() else None


def run(args):
    assets, scenes, _ = load_inventory(args.run)
    raw, processed = args.raw, args.processed
    results = {}
    for case, (path, scene) in scenes.items():
        if len(scene['placements']) != 1:
            continue
        (p,) = scene['placements']
        key = p['asset_id']
        folder, meta = assets[key]
        if meta.get('fragment_location') not in ('road', 'floor'):
            continue
        site = scene['site']
        glb = folder / 'asset.glb'
        survey = read_json(args.run / 'sensors/geometry/inventory.json')[key]
        if survey['mesh_sha256'] != sha256(glb):
            raise ValueError('Sensor asset identity')
        loose = {name for name, _, _ in glb_triangles(glb, names=True) if name.startswith('Lid fragment ')}
        fragment = Surface(glb, include_names=loose)
        body = Surface(glb)
        bounds = np.array(np.meshgrid(*zip(fragment.low, fragment.high), indexing='ij')).reshape(3, -1).T
        B = np.array(p['basis_columns'])
        center = np.array(p['center_world'])
        world = bounds @ B.T + center
        with np.load(folder / 'surface.npz') as field:
            models = [(p, body, {k: field[k] for k in ['width', 'length', 'defect_mask']})]
        dest = args.run / 'evaluation' / key
        cams = {}
        paired_dir = args.run / 'sensors/matched_camera' / case
        matched = {row['camera']: row for row in read_json(paired_dir / 'views.json') if row['available']}
        for camera in ['front_camera', 'back_camera']:
            times, poses, K, size = track(str(raw), site, camera)
            counts = []
            best = 0
            best_image = None
            for i, pose in enumerate(poses):
                rect = visible_rect(world, pose, K, size)
                row = [0, 0, 0]
                if rect:
                    l, t, r, b = rect
                    x, y = np.meshgrid(np.arange(l, r) + 0.5, np.arange(t, b) + 0.5)
                    dirs = np.dstack(
                        [(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(x)]
                    ).reshape(-1, 3)
                    orig = (pose[:3, 3] - center) @ B
                    dist, _, _ = fragment.cast(np.broadcast_to(orig, dirs.shape), dirs @ pose[:3, :3].T @ B)
                    single = dist.reshape(y.shape)
                    potential = np.isfinite(single) & (single > 0.05)
                    row[0] = int(potential.sum())
                    if row[0]:
                        depth, part, _, _ = camera_hits(pose, K, rect, models)
                        common = potential & np.isfinite(depth) & np.isin(part, [9, 10, 11])
                        direct = np.zeros(y.shape, bool)
                        direct[common] = abs(single[common] - depth[common]) < 0.0002
                        row[1] = int(direct.sum())
                        if row[1]:
                            masks = foreground(
                                str(raw), str(processed), args.run / 'sensors', site, camera, i, rect, depth
                            )
                            seen = direct & ~np.logical_or.reduce(masks)
                            row[2] = int(seen.sum())
                            if row[2] > best:
                                best = row[2]
                                with Image.open(raw / site / 'camera' / camera / f'{i:02d}.jpg') as im:
                                    photo = np.array(im.convert('RGB').crop(rect))
                                photo[seen] = np.uint8(0.35 * photo[seen] + 0.65 * np.array([255, 148, 70]))
                                best_image = Image.fromarray(photo)
                counts.append(row)
            array = np.asarray(counts)
            if not ((array[:, 0] >= array[:, 1]) & (array[:, 1] >= array[:, 2])).all():
                raise ValueError('Invalid visibility counts')
            if best_image:
                best_image.save(dest / (site + '-' + camera + '-visible-piece.png'))
            cams[camera] = dict(
                frames=len(times),
                frame_counts=counts,
                frames_with_unobstructed_pixel=int((array[:, 2] > 0).sum()),
                peak_visible_pixels=int(array[:, 2].max()),
                peak_frame=int(array[:, 2].argmax()) if best else None,
            )
            if camera in matched:
                record = matched[camera]
                frame = record['frame']
                l, t, r, b = record['crop_xyxy']
                if not np.allclose(record['camera_to_world'], poses[frame]) or not np.allclose(
                    record['K_full'], K
                ):
                    raise ValueError('Archived road camera disagrees with capture calibration')
                xx, yy = np.meshgrid(np.arange(l, r) + 0.5, np.arange(t, b) + 0.5)
                cam_rays = np.dstack(
                    [(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)]
                ).reshape(-1, 3)
                org = (poses[frame, :3, 3] - center) @ B
                hits, _, _ = fragment.cast(
                    np.broadcast_to(org, cam_rays.shape), cam_rays @ poses[frame, :3, :3].T @ B
                )
                with np.load(paired_dir / record['geometry']) as saved:
                    shared = (
                        (saved['protected'] == 0)
                        & np.isin(saved['part'], [9, 10, 11])
                        & np.isfinite(saved['depth_z_m'])
                    )
                    ds = hits.reshape(xx.shape)[shared] - saved['depth_z_m'][shared]
                verified = int((abs(ds) < 0.0002).sum())
                if verified != counts[frame][2]:
                    raise ValueError('Fragment visibility disagrees with the archived sensor depth image')
                cams[camera]['matched_geometry_check'] = dict(frame=frame, visible_pixels=verified)
        results[case] = dict(
            asset=key,
            location=meta['fragment_location'],
            scene_sha256=sha256(path),
            glb_sha256=sha256(glb),
            cameras=cams,
        )
        publish_plot(cams, key, dest, site)
    return dict(
        columns=['isolated_fragment', 'visible_on_complete_asset', 'after_foreground_protection'],
        cases=results,
    )


def publish_plot(cams, key, dest, site):
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.45), dpi=140, layout='constrained')
    fig.patch.set_facecolor('#f4f3ec')
    for ax, (name, rec) in zip(axes, cams.items()):
        data = np.array(rec['frame_counts'])
        x = np.arange(len(data))
        ax.plot(x, data[:, 0], color='#a6a28d', lw=1.7, label='碎片单独投影')
        ax.plot(x, data[:, 1], color='#9e4c38', lw=1.4, ls='--', label='完整几何中可见')
        ax.plot(x, data[:, 2], color='#13695b', lw=1.6, label='去除前景遮挡后')
        ax.set_yscale('symlog', linthresh=2)
        ax.set_ylim(bottom=-0.15)
        ax.grid(axis='y', alpha=0.15)
        ax.set_xlabel('原始帧号', fontproperties=font)
        ax.set_ylabel('碎片像素数', fontproperties=font)
        ax.set_title(
            f'{key} · {site} · ' + ('前视' if name == 'front_camera' else '后视'), fontproperties=font
        )
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(prop=font, frameon=False)
    fig.savefig(dest / (site + '-visibility.png'))
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    cfg = read_json(ROOT / 'gen2hole/config/local.json')
    p.add_argument('--raw', type=Path, default=Path(cfg['pandaset_raw']))
    p.add_argument('--processed', type=Path, default=Path(cfg['pandaset_processed']))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(json.dumps(run(args), indent=2, allow_nan=False) + '\n')
