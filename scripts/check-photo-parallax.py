"""Ray-test RGB-only pothole GLBs against archived depth and separately located photographs."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole'), str(ROOT / 'gen2hole/downstream/sensors')]
from fuse_pit import plane
from photo_parallax import camera_rays
from pit_captures import View, features, pinhole, polygons
from pit_texture_eval import read_texture, compare
from road_geometry import Surface, sha256 as file_hash

module = importlib.util.spec_from_file_location(
    'pit_views', Path(__file__).with_name('check-pit-views.py')
)
projection = importlib.util.module_from_spec(module)
module.loader.exec_module(projection)


def read(path):
    return json.loads(path.read_text())


def sensor_view(key, index, archives, K):
    record = index['rgbd_pothole_' + key]
    members = {}
    for role, archive in archives.items():
        raw = archive.read(record[role]['filename'])
        if sha256(raw).hexdigest() != record[role]['sha256']:
            raise ValueError('Validation input hash ' + key)
        members[role] = raw
    im = Image.open(BytesIO(members['rgb'])).convert('RGB')
    z = np.load(BytesIO(members['depth']), allow_pickle=False).astype('f4') * 0.001
    return View(key, np.array(im), z, polygons(members['mask'], im.size), K, record)


def stereo_view(key, directory, K):
    xyz_path = directory / 'ptcloud' / (key + '.mat')
    raw = xyz_path.read_bytes()
    optical = loadmat(BytesIO(raw))['xyzPoints'][:, :, 2].astype('f4') * 0.001
    optical = np.where(np.isfinite(optical) & (optical > 0.15) & (optical < 15), optical, 0)
    with Image.open(directory / 'rgb' / (key + '.png')) as im:
        rgb = np.array(im.convert('RGB'))
    with Image.open(directory / 'label' / (key + '.png')) as im:
        mask = np.array(im.convert('L')) > 0
    return View(key, rgb, optical, mask, K, dict(depth=dict(sha256=sha256(raw).hexdigest())))


def pose_on_road(anchor, test, road):
    # Features use only exterior color. The unit depths serve solely to unproject those pixels to rays.
    def color_features(v):
        proxy = View(v.key, v.rgb, np.ones(v.mask.shape, 'f4'), v.mask, v.K, {})
        uv, _, d = features(proxy)
        return uv, d

    p, a = color_features(anchor)
    q, b = color_features(test)
    matcher = cv2.BFMatcher()

    def pairs(x, y):
        return {
            m.queryIdx: m.trainIdx
            for m, n in matcher.knnMatch(x, y, k=2)
            if m.distance < 0.78 * n.distance
        }

    f = pairs(a, b)
    r = pairs(b, a)
    ij = [(i, j) for i, j in f.items() if r.get(j) == i]
    if len(ij) < 16:
        raise ValueError('Too few road features for withheld photo ' + test.key)
    i, j = np.array(ij).T
    ray = camera_rays(p[i], anchor.K)
    normal = np.array(road['normal_away_from_camera'])
    xyz = ray * (road['distance_m'] / (ray @ normal))[:, None]
    ok, rot, trans, ids = cv2.solvePnPRansac(
        xyz,
        q[j],
        test.K,
        None,
        iterationsCount=3000,
        reprojectionError=2.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or len(ids) < 16:
        raise ValueError('Unresolved withheld camera ' + test.key)
    take = ids.ravel()
    if np.min(np.ptp(q[j][take], axis=0)) < 40:
        raise ValueError('Poorly distributed withheld matches')
    rot, trans = cv2.solvePnPRefineLM(xyz[take], q[j][take], test.K, None, rot, trans)
    tf = np.eye(4)
    tf[:3, :3] = cv2.Rodrigues(rot)[0]
    tf[:3, 3] = trans.ravel()
    pred = cv2.projectPoints(xyz[take], rot, trans, test.K, None)[0].reshape(-1, 2)
    return np.linalg.inv(tf), dict(
        exterior_inliers=len(take),
        median_reprojection_px=float(np.median(np.linalg.norm(pred - q[j][take], axis=1))),
    )


def section_plot(view, pose, frame, rays, rendered, mask, gt_height, valid, dest):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    # Fix the line from the projected model footprint before inspecting the reference depths.
    row = int(np.median(np.nonzero(mask)[0]))
    xs = np.flatnonzero(mask[row])
    if not len(xs):
        row = int(np.argmax(mask.sum(1)))
        xs = np.flatnonzero(mask[row])
    pad = max(12, int(0.14 * (xs[-1] - xs[0] + 1)))
    cols = np.arange(max(0, xs[0] - pad), min(view.mask.shape[1], xs[-1] + pad + 1))
    z = rendered[row, cols]
    local_rays = rays[row, cols] @ pose[:3, :3].T
    eye = pose[:3, 3] - np.array(frame['local_origin_in_anchor_m'])
    up = np.array(frame['local_axes_in_anchor_camera'])[:, 2]
    road = local_rays * (-(eye @ up) / (local_rays @ up))[:, None]
    along = np.linalg.norm(road - road[:1], axis=1) * 100
    pred = np.full(len(z), np.nan)
    hits = np.isfinite(z)
    pred[hits] = (eye @ up + z[hits] * (local_rays[hits] @ up)) * 1000
    ref = np.where(valid[row, cols], gt_height[row, cols] * 1000, np.nan)
    im = Image.fromarray(view.rgb.copy())
    draw = ImageDraw.Draw(im)
    draw.line((int(cols[0]), row, int(cols[-1]), row), fill='#fdd66b', width=max(2, im.width // 600))
    yy, xx = np.nonzero(mask | view.mask)
    crop = (
        int(cols[0]),
        max(0, int(yy.min()) - pad),
        int(cols[-1] + 1),
        min(im.height, int(yy.max()) + pad + 1),
    )
    fig, (photo, chart) = plt.subplots(
        1, 2, figsize=(9.0, 3.05), dpi=130, gridspec_kw={'width_ratios': [1, 1.7]}, layout='constrained'
    )
    fig.patch.set_facecolor('#f6f5ef')
    photo.imshow(im.crop(crop))
    photo.set_axis_off()
    chart.axhline(0, color='#aaa99e', lw=0.9)
    chart.scatter(along, ref, s=8, color='#bb7548', label='传感器深度', alpha=0.8)
    chart.plot(along, pred, lw=1.7, color='#185f57', label='双照片模型')
    chart.set_xlabel('沿路面距离（厘米）', fontproperties=font)
    chart.set_ylabel('相对路面高度（毫米）', fontproperties=font)
    chart.legend(prop=font, frameon=False, ncol=2, loc='best', fontsize=8)
    chart.spines[['top', 'right']].set_visible(False)
    chart.grid(axis='y', alpha=0.16)
    chart.tick_params(labelsize=8)
    fig.savefig(dest / 'section.png', facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(
        image_row=row,
        image_x=[int(cols[0]), int(cols[-1])],
        length_cm=float(along[-1]),
        sensor_samples=int(np.isfinite(ref).sum()),
    )


def sensor_datum(view):
    h, w = view.mask.shape
    y, x = np.indices((h, w))
    rays = camera_rays(np.c_[x.ravel(), y.ravel()], view.K).reshape(h, w, 3)
    region = cv2.dilate(view.mask.astype('u1'), np.ones((81, 81), 'u1')) > 0
    inner = cv2.dilate(view.mask.astype('u1'), np.ones((17, 17), 'u1')) > 0
    true_ok = np.isfinite(view.depth) & (view.depth > 0.1) & (view.depth < 15)
    collar = region & ~inner & true_ok
    if collar.sum() < 120:
        raise ValueError('Not enough exterior sensor depth')
    xyz = rays * view.depth[:, :, None]
    origin, normal, rms = plane([xyz[collar][::2]])
    return rays, origin, normal, int(collar.sum()), rms, true_ok


def relief_metrics(view, pose, frame, rendered, model_mask, dest, datum):
    rays, origin, normal, collar_count, rms, true_ok = datum
    h, w = view.mask.shape
    xyz = rays * view.depth[:, :, None]
    gt_height = (xyz - origin) @ normal
    measured = view.mask & true_ok
    good = measured & model_mask & np.isfinite(rendered)
    if good.sum() < 100:
        raise ValueError('Too few shared GLB/depth pixels ' + view.key)
    surf = rays[good] * rendered[good, None]
    in_anchor = surf @ pose[:3, :3].T + pose[:3, 3]
    recovered = (in_anchor - np.array(frame['local_origin_in_anchor_m'])) @ np.array(
        frame['local_axes_in_anchor_camera']
    )[:, 2]
    err = abs(recovered - gt_height[good])
    data = dict(
        sensor_road_points=collar_count,
        sensor_road_rms_mm=rms * 1000,
        overlap_points=int(good.sum()),
        relief_abs_median_mm=float(np.median(err) * 1000),
        relief_abs_p95_mm=float(np.percentile(err, 95) * 1000),
        flat_relief_abs_median_mm=float(np.median(abs(gt_height[good])) * 1000),
        flat_relief_abs_p95_mm=float(np.percentile(abs(gt_height[good]), 95) * 1000),
        sensor_concavity_p95_mm=float(np.percentile(-gt_height[good], 95) * 1000),
        rendered_concavity_p95_mm=float(np.percentile(-recovered, 95) * 1000),
    )
    error_map = np.full((h, w), 0, 'f4')
    error_map[good] = err
    colored = cv2.applyColorMap(np.uint8(np.minimum(error_map / 0.035, 1) * 255), cv2.COLORMAP_INFERNO)[
        :, :, ::-1
    ]
    colored[~good] = [98, 106, 103]
    data['section'] = section_plot(
        view, pose, frame, rays, rendered, model_mask, gt_height, true_ok, dest
    )
    return data, colored


def paired_section(view, datum, candidates, row, dest):
    """Compare the same visible pixels in a common sensor camera and road datum."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    rays, origin, normal, _, _, depth_valid = datum
    first, second = candidates
    common = view.mask & depth_valid & first[2] & second[2]
    if common.sum() < 100:
        raise ValueError('Too few shared pixels across both reconstructions')
    rgb, depth = first[1][common], second[1][common]
    measured = view.depth[common]
    stats = dict(
        models=[first[0], second[0]],
        common_interior_pixels=int(common.sum()),
        median_abs_optical_error_mm={
            key: float(1000 * np.median(abs(z - measured)))
            for key, z in [(first[0], rgb), (second[0], depth)]
        },
        median_abs_model_difference_mm=float(1000 * np.median(abs(rgb - depth))),
    )
    both = view.mask | first[2] | second[2]
    yy, xx = np.nonzero(both)
    pad = max(12, int(0.13 * np.ptp(xx)))
    x0, x1 = max(0, int(xx.min()) - pad), min(view.mask.shape[1], int(xx.max()) + 1 + pad)
    y0, y1 = max(0, int(yy.min()) - pad), min(view.mask.shape[0], int(yy.max()) + 1 + pad)
    if not common[row].any():
        row = int(np.argmax(common.sum(axis=1)))
    cols = np.arange(x0, x1)
    ray = rays[row, cols]
    factor = ray @ normal
    road_points = ray * ((origin @ normal) / factor)[:, None]
    along = np.linalg.norm(road_points - road_points[0], axis=1) * 100
    stats.update(section_image_row=row, section_image_x=[x0, x1 - 1])

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, (photo, chart) = plt.subplots(
        1,
        2,
        figsize=(10.5, 3.65),
        dpi=145,
        gridspec_kw={'width_ratios': [1, 1.65]},
        layout='constrained',
    )
    fig.patch.set_facecolor('#f6f5ef')
    photo.imshow(view.rgb[y0:y1, x0:x1])
    colors = ['#ffdf69', '#53e9ca', '#e68ed9']
    for mask, color in zip([view.mask, first[2], second[2]], colors):
        photo.contour(mask[y0:y1, x0:x1], levels=[0.5], colors=[color], linewidths=1)
    photo.axhline(row - y0, color='white', linewidth=0.7)
    photo.set_title('留出帧 ' + view.key + ' · 白线处取样', fontproperties=font)
    photo.set_axis_off()
    ref = np.where(depth_valid[row, cols], view.depth[row, cols], np.nan)
    chart.scatter(
        along, (ref * factor - origin @ normal) * 1000, color='#9a752d', s=9, alpha=0.8, label='传感器'
    )
    for (key, rendered, _), color, method in zip(
        candidates, ['#147b6e', '#9c4087'], ['双照片', '多帧双目']
    ):
        z = np.where(np.isfinite(rendered[row, cols]), rendered[row, cols], np.nan)
        chart.plot(
            along, (z * factor - origin @ normal) * 1000, color=color, lw=1.7, label=key + ' ' + method
        )
    chart.axhline(0, color='#aaa99e', lw=0.8)
    chart.set_xlabel('沿路面距离（厘米）', fontproperties=font)
    chart.set_ylabel('相对同一实测路面的高度（毫米）', fontproperties=font)
    chart.grid(axis='y', alpha=0.15)
    chart.spines[['top', 'right']].set_visible(False)
    chart.legend(prop=font, loc='best', frameon=False)
    fig.savefig(dest / 'same-pixels.png')
    plt.close(fig)
    return stats


def evaluate(args):
    bundle = args.run / 'series'
    out = args.run / 'evaluation'
    out.mkdir(exist_ok=True)
    specs = read(bundle / 'calibration/survey.json')
    checks = read(bundle / 'checks.json')
    index = read(args.catalog)
    extra = read(args.check) if args.check else {}
    result = {}
    folder = args.source / 'rgbd'
    with ExitStack() as stack:
        archives = {}
        if any(s['source'] == 'PothRGBD' and s.get('mode') == 'rgb_parallax' for s in specs):
            archives = {
                role: stack.enter_context(ZipFile(folder / name))
                for role, name in [
                    ('rgb', 'images.zip'),
                    ('mask', 'labels.zip'),
                    ('depth', '-depths.zip'),
                ]
            }
        for spec in specs:
            if spec.get('mode') != 'rgb_parallax':
                continue
            key = spec['id']
            cal = read(bundle / 'calibration' / (key + '_capture.json'))
            root = out / key
            root.mkdir(exist_ok=True)
            mesh = bundle / 'assets/pothole' / key / 'asset.glb'
            digest = file_hash(mesh)
            if digest != checks[key]['sha256']['asset.glb']:
                raise ValueError(key + ' model identity')
            scene = Surface(mesh)
            comparison = None
            if args.compare_run and spec.get('related'):
                other = spec['related']
                ref_bundle = args.compare_run / 'series'
                ref = read(ref_bundle / 'calibration' / (other + '_capture.json'))
                other_mesh = ref_bundle / 'assets/pothole' / other / 'asset.glb'
                other_hash = file_hash(other_mesh)
                if (
                    other_hash != read(ref_bundle / 'checks.json')[other]['sha256']['asset.glb']
                    or ref['spec']['source'] != spec['source']
                    or ref['spec'].get('dataset') != spec.get('dataset')
                    or set(ref['holdout']) & set(ref['fit'])
                ):
                    raise ValueError('Incompatible comparison reconstruction ' + other)
                comparison = dict(key=other, scene=Surface(other_mesh), record=ref, digest=other_hash)
            frame = cal['geometry']
            poses = {k: np.array(p) for k, p in cal['poses_view_to_anchor'].items()}
            field = dict(np.load(bundle / 'calibration' / (key + '.npz'), allow_pickle=False))
            width = float(field['width'])
            texture = read_texture(mesh, width)
            K = pinhole(spec['fx_fy_cx_cy'])
            novel = extra.get(key, {}).get('unseen_rgb', [])
            if set(novel) & set(spec['fit']):
                raise ValueError('Repeated evaluation photograph')
            if spec['source'] == 'PothRGBD':
                views = {k: sensor_view(k, index, archives, K) for k in spec['fit'] + novel}
            elif spec['source'] == 'FanStereo':
                views = {
                    k: stereo_view(k, args.source / 'stereo' / spec['dataset'], K)
                    for k in spec['fit'] + novel
                }
            else:
                raise ValueError('This reference evaluation uses PothRGBD or FanStereo sensor captures')
            camera_check = {}
            for k in novel:
                poses[k], camera_check[k] = pose_on_road(
                    views[spec['fit'][0]], views[k], cal['reference_road_plane']
                )
            evidence = {}
            for i, (k, view) in enumerate(views.items()):
                dest = root / k
                dest.mkdir(exist_ok=True)
                z, mask = projection.camera_image(view, poses[k], scene, frame)
                audit = projection.summary(view, z, mask)
                datum = sensor_datum(view)
                values, colored = relief_metrics(view, poses[k], frame, z, mask, dest, datum)
                if comparison and k in novel and k in comparison['record']['holdout']:
                    ref = comparison['record']
                    files = ref['inputs'][k]['files']
                    sensor_file = files.get('xyz', files.get('depth'))
                    if view.sources['depth']['sha256'] != sensor_file['sha256'] or not np.allclose(
                        view.K, ref['inputs'][k]['K'], atol=0.001
                    ):
                        raise ValueError('Models must use the same held-out depth and pixel calibration')
                    other_depth, other_mask = projection.camera_image(
                        view,
                        np.array(ref['poses_view_to_anchor'][k]),
                        comparison['scene'],
                        ref['geometry'],
                    )
                    values['same_pixel_comparison'] = dict(
                        other_glb_sha256=comparison['digest'],
                        **paired_section(
                            view,
                            datum,
                            [(key, z, mask), (comparison['key'], other_depth, other_mask)],
                            values['section']['image_row'],
                            dest,
                        ),
                    )
                crop = projection.overlays(view, z, mask, dest)
                Image.fromarray(colored[crop]).save(dest / 'relief-error.png')
                material = compare(
                    view, poses[k], frame, width, texture, z, mask, crop, dest / 'material.jpg'
                )
                evidence[k] = dict(
                    role='new_view' if k in novel else ('texture_photo' if i == 0 else 'matching_photo'),
                    input_depth_sha256=view.sources['depth']['sha256'],
                    **audit,
                    **values,
                    material=material,
                )
            anchor = views[spec['fit'][0]]
            picture = Image.fromarray(anchor.rgb)
            draw = ImageDraw.Draw(picture)
            ends = [tuple(p) for p in spec['ruler']['pixels']]
            draw.line(ends, fill='#ffdf69', width=2)
            for x, y in ends:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill='#ffdf69', outline='#23392e')
            picture.save(root / 'scale.jpg', quality=95)
            support = field['rgb_supported']
            target = field['defect_mask']
            base = (
                Image.open(bundle / 'references' / (key + '_ortho.png'))
                .convert('RGB')
                .resize(target.shape[::-1])
            )
            tone = np.array(base) * 0.55
            for region, color in [
                (target & support, [51, 211, 154]),
                (target & ~support, [247, 170, 62]),
            ]:
                tone[region] = 0.40 * np.array(base)[region] + 0.6 * np.array(color)
            Image.fromarray(tone.astype('u1')).save(root / 'support.jpg', quality=96)
            result[key] = dict(
                label=spec['label'],
                glb_sha256=digest,
                source=spec['source'],
                fit=spec['fit'],
                unseen_rgb=novel,
                heldout_camera_fit=camera_check,
                geometry=frame,
                scale=spec['ruler'],
                frames=evidence,
            )
            print(
                key,
                [
                    (
                        k,
                        round(v['mask_iou'], 3),
                        round(v['depth_abs_median_mm'], 1),
                        round(v['relief_abs_median_mm'], 1),
                        round(v['flat_relief_abs_median_mm'], 1),
                    )
                    for k, v in evidence.items()
                ],
                flush=True,
            )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'work/reconstruction-inputs')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'catalog/capture-inputs.json')
    parser.add_argument('--check', type=Path)
    parser.add_argument(
        '--compare-run', type=Path, help='Compare linked depth models in shared held-out frames'
    )
    args = parser.parse_args()
    path = args.out or args.run / 'evaluation/measurements.json'
    report = evaluate(args)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
