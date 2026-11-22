"""Show hydrostatic states against their dry reconstructions, sections and calibrated roads."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFont

from road_camera_frames import camera_frames
from cover_observations import laser_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import Surface, read_json, save_json, sha256

plate = import_module('render-pit-survey')
Plate = plate.Plate
TEAL = '#007d83'
YELLOW = '#f3cb56'


def section(folder, dry, meta, out):
    with np.load(folder / 'surface.npz') as f:
        wet = f['water_mask'] > 0.5
        pit = f['defect_mask'] > 0.5
        width = float(f['width'])
    opts = meta['water']
    up = np.array(opts['up_local'])
    # Work in the same photo and bed coordinates at every fill for a given pit.
    nr, nc = pit.shape
    r, c = np.where(pit)
    left, right = max(c.min() - 12, 0), min(c.max() + 13, nc)
    top, bottom = max(r.min() - 12, 0), min(r.max() + 13, nr)
    with Image.open(dry) as photo:
        ortho = np.array(photo.convert('RGB').resize((nc, nr), Image.Resampling.LANCZOS))
    contour = cv2.findContours(wet.astype('u1'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(ortho, contour, -1, (45, 232, 218), 1, cv2.LINE_AA)
    cx = round(opts['seed_uv'][0] * (nc - 1))
    for y in range(top, bottom, 12):
        cv2.line(ortho, (cx, y), (cx, min(y + 6, bottom - 1)), (254, 213, 87), 1, cv2.LINE_AA)
    overlay = Image.fromarray(ortho[top:bottom, left:right])
    shoreline = out / 'shoreline.png'
    overlay.resize((overlay.width * 2, overlay.height * 2), Image.Resampling.LANCZOS).save(shoreline)
    x = (cx / (nc - 1) - 0.5) * width
    y = np.linspace((0.5 - (bottom - 1) / (nr - 1)) * width, (0.5 - top / (nr - 1)) * width, 900)
    origins = np.column_stack([np.full_like(y, x), y, np.full_like(y, 0.5)])
    rays = np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    solid = Surface(folder / 'asset.glb', exclude=(4,))
    patch = Surface(folder / 'asset.glb', include_names={'Water surface'})

    def heights(mesh):
        d, _, _ = mesh.cast(origins, rays)
        return x * up[0] + y * up[1] + (0.5 - d) * up[2]

    bed = heights(solid)
    top_water = heights(patch)
    valid = np.isfinite(top_water)
    if not valid.any() or not np.isfinite(bed).all():
        raise ValueError('Section misses the water or the bed')
    page = Plate(1120, 505)
    page.text(20, 14, meta['id'] + '   ' + meta['label'], 29, True)
    page.text(20, 63, '实拍正射纹理与岸线', 19, True)
    page.text(478, 63, '沿黄线切开网格 · 从图下端向上端', 19, True)
    page.picture(shoreline, 20, 100, 426, 342)
    l, t, rr, bb = 544, 116, 1088, 380
    # Keep the same range for all states of a bed to support direct comparisons.
    lo, hi = (-160.0, 90.0) if meta['depth_p95_m'] > 0.05 else (-45.0, 35.0)

    def py(z):
        return bb - (z * 1000 - lo) / (hi - lo) * (bb - t)

    px = np.linspace(l, rr, len(y))
    g = page.draw
    font = ImageFont.truetype(plate.FONT, 17, index=2)
    for v in np.arange(np.ceil(lo / 20) * 20, hi + 1, 20 if hi < 50 else 40):
        gy = py(v / 1000)
        g.line((l, gy, rr, gy), fill='#c8cec3', width=1)
        g.text((l - 9, gy), f'{v:.0f}', font=font, fill=plate.INK, anchor='rm')
    page.text(l - 58, t - 20, '毫米', 14)
    profile = list(zip(px, py(bed)))
    g.polygon([(l, bb), *profile, (rr, bb)], fill='#c0ae94')
    g.line(profile, fill='#574535', width=2)
    segments = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1)
    for ids in segments:
        if len(ids) < 2:
            continue
        top_line = list(zip(px[ids], py(top_water[ids])))
        base_line = list(zip(px[ids][::-1], py(bed[ids][::-1])))
        g.polygon(top_line + base_line, fill='#88c6c7')
        g.line(top_line, fill=TEAL, width=4)
    for frac in (0, 0.5, 1):
        v = (y[-1] - y[0]) * frac
        g.text((l + (rr - l) * frac, bb + 8), f'{v:.2f} m', font=font, fill=plate.INK, anchor='mt')
    page.text(
        476,
        426,
        f'最大水深 {opts["max_water_depth_m"] * 1000:.1f} mm　　水面 {opts["surface_area_m2"]:.3f} m²',
        19,
        True,
    )
    page.text(
        20, 464, '青线为计算岸线。右图按场景铅直方向标高，零点通过资产原点。浅蓝为积水，棕色为重建坑底。', 17
    )
    page.save(out / 'section.png')


def assets(run, runs, target):
    bundle = run / 'series'
    spec = read_json(bundle / 'calibration/survey.json')
    rows = {r['id']: r for r in read_json(bundle / 'assets/asset_manifest.json')}
    for s in spec['states']:
        key = s['id']
        dest = run / 'evaluation' / key
        dest.mkdir(parents=True, exist_ok=True)
        section(
            bundle / 'assets/pothole' / key, bundle / 'references' / (key + '_ortho.png'), rows[key], dest
        )
    # The published comparison uses the deep pit in all four main columns.
    deep = spec['beds']['P122']
    base = runs / deep['run'] / 'series'
    page = Plate(1552, 1274)
    page.text(28, 20, '同一个实拍深坑，在两种道路坡度上蓄水', 42, True)
    page.text(29, 85, '坑底由两张彩照重建。水量由输入指定，水面随目标道路坡度重新计算。', 24)
    xs = [28, 407, 786, 1165]
    photos = [
        base / 'references/P122.png',
        base / 'references/P122_second.png',
        base / 'views/P122_inspection.jpg',
        bundle / 'views/P130_inspection.jpg',
    ]
    for x, src, title in zip(xs, photos, ['第一张实拍', '第二张实拍', '干坑 GLB · P122', '积水 GLB · P130']):
        page.text(x, 152, title, 22, True)
        page.picture(src, x, 192, 358, 226)
    page.text(28, 450, '同一尺度的俯视岸线', 29, True)
    page.text(410, 458, '青线是水面与坑底的交线，黄色虚线标出下方剖面的位置。', 20)
    for x, key in zip(xs, ['P129', 'P130', 'P131', 'P132']):
        w = rows[key]['water']
        slope = 100 * np.linalg.norm(w['up_local'][:2]) / w['up_local'][2]
        page.text(x, 517, f'{key}   路段 {w["site"]} · 加入 {w["input_liters"]:g} L', 21, True)
        dest = run / 'evaluation' / key
        page.picture(dest / 'shoreline.png', x, 555, 358, 240)
        page.text(x, 808, f'坡度 {slope:.1f}%　留存 {w["retained_liters"]:.2f} L', 19)
        if key == 'P132':
            page.text(x, 837, f'溢出 {w["spilled_liters"]:.2f} L', 18, True)
    page.text(28, 898, '5 升水在较陡的坡面上更深，覆盖面积更小', 25, True)
    for x, key in [(28, 'P130'), (786, 'P131')]:
        section_png = run / 'evaluation' / key / 'section.png'
        with Image.open(section_png) as im:
            region = im.crop((465, 96, 1110, 457)).resize((495, 277), Image.Resampling.LANCZOS)
        page.image.paste(region, (x + 185, 955))
        page.text(x, 978, 'PandaSet ' + rows[key]['water']['site'], 19, True)
        page.text(x, 1020, '世界高程 / mm', 16)
    page.text(
        28,
        1230,
        '照片来源：PothRGBD。横向为截线距离；各图共用高度刻度。输入照片记录干坑，积水是基于该坑底的模拟状态。',
        18,
    )
    page.save(target)


def roads(run, runs, out):
    b = run / 'series'
    cfg = read_json(ROOT / 'gen2hole/config/local.json')
    streets = camera_frames(run, Path(cfg['pandaset_processed']))
    spec = read_json(b / 'calibration/survey.json')
    summary = {}
    cases = sorted((b / 'scenes').glob('*/scene.json'))
    for path in cases:
        scene = read_json(path)
        case = scene['id']
        (place,) = scene['placements']
        key = place['asset_id']
        folder = run / 'sensors/matched_camera' / case
        meta = read_json(b / 'assets/pothole' / key / 'metadata.json')
        # Check the delivered plane against the actual reloaded placement basis.
        normal = np.array(place['basis_columns']) @ np.array(meta['water']['up_local'])
        if not np.allclose(normal, [0, 0, 1], atol=1e-6) or not np.allclose(normal, place['water_up_world']):
            raise ValueError('Water plane tilted in world frame ' + case)
        bed = meta['dry_source_id']
        base = runs / spec['beds'][bed]['run']
        originals = [
            (p, pl)
            for p in (base / 'series/scenes').glob(scene['site'] + '_*/scene.json')
            for pl in read_json(p)['placements']
            if pl['asset_id'] == bed and pl['slot'] == place['slot']
        ]
        orig = next(
            p
            for p, pl in originals
            if pl['basis_columns'] == place['basis_columns'] and pl['center_world'] == place['center_world']
        )
        dry_folder = base / 'sensors/camera' / orig.parent.name
        dry_views = {v['camera']: v for v in read_json(dry_folder / 'views.json') if v['available']}
        views = {}
        for v in read_json(folder / 'views.json'):
            if not v['available']:
                continue
            with np.load(folder / v['geometry']) as block:
                mask = block['part'] == 4
                dry = dry_views[v['camera']]
                if any(
                    dry[k] != v[k]
                    for k in ('frame', 'crop_xyxy', 'camera_to_world', 'K_full', 'camera_timestamp')
                ) or (sha256(folder / v['before_stem']) != sha256(dry_folder / dry['before_stem'])):
                    raise ValueError('Dry and wet source camera geometry differs')
                with np.load(dry_folder / dry['geometry']) as original:
                    both = mask & np.isfinite(original['depth_z_m'])
                    delta = original['depth_z_m'][both] - block['depth_z_m'][both]
                    if len(delta) and delta.min() < -0.0001:
                        raise ValueError('Visible water ray is farther away than the dry bed')
                    depth_median = float(np.median(delta)) * 1000 if len(delta) else None
                if (
                    mask.any()
                    and np.linalg.norm(block['world_normal'][mask].astype(float) - [0, 0, 1], axis=1).max()
                    > 0.002
                ):
                    raise ValueError('Camera sees a tilted free-surface triangle')
            overlay = np.array(Image.open(folder / v['after_stem']).convert('RGB'))
            overlay[mask] = (0.53 * overlay[mask] + 0.47 * np.array([33, 214, 211])).astype('u1')
            cv2.drawContours(
                overlay,
                cv2.findContours(mask.astype('u1'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0],
                -1,
                (30, 240, 223),
                1,
            )
            Image.fromarray(overlay).save(run / 'evaluation' / key / (v['camera'] + '.png'))
            views[v['camera']] = dict(
                frame=v['frame'],
                crop_xyxy=v['crop_xyxy'],
                water_pixels=int(mask.sum()),
                dry_case=orig.parent.name,
                dry_photo=dry['after_stem'],
                paired_depth_pixels=len(delta),
                median_optical_advance_mm=depth_median,
            )
        with np.load(run / 'sensors/cases' / (case + '.npz')) as data:
            mask = (data['part'] == 4) & (data['action'] == 1)
            lidar = np.bincount(data['sensor'][mask], minlength=2).tolist()
        source_rows = laser_rows(base / 'sensors/cases' / (orig.parent.name + '.npz'))
        wet_rows = laser_rows(run / 'sensors/cases' / (case + '.npz'))
        shifts = []
        for beam, wet in wet_rows.items():
            if wet[3] != 4 or not np.isfinite(wet[2]):
                continue
            dry = source_rows.get(beam)
            if dry and dry[:2] != wet[:2]:
                raise ValueError('Different original laser sample')
            if dry and np.isfinite(dry[2]) and dry[3] > 1:
                shifts.append((dry[2] - wet[2]) * 1000)
        if shifts and min(shifts) < -0.1:
            raise ValueError('Water return behind dry geometry')
        summary[key] = dict(
            paired_dry_beams=len(shifts),
            median_beam_advance_mm=float(np.median(shifts)) if shifts else None,
            case=case,
            site=scene['site'],
            dry_source=meta['dry_source_id'],
            camera=views,
            water_up_world=normal.tolist(),
            water_beam_intersections=lidar,
        )
    for a, other in [('P129', 'P130'), ('P131', 'P132')]:
        for cam, view in summary[a]['camera'].items():
            twin = summary[other]['camera'][cam]
            if (view['frame'], view['crop_xyxy']) != (twin['frame'], twin['crop_xyxy']):
                raise ValueError('Unequal paired camera geometry')
    page = Plate(1500, 1335)
    page.text(28, 18, '从行车相机看积水坑', 42, True)
    page.text(28, 82, '上排保留前视相机完整视野，黄框对应右上角放大图。下排比较匹配机位的局部画面。', 22)
    for x, key in [(28, 'P130'), (760, 'P131')]:
        row = summary[key]
        case = row['case']
        page.text(
            x, 141, f'PandaSet {row["site"]} · 同坑注入 5 L · 前视第 {streets[case]["frame"]} 帧', 22, True
        )
        page.picture(run / 'evaluation/streets' / (case + '-after.jpg'), x, 182, 708, 398)
    cols = [28, 520, 1012]
    for x, t in zip(cols, ['109 · 深坑，1 L', '109 · 同一深坑，5 L', '109 · 相邻浅坑，75 mL']):
        page.text(x, 625, t, 21, True)
    for x, key in zip(cols, ['P129', 'P130', 'P133']):
        row = summary[key]
        folder = run / 'sensors/matched_camera' / row['case']
        v = next(
            a for a in read_json(folder / 'views.json') if a['camera'] == 'front_camera' and a['available']
        )
        box = np.array(v['xyxy_target_in_crop'])
        # Preserve the same pixel region for the two matched states.
        if key == 'P129':
            reference_box = box
        if key == 'P130':
            box = reference_box
        cx, cy = (box[:2] + box[2:]) / 2
        halfw = max(30, (box[2] - box[0]) * 0.85, (box[3] - box[1]) * 1.15)
        zoom = (int(cx - halfw), int(cy - halfw * 0.53), int(cx + halfw), int(cy + halfw * 0.53))
        for yy, path in [
            (666, folder / v['after_stem']),
            (939, run / 'evaluation' / key / 'front_camera.png'),
        ]:
            with Image.open(path) as im:
                tile = im.crop(zoom).resize((462, 245), Image.Resampling.LANCZOS)
            page.image.paste(tile, (x, yy))
        view = row['camera']['front_camera']
        page.text(x, 1204, f'第 {view["frame"]} 帧　水面覆盖 {view["water_pixels"]:,} 像素', 18, True)
    page.text(
        28, 1262, '中排为模拟迁入画面。下排在相同画面上标出相机射线最先击中的水面，青色表示可见水面。', 20
    )
    page.save(out)
    save_json(run / 'evaluation/water-observations.json', summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=ROOT / 'runs')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--roads', action='store_true')
    a = p.parse_args()
    if a.roads:
        roads(a.run, a.runs, a.out)
    else:
        assets(a.run, a.runs, a.out)
