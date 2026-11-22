"""Publish source-linked sections, bare geometry, and calibrated road views of the pavement scans."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from road_camera_frames import camera_frames, read

Plate = import_module('render-pit-survey').Plate
GREEN = '#008574'
GOLD = '#c47d2d'
BLUE = '#3a4676'
INK = '#213c36'


def panel(run, key, detail):
    folder = run / 'evaluation' / key
    bundle = run / 'series'
    meta = read(bundle / 'assets/pothole' / key / 'metadata.json')
    with np.load(folder / 'comparison.npz') as payload:
        line = payload['section']
    with Image.open(bundle / meta['source_image']) as source:
        rgb = source.convert('RGB')
    draw = ImageDraw.Draw(rgb)
    width = max(3, rgb.width // 225)
    if key == 'P135':
        left, top = np.subtract(detail['origin_col_row'], meta['origin_col_row'])
        draw.rectangle(
            (left, top, left + detail['intervals'], top + detail['intervals']), outline='#6be8cb', width=width
        )
    start, end = (tuple(row[:2]) for row in (line[0], line[-1]))
    draw.line([start, end], fill='#ffd25f', width=width)
    for x, y in [start, end]:
        draw.ellipse([x - width, y - width, x + width, y + width], fill='#ffe9a0')
    rgb.thumbnail((820, 820), Image.Resampling.LANCZOS)
    rgb.save(folder / 'transect.jpg', quality=95)
    return meta, line


def plot(page, box, distance, curves):
    x, y, w, h = box
    values = np.concatenate([v for _, v, _ in curves])
    low, high = float(values.min()), float(values.max())
    ticks = np.linspace(np.floor(low), np.ceil(high), 4)
    low, high = ticks[0], ticks[-1]
    left, right = x + 64, x + w - 12
    top, bottom = y + 30, y + h - 37

    def vertical(value):
        return bottom - (bottom - top) * (value - low) / max(high - low, 1.0e-8)

    page.text(x, y, '相对路面高度（mm）', 15)
    for v in ticks:
        pos = vertical(v)
        page.draw.line((left, pos, right, pos), fill='#ccd5cb')
        page.text(x + 3, pos - 8, f'{v:.1f}', 15)
    px = left + (right - left) * distance / distance[-1]
    for _, v, color in curves:
        page.draw.line(list(zip(px, vertical(v))), fill=color, width=3)
    page.text(left, bottom + 8, '0', 15)
    page.text(right - 83, bottom + 8, f'{distance[-1]:.0f} mm', 15)
    advance = left
    for label, _, color in curves:
        page.draw.line((advance, bottom + 54, advance + 26, bottom + 54), fill=color, width=4)
        page.text(advance + 34, bottom + 45, label, 16)
        advance += max(142, 54 + 16 * len(label))


def section(run, key, meta, line, row):
    p = Plate(1350, 603)
    p.text(26, 18, f'{key}   {meta["label"]}', 36, True)
    p.text(26, 75, '黄线为源图上的截线。右图从线段起点读到终点，高度以同一片周边路面为零。', 21)
    p.picture(run / 'evaluation' / key / 'transect.jpg', 26, 123, 410, 410)
    p.text(467, 126, '同处的三条高度剖面', 24, True)
    distances = np.hypot(line[:, 0] - line[0, 0], line[:, 1] - line[0, 1]) * row['source_texel_m'] * 1000
    plot(
        p,
        (466, 177, 850, 290),
        distances,
        [
            ('拟合 V 形沟', line[:, 4] * 1000, GOLD),
            ('源位移', line[:, 2] * 1000, BLUE),
            ('导出网格', line[:, 3] * 1000, GREEN),
        ],
    )
    text = f'网格间距 {1000 * row["mesh_pitch_m"]:.2f} mm    截线凹深 {row["section_depth_below_shoulders_mm"]:.2f} mm'
    p.text(475, 505, text, 20, True)
    p.text(
        26,
        560,
        '源位移和网格使用同一竖向尺度。P137 采用法线估计尺度。近乎重叠的曲线可在图谱中另取截线核对。',
        18,
    )
    p.save(run / 'evaluation' / key / 'section.jpg')


def portfolio(run, rows, items, out):
    p = Plate(1648, 1328)
    p.text(32, 23, '道路裂缝的三维重建', 46, True)
    p.text(
        33,
        96,
        '两份道路扫描，四件可迁移资产。颜色来自实拍纹理，无纹理视图以低角度光照显示导出网格的起伏。',
        24,
    )
    order = ['P135', 'P136', 'P134', 'P137']
    for i, key in enumerate(order):
        x = 32 + 399 * i
        w = 386
        m, line = items[key]
        row = rows[key]
        p.text(
            x,
            169,
            key
            + '   '
            + {'P135': '网裂整片', 'P136': '交汇局部', 'P134': '沥青纵缝', 'P137': '纵缝深度复核'}[key],
            25,
            True,
        )
        p.text(
            x, 211, f'{m["metric_source_span_m"]:.3f} m 见方  ·  网格 {1000 * row["mesh_pitch_m"]:.2f} mm', 18
        )
        p.picture(run / 'evaluation' / key / 'transect.jpg', x, 248, w, 262)
        p.picture(run / 'series/views' / (key + '_inspection.jpg'), x, 536, w, 226)
        p.picture(run / 'evaluation' / key / 'clay.jpg', x, 788, w, 226)
    for y, label in [(245, '正射原图'), (533, '纹理网格'), (785, '裸网格')]:
        p.draw.rectangle((31, y, 150, y + 28), fill='#e5e9dd')
        p.text(38, y + 4, label, 17, True)
    p.text(32, 1048, '混凝土  /  同一局部加密', 26, True)
    pair = rows['P136']['matched_detail']
    p.text(32, 1100, '青框为 P136 的取景。两张图上的黄线穿过同一处断边。', 20)
    p.text(32, 1140, f'在相同的 {pair["detail_error"]["pixels"]:,} 个留出缝内像素上，对源位移的', 20)
    p.text(
        32,
        1180,
        f'绝对误差中位数从 {pair["parent_error"]["median_mm"]:.3f} 降至 {pair["detail_error"]["median_mm"]:.3f} mm。',
        21,
        True,
    )
    p.text(835, 1048, '沥青  /  同一纵缝的两种深度', 26, True)
    p.text(835, 1100, '两件资产逐像素共用颜色，P134 沿用作者材质的尺度。', 20)
    a, b = (rows[k] for k in ('P134', 'P137'))
    p.text(
        835,
        1140,
        f'P137 用法线估计竖向尺度，示例截线从 {a["section_depth_below_shoulders_mm"]:.2f} 变为 {b["section_depth_below_shoulders_mm"]:.2f} mm。',
        20,
    )
    p.text(
        835,
        1180,
        f'留出区域中，缝内法线夹角中位数从 {a["holdout_crack_normals"]["median_deg"]:.2f}° 降至 {b["holdout_crack_normals"]["median_deg"]:.2f}°。',
        20,
        True,
    )
    p.draw.line((32, 1247, 1613, 1247), fill='#b9c8bb', width=2)
    p.text(
        32,
        1271,
        '纹理与位移来源  Poly Haven · Cracked Concrete / Dimitrios Savva · Asphalt 02 / Rob Tuytel · CC0。两种沥青深度均缺少独立的现场量测。',
        17,
    )
    p.save(out)


def roads(run, rows, streets, out):
    vis = read(run / 'evaluation/visibility.json')
    p = Plate(1512, 1220)
    p.text(28, 23, '同一张网裂，放到两段自驾道路上', 43, True)
    p.text(
        28, 87, 'P135 的 1.75 m 混凝土网裂保持原尺寸。黄框放大行车全景中的资产，下面两排为标定相机裁切。', 22
    )
    for i, case in enumerate(['034_135', '109_135']):
        x = 28 + 741 * i
        w = 716
        obs = vis[case]
        full = streets[case]
        p.text(x, 143, f'PandaSet {obs["site"]}   ·   行车前视第 {full["frame"]} 帧', 25, True)
        p.picture(run / 'evaluation/streets' / (case + '-after.jpg'), x, 186, w, 395)
        for j, (cam, name) in enumerate([('front_camera', '前视'), ('back_camera', '后视')]):
            r = obs['cameras'][cam]
            y = 611 + j * 233
            p.text(
                x,
                y,
                f'{name}第 {r["frame"]} 帧   ·   {r["visible_crack_pixels"]:,} 个缝内像素   ·   {len(r["enclosed_islands"])} 个闭合板块',
                20,
                True,
            )
            src = run / 'sensors/camera' / case
            shot = next(s for s in read(src / 'views.json') if s['camera'] == cam)
            p.picture(src / shot['after_stem'], x, y + 39, 352, 178)
            p.picture(run / 'evaluation/P135' / case / (cam + '.png'), x + 364, y + 39, 352, 178)
    p.draw.line((28, 1089, 1480, 1089), fill='#b9c8bb', width=2)
    p.text(
        28,
        1110,
        '每对裁切左为迁入原色，右为几何核对。黄色标出裂缝，青色标出在该机位仍闭合、可与源图对应的板块。',
        19,
    )
    p.text(
        28,
        1150,
        f'原图有 {rows["P135"]["enclosed_islands"]["count"]} 个面积至少 25 cm² 的闭合板块。相机计数要求至少 3 个像素、来源匹配 ≥80%、覆盖该板块可见投影 ≥50%。',
        18,
    )
    p.save(out)


def asphalt_depth(run, items, rows, out):
    p = Plate(1430, 1050)
    p.text(28, 24, '同一条沥青纵缝，深度取多少？', 43, True)
    p.text(
        28, 93, '两个模型使用相同的实拍纹理、轮廓和平面坐标。右图沿黄线比较两件 GLB，纵轴共用毫米尺度。', 21
    )
    p.picture(run / 'evaluation/P134/transect.jpg', 28, 150, 432, 350)
    a, b = items['P134'][1], items['P137'][1]
    if not np.array_equal(a[:, :2], b[:, :2]):
        raise ValueError('The two depth profiles do not share a transect')
    dist = np.linalg.norm(a[:, :2] - a[0, :2], axis=1) * 1000 * rows['P134']['source_texel_m']
    plot(
        p,
        (495, 150, 886, 305),
        dist,
        [('P134 材质标尺', a[:, 3] * 1000, BLUE), ('P137 法线定标', b[:, 3] * 1000, GREEN)],
    )
    p.text(28, 548, '两种深度对同机位行车观测的影响', 28, True)
    visibility = read(run / 'evaluation/visibility.json')
    for x, site in [(28, '034'), (730, '109')]:
        comparison = visibility[site + '_137']['scale_comparison']
        camera = comparison['cameras']['front_camera']
        laser = comparison['lidar']
        p.text(x, 605, f'PandaSet {site}   ·   前视第 {camera["frame"]} 帧', 21, True)
        p.picture(run / 'evaluation/P137' / camera['image'], x, 647, 674, 185)
        p.text(x, 857, f'图中 {camera["changed_pixels"]:,} 个像素的命中部件改变或光轴深度相差超过 1 cm。', 17)
        p.text(
            x,
            897,
            f'80 帧中有 {laser["range_change_over_1cm"]} 束原始射线在两种模型上的模拟量程相差超过 1 cm。',
            17,
        )
    p.draw.line((28, 956, 1399, 956), fill='#bbc9bd', width=2)
    p.text(
        28,
        981,
        '每组左为 P134，右为 P137 并叠加黄色差异标记。数字衡量两种重建在相机与雷达中的差异。现场裂缝深度仍待实际量测。',
        18,
    )
    p.save(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument(
        '--processed',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'tools/multiscene/data/processed',
    )
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = read(a.run / 'evaluation/measurements.json')
    detail = read(a.run / 'series/assets/pothole/P136/metadata.json')
    items = {key: panel(a.run, key, detail) for key in rows}
    for key, (meta, line) in items.items():
        section(a.run, key, meta, line, rows[key])
    portfolio(a.run, rows, items, a.out / 'scanned-cracks.jpg')
    asphalt_depth(a.run, items, rows, a.out / 'asphalt-crack-depth.jpg')
    roads(a.run, rows, camera_frames(a.run, a.processed), a.out / 'scanned-crack-roads.jpg')
