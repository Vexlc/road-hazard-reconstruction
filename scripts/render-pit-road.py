"""Display native Gaussian pits, their road transitions and recorded vehicle-camera observations."""

import argparse
from importlib import import_module
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
Plate = import_module('render-pit-survey').Plate


def paste(plate, image, x, y, w, h):
    small = ImageOps.contain(image.convert('RGB'), (w, h), Image.Resampling.LANCZOS)
    plate.image.paste(small, (x + (w - small.width) // 2, y + (h - small.height) // 2))


def portfolio(rows, checks, folder, out):
    cases = [
        ('rgbd_pothole_20250227_135438', '深度相机粗骨料坑'),
        ('stereo_pothole_fan_1_03', '双目小坑'),
        ('P112', '沿缝相邻的双坑'),
        ('P130', '深坑积水'),
        ('P135', '混凝土网裂与碎板'),
        ('P02_wet_wheel_track', '积水车辙'),
    ]
    xs = [24, 399, 774, 1149]
    top, row_height = 169, 252
    plate = Plate(1536, top + len(cases) * row_height + 55)
    plate.text(24, 23, '同一件坑洞资产，从拍摄底色移到道路上', 41, True)
    plate.text(24, 83, '前三列是同一 PLY、机位与尺度，第四列是同一资产迁入 PandaSet 后的前视局部。', 21)
    for x, t in zip(xs, ['独立高斯资产', '保留来源路面', '沿坑缘过渡到扫描沥青', '车载前视']):
        plate.text(x, 129, t, 22, True)
    for i, (key, label) in enumerate(cases):
        row = rows[key]
        y = top + i * row_height
        view = next(s for s in row['scenes'] if s.get('camera') == 'front')
        plate.text(24, y, label + '    ·    ' + row['method'], 21, True)
        plate.text(xs[-1], y, view['label'], 17)
        for x, n in zip(xs, ['gaussian', 'unblended', 'road']):
            plate.picture(folder / f'{key}-oblique-{n}.png', x, y + 32, 360, 174)
        plate.picture(ATLAS / view['after'], xs[-1], y + 32, 360, 174)
        q = checks[key]['views']['oblique']
        plate.text(
            25,
            y + 213,
            f'{key}    ·    开孔投影交并比 {100 * q["footprint_iou"]:.2f}%    ·    开孔内 {q["interior_pixels"]:,} 个核验像素的最大颜色差 {q["interior_max_rgb_change"]} / 255',
            17,
        )
    plate.text(
        24,
        top + len(cases) * row_height + 10,
        '前视局部采用各自的实车相机和放置参数。积水为施加在实拍坑底上的模拟水面。',
        17,
    )
    plate.save(out)


def driving(rows, observations, folder, out):
    # Select a distinct measured instance per road and sensor near the upper
    # quartile of projected support, with a visible change on most target pixels.
    selected = []
    for method in ('实拍 RGB-D', '实拍双目'):
        for site in ('055', '056'):
            shots = []
            for key, a in observations.items():
                if rows[key]['method'] != method or a['site'] != site:
                    continue
                candidates = [
                    (s['eligible_pixels'], f, s)
                    for f, s in a['frames'].items()
                    if s['status'] == 'visible' and s['changed_pixels'] > 0.6 * s['eligible_pixels']
                ]
                if candidates:
                    count, frame, shot = max(candidates)
                    shots.append((count, key, frame, shot, a['case'], site))
            shots.sort()
            if not shots:
                raise ValueError('No verified driving frame for ' + method + ' ' + site)
            selected.append(shots[round((len(shots) - 1) * 0.8)])
    xs = [24, 555, 852, 1149]
    p = Plate(1500, 1412)
    p.text(24, 24, '从驾驶员的位置，坑洞在画面里占多少像素？', 40, True)
    p.text(
        24, 84, '第一列保留完整前视画面，黄框标出局部。中间两列截取同一位置。右列是对应的独立高斯资产。', 21
    )
    for x, t in zip(xs, ['联合场景的车载机位', '迁入前', '迁入后', '原始资产']):
        p.text(x, 142, t, 22, True)
    for i, (count, key, frame, shot, case, site) in enumerate(selected):
        y = 190 + 295 * i
        row = rows[key]
        p.text(24, y, f'{site} 路段    ·    前视第 {frame} 帧    ·    ' + row['method'], 21, True)
        full = ROOT / 'outputs/road_hazard_joint_scenes' / case / f'drive_{frame}.jpg'
        with Image.open(full) as img:
            marked = img.convert('RGB')
            ImageDraw.Draw(marked).rectangle(shot['crop'], outline='#ffe16f', width=5)
            paste(p, marked, xs[0], y + 32, 512, 191)
        for x, n in zip(xs[1:3], ('before', 'after')):
            p.picture(folder / shot[n], x, y + 32, 282, 191)
        p.picture(ATLAS / row['gaussianPreview'], xs[-1], y + 32, 323, 191)
        x0, y0, x1, y1 = shot['bbox']
        p.text(
            25,
            y + 237,
            f'{key}    ·    坑面投影外框 {x1 - x0} × {y1 - y0} 像素    ·    {count:,} 个可核验像素中有 {shot["changed_pixels"]:,} 个超过画面差值阈值。',
            18,
        )
    p.text(
        24, 1370, '前后画面由同一道路重建在实车标定机位渲染。核验排除了动态目标和同组其他模型遮挡的射线。', 17
    )
    p.save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/figures')
    a = parser.parse_args()
    rows = {
        r['id']: r
        for r in json.loads((ATLAS / 'catalog.js').read_text().removeprefix('window.roadAssets=')[:-2])
    }
    checks = json.loads((ROOT / 'validation/atlas-pit-road.json').read_text())['audited_assets']
    portfolio(rows, checks, ROOT / 'runs/atlas-pit-road/browser', a.out / 'gaussian-pit-roads.jpg')
    folder = ROOT / 'runs/capture-driving'
    driving(rows, json.loads((folder / 'views.json').read_text()), folder, a.out / 'capture-pit-driving.jpg')


if __name__ == '__main__':
    main()
