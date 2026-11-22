"""Publish the two photographs, recovered geometry, measured profiles and calibrated driving views."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from road_camera_frames import camera_frames, read

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location(
    'pit_plates', Path(__file__).with_name('render-pit-survey.py')
)
plates = importlib.util.module_from_spec(module)
module.loader.exec_module(plates)


def source_crops(bundle, spec, folder):
    for suffix in ['', '_second']:
        key = spec['id'] + suffix
        mask = np.array(Image.open(bundle / 'calibration' / (key + '_outline.png')).convert('L')) > 0
        y, x = np.where(mask)
        pad = int(max(18, max(np.ptp(x), np.ptp(y)) * 0.15))
        box = (
            max(0, x.min() - pad),
            max(0, y.min() - pad),
            min(mask.shape[1], x.max() + pad + 1),
            min(mask.shape[0], y.max() + pad + 1),
        )
        with Image.open(bundle / 'references' / (key + '.png')) as image:
            image.crop(box).save(folder / ('photo' + suffix + '.jpg'), quality=95)


def portfolio(run, measurements, out):
    bundle = run / 'series'
    specs = read(bundle / 'calibration/survey.json')
    street = run / 'evaluation/streets'
    observations = camera_frames(run, Path(read(ROOT / 'gen2hole/config/local.json')['pandaset_processed']))
    groups = {}
    for case, view in observations.items():
        (key,) = read(bundle / 'scenes' / case / 'scene.json')['assets']
        groups.setdefault(key, {})[view['site']] = (case, view)
    end = 169 + len(specs) * 231
    plate = plates.Plate(1530, end + 497)
    plate.text(26, 20, '用两张彩照重建路面凹陷', 44, True)
    plate.text(
        27,
        84,
        '两次曝光、相机内参和一段已知路面长度确定尺度。绿色标出两张照片均支持的区域，橙色为插值。',
        23,
    )
    cols = [26, 275, 524, 880, 1136]
    widths = [234, 234, 339, 240, 368]
    for x, label in zip(
        cols, ['主照片', '第二张照片', '导出 GLB', '视差匹配覆盖', 'PandaSet 109 · 同机位迁入']
    ):
        plate.text(x, 138, label, 21, True)
    for i, s in enumerate(specs):
        key = s['id']
        dest = run / 'evaluation' / key
        source_crops(bundle, s, dest)
        y = 175 + 231 * i
        meta = measurements[key]['geometry']
        plate.text(
            26,
            y,
            f"{key}    {s['label']}     ·     P95 凹深 {1000 * meta['depth_p95_m']:.0f} 毫米     ·     坑内双向匹配 {meta['rgb_supported_target_fraction']:.0%}",
            22,
            True,
        )
        case = groups[key]['109'][0]
        for x, w, image in zip(
            cols,
            widths,
            [
                dest / 'photo.jpg',
                dest / 'photo_second.jpg',
                bundle / 'views' / (key + '_inspection.jpg'),
                dest / 'support.jpg',
                bundle / 'scenes' / case / 'pothole_after.png',
            ],
        ):
            plate.picture(image, x, y + 36, w, 172)
    common = [(k, entry) for k, entry in groups.items() if {'109', '034'} <= entry.keys()]
    if not common:
        raise ValueError('Expected one asset placed on both roads')
    key, sites = common[0]
    for x, site in zip([26, 782], ['034', '109']):
        case, obs = sites[site]
        plate.text(x, end + 12, f"{key}   路段 {site} · 前视第 {obs['frame']} 帧", 24, True)
        plate.picture(street / (case + '-after.jpg'), x, end + 54, 718, 384)
    plate.text(
        27,
        end + 462,
        '参考照片与复核深度来自 PothRGBD / PData、Fan Stereo Pothole Datasets。道路来自 PandaSet · Scale AI and Hesai。',
        16,
    )
    plate.save(out)


def withheld(run, measurements, out):
    key, case = next((k, c) for k, c in measurements.items() if len(c['unseen_rgb']) >= 2)
    bundle = run / 'series'
    root = run / 'evaluation' / key
    n = len(case['unseen_rgb'])
    plate = plates.Plate(1494, 449 + n * 236 + 58)
    plate.text(25, 20, '同一个坑，在留出机位上的形状', 43, True)
    plate.text(
        26, 84, f"{key}    模型与纹理由两张彩照恢复。下面 {n} 个机位在建模完成后用坑外路面纹理定位。", 22
    )
    xs = [25, 378, 731, 1084]
    fit = case['fit']
    for x, title, path in zip(
        xs,
        [f'输入 {fit[0]}', f'输入 {fit[1]}', '双照片 GLB', '双向匹配覆盖'],
        [
            root / 'photo.jpg',
            root / 'photo_second.jpg',
            bundle / 'views' / (key + '_inspection.jpg'),
            root / 'support.jpg',
        ],
    ):
        plate.text(x, 143, title, 22, True)
        plate.picture(path, x, 178, 331, 194)
    plate.text(25, 403, '留出照片上的坑缘与材质', 25, True)
    plate.text(673, 403, '同一机位的截线与相对高度', 25, True)
    for i, name in enumerate(case['unseen_rgb']):
        row = case['frames'][name]
        top = 449 + i * 236
        d = root / name
        plate.text(
            26,
            top,
            f"帧 {name}    坑缘 IoU {row['mask_iou']:.3f}    ·    光轴深度误差中位数 {row['depth_abs_median_mm']:.1f} 毫米",
            20,
            True,
        )
        plate.picture(d / 'contours.jpg', 26, top + 39, 302, 174)
        plate.picture(d / 'material.jpg', 342, top + 39, 302, 174)
        plate.picture(d / 'section.png', 673, top + 1, 794, 211)
    plate.text(
        26,
        plate.image.height - 38,
        '黄线为照片标注，青线为模型回投。剖面按各自的周边路面取零。Fan Stereo Pothole Datasets · Rui Fan et al.',
        16,
    )
    plate.save(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--heldout', type=Path)
    args = parser.parse_args()
    report = read(args.run / 'evaluation/measurements.json')
    portfolio(args.run, report, args.out)
    if args.heldout:
        withheld(args.run, report, args.heldout)
