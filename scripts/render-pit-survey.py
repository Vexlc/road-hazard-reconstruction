"""Render a measured multi-view survey plate and a matched double-pit reconstruction comparison."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
INK = '#1e3930'
PAPER = '#f2f1e9'
PALE = '#dfe3da'


def read(p):
    return json.loads(p.read_text())


def driving_frame(image, bounds):
    """Keep the driver's full field of view and enlarge the same marked patch."""
    w, h = image.size
    left, top, right, bottom = bounds
    half_width = max(28, int((right - left) * 0.63), int((bottom - top) * 1.40))
    half_height = max(14, round(half_width / 2.2))
    cx = min(max((left + right) // 2, half_width), w - half_width)
    cy = min(max((top + bottom) // 2, half_height), h - half_height)
    region = (cx - half_width, cy - half_height, cx + half_width, cy + half_height)
    detail = ImageOps.fit(image.crop(region), (308, 140), method=Image.Resampling.LANCZOS)
    full = image.resize((1280, 720), Image.Resampling.LANCZOS)
    border = '#ffe181'
    ImageDraw.Draw(full).rectangle(
        tuple(round(v * 1280 / w) if i % 2 == 0 else round(v * 720 / h) for i, v in enumerate(region)),
        outline=border,
        width=2,
    )
    full.paste(ImageOps.expand(detail, border=2, fill=border), (950, 18))
    return full


def camera_frames(run, processed):
    folder = run / 'evaluation/streets'
    folder.mkdir(parents=True, exist_ok=True)
    frames = {}
    for path in sorted((run / 'series/sensor_views').glob('*/frames.json')):
        sequence = read(path)
        case = sequence['case']
        site = read(run / 'series/scenes' / case / 'scene.json')['site']
        example = run / 'series/migration' / case / 'sensor_examples' / case
        best = read(example / 'pothole.json')
        frame, box = best['frame'], best['bbox']
        row = next(f for f in sequence['frames'] if f['frame'] == frame)
        if row['class_bboxes_xyxy'][0] != box or row['class_pixels'][0] != best['count']:
            raise ValueError('Mismatched source frame or asset bounds in ' + case)
        with Image.open(example / 'pothole_after.jpg') as after:
            if list(after.size) != sequence['size']:
                raise ValueError('Mismatched full driving frame in ' + case)
            driving_frame(after, box).save(folder / (case + '-after.jpg'), quality=94)
        with Image.open(processed / site / 'images' / f'{frame:03d}_0.jpg') as before:
            if list(before.size) != sequence['size']:
                raise ValueError('Mismatched camera image size in ' + case)
            driving_frame(before, box).save(folder / (case + '-before.jpg'), quality=94)
        frames[case] = dict(site=site, frame=frame, bbox=box, visible_pixels=row['class_pixels'][0])
    (folder / 'frames.json').write_text(json.dumps(frames, indent=2) + '\n')
    return frames


class Plate:
    def __init__(self, w, h):
        self.image = Image.new('RGB', (w, h), PAPER)
        self.draw = ImageDraw.Draw(self.image)

    def text(self, x, y, s, size=20, bold=False):
        font = ImageFont.truetype(FONT.replace('Regular', 'Bold') if bold else FONT, size, index=2)
        self.draw.text((x, y), s, font=font, fill=INK, anchor='lt')

    def picture(self, p, x, y, w, h):
        self.draw.rectangle((x, y, x + w - 1, y + h - 1), fill=PALE)
        with Image.open(p) as im:
            im = ImageOps.contain(im.convert('RGB'), (w, h), Image.Resampling.LANCZOS)
            self.image.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))

    def save(self, p):
        self.image.save(p, quality=94, subsampling=0)


def portfolio(run, report, streets, out):
    b = run / 'series'
    cols = [28, 382, 736, 1090]
    w = 340
    cases = {
        p['asset_id']: (case.parent, p['category'])
        for case in sorted((b / 'scenes').glob('109_*/scene.json'))
        for p in read(case)['placements']
    }
    road_top = 178 + len(report) * 255
    plate = Plate(1458, road_top + 540)
    plate.text(28, 20, '多帧重建的五组坑洞与道路迁入', 43, True)
    plate.text(28, road_top + 4, '同一组双坑 P113 在两段自驾道路上的可见尺寸', 29, True)
    for x, site in [(28, '034'), (736, '109')]:
        case = site + '_113'
        obs = streets[case]
        plate.text(x, road_top + 57, f"PandaSet {site} · 前视第 {obs['frame']} 帧", 20, True)
        plate.picture(run / 'evaluation/streets' / (case + '-after.jpg'), x, road_top + 92, 694, 390)
    plate.text(
        29,
        84,
        '黄线为留出照片的原始标注，青线为导出 GLB 的回投。深度误差取两者重合且有有效深度的像素。',
        21,
    )
    for x, s in zip(cols, ['留出照片', '实拍纹理 GLB', '坑缘回投', '迁入 PandaSet 109']):
        plate.text(x, 136, s, 22, True)
    for i, (key, row) in enumerate(report.items()):
        meta = read(b / 'assets/pothole' / key / 'metadata.json')
        frame = next(iter(row['heldout']))
        evaluation = run / 'evaluation' / key / frame
        case, cat = cases[key]
        top = 177 + i * 255
        plate.text(28, top, f"{key}   {meta['label']}     ·     {row['source']}", 20, True)
        for x, p in zip(
            cols,
            [
                evaluation / 'photo.jpg',
                b / 'views' / (key + '_inspection.jpg'),
                evaluation / 'contours.jpg',
                case / (cat + '_after.png'),
            ],
        ):
            plate.picture(p, x, top + 33, w, 173)
        s = row['heldout'][frame]['exported_glb']
        plate.text(
            28,
            top + 216,
            f"宽 × 长 {meta['width_m']:.2f} × {meta['length_m']:.2f} m      P95 凹深 {meta['depth_p95_m'] * 1000:.0f} mm      留出帧 {frame}      坑缘 IoU {s['mask_iou']:.3f}      深度绝对误差中位数 {s['depth_abs_median_mm']:.1f} mm",
            17,
        )
    plate.text(
        28,
        plate.image.height - 37,
        '图像来自 PothRGBD / PData、Fan Stereo Pothole Datasets 和 PandaSet。PothRGBD 使用标称内参。',
        16,
    )
    plate.save(out)


def double(run, report, out):
    key = 'P112'
    row = report[key]
    frame, data = next(iter(row['heldout'].items()))
    fit = row['fit']
    b = run / 'series'
    obs = run / 'evaluation' / key
    page = Plate(1430, 1012)
    page.text(30, 19, '两个近景，复原一组双坑', 43, True)
    page.text(30, 82, '两张近景分别记录两个坑，利用坑外路面定位。全景中的坑缘、纹理和深度留作核验。', 22)
    xs = [30, 497, 964]
    for x, title in zip(xs, ['近景一', '近景二', '融合 GLB · 同一坐标系']):
        page.text(x, 141, title, 23, True)
    for x, p in zip(
        xs,
        [
            obs / 'input' / (fit[0] + '.jpg'),
            obs / 'input' / (fit[1] + '.jpg'),
            b / 'views' / (key + '_inspection.jpg'),
        ],
    ):
        page.picture(p, x, 179, 436, 246)
    for x, title in zip(xs, ['只用近景一', '只用近景二', '融合两个近景']):
        page.text(x, 464, title, 24, True)
    for x, p in zip(
        xs,
        [
            obs / frame / ('single-' + fit[0] + '.jpg'),
            obs / frame / ('single-' + fit[1] + '.jpg'),
            obs / frame / 'contours.jpg',
        ],
    ):
        page.picture(p, x, 507, 436, 242)
    values = [data['single_fit_frame_controls'][k]['mask_iou'] for k in fit] + [
        data['exported_glb']['mask_iou']
    ]
    for x, iou in zip(xs, values):
        page.text(x, 766, f'留出全景坑缘 IoU   {iou:.3f}', 22, True)
    page.text(30, 827, '黄线显示同一张全景中的两个坑缘，青线显示各次重建的回投轮廓。', 21)
    page.text(
        30,
        885,
        f"模型中心距 {row['glb_center_spacing_m']:.3f} m     留出深度中心距 {data['heldout_center_spacing_m']:.3f} m     两坑标注召回率 {data['exported_glb']['mask_recall'] * 100:.1f}%",
        24,
        True,
    )
    page.text(
        30, 958, 'PothRGBD / PData · 标称深度相机内参。中心距由局部路面上两个标注区域的面积中心估计。', 18
    )
    page.save(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument(
        '--processed',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'tools/multiscene/data/processed',
    )
    p.add_argument('--out', type=Path, required=True, help='Output directory for two JPEG figures')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = read(args.run / 'evaluation/measurements.json')
    streets = camera_frames(args.run, args.processed)
    portfolio(args.run, results, streets, args.out / 'multiview-potholes.jpg')
    double(args.run, results, args.out / 'double-pothole.jpg')
