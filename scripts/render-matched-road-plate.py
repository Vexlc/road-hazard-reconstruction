"""Show independent assets and two original-scale camera views at one physical PandaSet road position."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from road_comparison_data import read_catalog

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
COLS = [
    ('P112', '沿缝双坑'),
    ('P122', '骨料裸露深坑'),
    ('P130', '同一深坑积水'),
    ('P02_wet_wheel_track', '积水车辙坑'),
    ('P135', '混凝土网裂与碎板'),
    ('A117', '矩形雨水篦断条'),
    ('A119', '铺石圆盖掀开'),
    ('M118', '同一井体缺盖'),
]
PAPER, INK, MUTED, RULE = '#f7f5ee', '#20362e', '#516359', '#c7cdbf'


def label(canvas, x, y, s, sz=22, bold=False, color=INK, anchor='lt'):
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-' + ('Bold' if bold else 'Regular') + '.ttc'
    canvas.text((x, y), s, font=ImageFont.truetype(font, sz, index=2), fill=color, anchor=anchor)


def image(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def draw(run, out):
    group = read_catalog(run)['109-01']
    slots = ['front-near', 'front-far']
    base = run / '109-01'
    sheets = {v: image(base / v / 'before.webp') for v in slots}
    photos = {}
    bounds = {}
    for view in slots:
        boxes = []
        for key, _ in COLS:
            directory = base / view
            with (
                Image.open(directory / (key + '.png')) as mask,
                Image.open(directory / (key + '.webp')) as added,
            ):
                roi = mask.getbbox()
                if roi:
                    boxes.append(roi)
                photos[view, key] = Image.alpha_composite(
                    sheets[view].convert('RGBA'), added.convert('RGBA')
                ).convert('RGB')
        boxes = np.array(boxes)
        bounds[view] = np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)]
    width = int(max(max(b[2] - b[0] + 28, 3 * (b[3] - b[1] + 12)) for b in bounds.values()))
    height = int(np.ceil(width / 3))
    boxes = {}
    for view, b in bounds.items():
        W, H = sheets[view].size
        x = int(np.clip((b[0] + b[2] - width) // 2, 0, W - width))
        y = int(np.clip((b[1] + b[3] - height) // 2, 0, H - height))
        if x > b[0] or y > b[1] or x + width < b[2] or y + height < b[3] or width > W or height > H:
            raise ValueError('Common crop clips an illustrated part')
        boxes[view] = (x, y, x + width, y + height)
    w = 1710
    im = Image.new('RGB', (w, 1598), PAPER)
    d = ImageDraw.Draw(im)

    def text(x, y, s, sz=22, bold=False, color=INK):
        label(d, x, y, s, sz, bold, color)

    def paste(photo, x, y, w, h):
        sample = ImageOps.contain(photo, (w, h), method=Image.Resampling.LANCZOS)
        im.paste(sample, (x + (w - sample.width) // 2, y + (h - sample.height) // 2))

    text(38, 28, '八件道路病害资产，同一处车道', 46, True)
    text(40, 100, 'PandaSet 109    /    两个实拍前视机位    /    每次只显示一件原尺度资产', 26, color=MUTED)
    cfg = json.loads((ROOT / 'gen2hole/config/local.json').read_text())
    for j, view in enumerate(slots):
        s = group['shots'][view]
        raw = image(Path(cfg['pandaset_raw']) / '109/camera/front_camera' / f"{s['frame']:02d}.jpg")
        rect = np.array(boxes[view]) + s['rect'][:2] * 2
        ImageDraw.Draw(raw).rectangle(rect.tolist(), outline='#ffdb71', width=5)
        x = 38 + 820 * j
        paste(raw, x, 154, 328, 184)
        text(x + 347, 182, '近处前视' if j == 0 else '远处前视', 28, True)
        text(x + 347, 234, f"第 {s['frame']} 帧    {s['distance_m']:.1f} 米", 24)
        text(x + 347, 280, f'黄框裁切 {width} × {height} 像素', 21, color=MUTED)
    for i, (key, title) in enumerate(COLS):
        row = group['assets'][key]
        x = 38 + (i % 4) * 415
        y = 368 + (i // 4) * 588
        d.line((x, y, x + 390, y), fill=RULE, width=2)
        text(x, y + 12, title, 29, True)
        text(x, y + 54, key + '   ·   ' + row['method'], 17, color=MUTED)
        text(x, y + 82, '生成参考图' if row['method'] == '生成 RGB' else '实拍来源', 18, color=MUTED)
        text(x + 201, y + 82, '独立模型', 18, color=MUTED)
        paste(image(ATLAS / row['reference']['image']), x, y + 106, 189, 124)
        paste(image(ATLAS / row['preview']), x + 201, y + 106, 189, 124)
        for offset, view in [(244, 'front-near'), (413, 'front-far')]:
            crop = photos[view, key].crop(boxes[view])
            paste(crop, x, y + offset, 390, 130)
            n = row['views'][view]['visible']
            text(
                x,
                y + offset + 135,
                ('近处' if view == 'front-near' else '远处') + f'可见网格  {n:,} 像素',
                18,
                color=MUTED,
            )
    text(
        38,
        1555,
        '干坑与积水共用实拍坑底，掀盖与缺盖共用实拍盖面。两行道路局部使用相同的原图像素跨度。',
        22,
        color=MUTED,
    )
    im.save(out, quality=94, subsampling=0)


def beam_plate(run, out):
    site, group_id, view = '109', '109-01', 'front-near'
    group = read_catalog(run)[group_id]
    shot = group['shots'][view]
    folder = run / group_id / view
    keys = ['A119', 'M118']
    left, right = [
        {p[5]: p for p in group['assets'][key]['views'][view]['laser']['points']} for key in keys
    ]
    pairs = [(p, right[k]) for k, p in left.items() if k in right]
    consistent = [
        (p, q)
        for p, q in pairs
        if p[4] is not None
        and q[4] is not None
        and max(abs(p[4]), abs(q[4])) <= 50
        and p[3] == q[3]
        and p[6] == q[6]
    ]
    p, q = max(consistent, key=lambda pair: abs(pair[0][7] - pair[1][7]))
    frame = shot['frame']
    source = ROOT / 'gen2hole/published/道路病害_传感器与几何增强/base' / site / f'{frame:03d}.npz'
    with np.load(source) as f:
        position = f['xyz'][p[5]]
    pose = np.array(shot['pose'])
    cam = (position - pose[:3, 3]) @ pose[:3, :3]
    if cam[2] <= 0:
        raise ValueError('Original return is behind the camera')
    pixel = np.array(shot['K']) @ cam
    raw_xy = np.floor(pixel[:2] / pixel[2]).astype(int) - shot['rect'][:2]
    positions = [raw_xy, p[:2], q[:2]]
    background = image(folder / 'before.webp').convert('RGBA')
    photos = [background]
    boxes = []
    for key in keys:
        with Image.open(folder / (key + '.webp')) as layer, Image.open(folder / (key + '.png')) as mask:
            photos.append(Image.alpha_composite(background, layer.convert('RGBA')))
            boxes.append(mask.getbbox())
    # One crop and zoom for all three panels, including the original beam endpoint.
    hull = np.array([*positions, *[(x, y) for x, y, _, _ in boxes], *[(x, y) for _, _, x, y in boxes]])
    lo, hi = hull.min(0), hull.max(0) + 1
    ratio = 1.55
    crop_width = int(np.ceil(max(hi[0] - lo[0] + 45, (hi[1] - lo[1] + 26) * ratio)))
    crop_height = int(np.ceil(crop_width / ratio))
    size = np.array([crop_width, crop_height])
    corner = np.clip((lo + hi - size) // 2, 0, np.array(background.size) - size)
    box = tuple(np.r_[corner, corner + size])
    dims = (460, round(460 * crop_height / crop_width))
    if (lo < corner).any() or (hi > corner + size).any():
        raise ValueError('Beam illustration clipped')
    ranges = np.array([p[6], p[7], q[7]]) / 1e6
    part_names = {
        5: '井圈与座圈',
        6: '井壁',
        7: '井底',
        8: '混凝土井肩',
        9: '盖面',
        10: '盖板底面',
        11: '盖板边缘',
        12: '铰链',
    }
    captions = ['实拍路面', 'A119　铺石圆盖掀开', 'M118　同一井体缺盖']
    targets = ['记录回波', '模拟命中' + part_names[p[2]], '模拟命中' + part_names[q[2]]]
    shades = ['#615446', '#245c78', '#a45234']
    im = Image.new('RGB', (1500, 860), PAPER)
    d = ImageDraw.Draw(im)
    label(d, 38, 26, '同一束激光，掀盖和缺盖分别挡在哪里', 45, True)
    label(
        d,
        40,
        96,
        f'PandaSet {site}　/　前视第 {frame} 帧　/　传感器 {p[3]}　/　原始扫描点行号 {p[5]}',
        24,
        color=MUTED,
    )
    for j, photo in enumerate(photos):
        x = 38 + 482 * j
        y = 193
        label(d, x, 148, captions[j], 27, True)
        im.paste(photo.crop(box).resize(dims, Image.Resampling.LANCZOS).convert('RGB'), (x, y))
        cx, cy = (np.array(positions[j]) + 0.5 - corner) * np.array(dims) / size + [x, y]
        for radius, color, width in [(10, '#332d29', 5), (10, '#fff9e4', 3)]:
            d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=width)
        label(d, x, y + dims[1] + 11, targets[j] + f'　{ranges[j]:.3f} 米', 22, color=shades[j])
    label(d, 40, 560, '沿同一条发射射线的返回距离', 29, True)
    label(d, 40, 611, f'两种状态相差 {abs(ranges[1] - ranges[2]) * 1000:.1f} 毫米', 25, color=INK)
    label(d, 40, 660, f'{len(pairs)} 束在两种状态下都命中资产', 20, color=MUTED)
    beyond = sum(abs(first[7] - other[7]) > 10000 for first, other in pairs)
    label(d, 40, 699, f'其中 {beyond} 束相差超过 10 毫米', 20, color=MUTED)
    low = np.floor((min(ranges) - 0.05) * 10) / 10
    high = np.ceil((max(ranges) + 0.05) * 10) / 10

    def x_axis(value):
        return 670 + (value - low) / (high - low) * 730

    for tick in np.arange(low, high + 0.001, 0.1):
        x = x_axis(tick)
        d.line((x, 557, x, 729), fill=RULE, width=1)
        label(d, x, 742, f'{tick:.1f}', 16, color=MUTED, anchor='mt')
    for j, v in enumerate(ranges):
        y = 578 + 55 * j
        label(d, 650, y, ['原始', '掀盖', '缺盖'][j], 21, color=shades[j], anchor='rm')
        x = x_axis(v)
        d.line((x_axis(low), y, x, y), fill=shades[j], width=3)
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=shades[j])
    label(d, 1435, 742, '米', 16, color=MUTED)
    d.line((38, 788, 1462, 788), fill=RULE, width=2)
    label(
        d,
        40,
        809,
        '白圈表示各次返回在同一相机中的投影。右侧两栏使用原始光束和迁入资产的网格计算返回位置。',
        21,
        color=MUTED,
    )
    im.save(out, quality=94, subsampling=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/road-comparisons')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/figures/matched-road-assets.jpg')
    parser.add_argument(
        '--beam-figure', type=Path, help='Also illustrate shared laser ranges for the open and missing lid'
    )
    args = parser.parse_args()
    draw(args.run, args.out)
    if args.beam_figure:
        beam_plate(args.run, args.beam_figure)
