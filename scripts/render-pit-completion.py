"""Compare real missing ranges, the published surface and calibrated driving views for P142 and P143."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from road_camera_frames import camera_frames, read


ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc'
PAPER, INK, MUTED, RULE = '#f7f5ee', '#1d2b2c', '#475b59', '#c9cfca'


def plate(run, output):
    records = {m['id']: m for m in read(run / 'series/assets/asset_manifest.json')}
    audit = read(run / 'evaluation/measurements.json')
    frames = camera_frames(run, Path(read(ROOT / 'gen2hole/config/local.json')['pandaset_processed']))
    im = Image.new('RGB', (1480, 1634), PAPER)
    draw = ImageDraw.Draw(im)

    def label(x, y, value, size=24, bold=False, color=INK):
        font = ImageFont.truetype(FONT.format('Bold' if bold else 'Regular'), size, index=2)
        draw.text((x, y), value, fill=color, font=font, anchor='lt')

    def pane(path, x, y, w, h, fit=False, nearest=False):
        with Image.open(path) as src:
            resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
            if fit:
                view = ImageOps.fit(src, (w, h), method=resample)
            else:
                view = ImageOps.contain(src, (w, h), method=resample)
            im.paste(view, (x + (w - view.width) // 2, y + (h - view.height) // 2))

    label(44, 30, '双目深度缺失时，坑壁还能恢复多少？', 51, True)
    label(46, 111, '同一处实拍坑洞的原始观测、彩照补深、留出机位和原尺度道路迁入', 29, color=MUTED)
    draw.line((45, 167, 1435, 167), fill=RULE, width=2)
    for x, key, fit, holdout, site in [(45, 'P142', '03', '01', '034'), (755, 'P143', '18', '19', '109')]:
        record = records[key]
        folder = run / 'evaluation' / key
        label(x, 194, key + '   ' + record['label'], 38, True)
        label(
            x,
            253,
            '{:.1f} × {:.1f} 厘米   ·   坑内 P95 凹深 {:.1f} 毫米'.format(
                100 * record['width_m'], 100 * record['length_m'], 1000 * record['depth_p95_m']
            ),
            24,
            color=MUTED,
        )
        pane(run / 'series/views' / (key + '_inspection.jpg'), x, 302, 680, 320, fit=True)
        label(x, 637, '带实拍纹理的三维网格', 24)
        pane(folder / 'input' / (fit + '-missing.png'), x, 704, 332, 234)
        pane(folder / 'input' / (fit + '-recovered.png'), x + 348, 704, 332, 234)
        label(x, 953, '帧 ' + fit + '  橙色为缺失深度', 23)
        label(x + 348, 953, '青色为双向匹配补出的像素', 23)
        pane(folder / 'input/range-coverage.png', x, 1031, 232, 201, nearest=True)
        pane(folder / holdout / 'completion-errors.png', x + 244, 1031, 436, 201)
        pairs = audit[key]['heldout'][holdout]['paired_completion']['photographic']
        label(x, 1245, '坑内网格的高程来源', 22)
        label(
            x + 244,
            1245,
            '留出帧 ' + holdout + '  ·  ' + format(pairs['paired_rays'], ',') + ' 个同位置像素',
            22,
        )
        label(x, 1284, f"补出 {record['photo_only_pit_cells']} 格", 21, color=MUTED)
        label(
            x + 244,
            1284,
            '{:.1f} → {:.1f} 毫米 · 绝对误差中位数'.format(
                pairs['range_only']['median_mm'], pairs['completed']['median_mm']
            ),
            21,
            color=MUTED,
        )
        case = site + '_' + key[1:]
        car = frames[case]
        pane(run / 'evaluation/streets' / (case + '-after.jpg'), x, 1350, 396, 222)
        label(x + 414, 1384, 'PandaSet ' + site, 27, True)
        label(x + 414, 1435, '前视第 ' + str(car['frame']) + ' 帧', 23)
        label(x + 414, 1480, f"可见 {car['visible_pixels']} 像素", 23, color=MUTED)
    for y in (681, 1009, 1329):
        draw.line((45, y, 1435, y), fill=RULE, width=2)
    label(
        45,
        1596,
        '网格图中蓝灰为原始距离，青色为彩照补深，橙色为插值。留出帧中青色表示误差减小，红色表示增大，着色阈值为 1 毫米。',
        20,
        color=MUTED,
    )
    im.save(output, quality=95, subsampling=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/pit-depth-completion')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/figures/pit-depth-completion.jpg')
    args = parser.parse_args()
    plate(args.run, args.out)
