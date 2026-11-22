"""Publish aligned native-mesh, browser Gaussian and depth-comparison photographs from the atlas audit."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'tools/asset_atlas'
Plate = import_module('render-pit-survey').Plate
BG = (223, 225, 216)


def photo(path):
    with Image.open(path) as src:
        rgba = src.convert('RGBA')
        background = Image.new('RGBA', rgba.size, BG + (255,))
        background.alpha_composite(rgba)
        return background.convert('RGB')


def set_image(plate, image, x, y, w, h):
    im = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
    plate.draw.rectangle((x, y, x + w - 1, y + h - 1), fill=BG)
    plate.image.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def group(folder, key, mode):
    stem = key + '-' + mode
    gs = photo(folder / 'browser' / (stem + '-gaussian.png'))
    glb = photo(folder / 'browser' / (stem + '-mesh.png')).resize(gs.size, Image.Resampling.LANCZOS)
    err = photo(folder / 'geometry' / (stem + '.png'))
    marked = np.abs(np.array(gs, dtype='i2') - np.array(BG)).max(2) > 14
    ys, xs = np.where(marked)
    l, r = np.percentile(xs, [0.05, 99.95])
    t, b = np.percentile(ys, [0.05, 99.95])
    pad = max(6, 0.065 * max(r - l, b - t))
    roi = (
        max(0, int(l - pad)),
        max(0, int(t - pad)),
        min(gs.width, int(r + pad + 1)),
        min(gs.height, int(b + pad + 1)),
    )
    return [i.crop(roi) for i in (glb, gs, err)]


def legend(plate, x, y):
    for color, tag, dx in [
        ('#377f71', '<2 mm', 0),
        ('#85ab78', '2–10 mm', 115),
        ('#eeb55b', '10–50 mm', 265),
        ('#bd523a', '≥50 mm', 428),
    ]:
        plate.draw.rectangle((x + dx, y + 4, x + dx + 17, y + 21), fill=color)
        plate.text(x + dx + 25, y, tag, 18)


def portfolio(folder, checks, rows, out):
    picks = [
        ('P112', '双坑近景融合'),
        ('P130', '实拍深坑积水'),
        ('P135', '网裂与碎板'),
        ('M11', '矩形敞井'),
        ('A119', '穿孔圆盖掀开'),
        ('A128', '井旁盖板碎片'),
    ]
    p = Plate(1520, 176 + len(picks) * 237 + 100)
    p.text(28, 22, '同一件道路资产，网格与高斯如何对齐', 43, True)
    p.text(
        28,
        92,
        '前三列共用相机、尺度和裁切。高斯画面由浏览器读取原始 PLY 生成，右列来自同一资产迁入 PandaSet 后的前视相机。',
        21,
    )
    xs = [28, 397, 766, 1135]
    for x, t in zip(xs, ['GLB 实体网格', '3DGS 高斯渲染', '同像素光轴深度绝对差', '行车前视局部']):
        p.text(x, 144, t, 22, True)
    for i, (key, title) in enumerate(picks):
        y = 190 + i * 237
        row = rows[key]
        shot = next(
            v
            for v in row['scenes']
            if v.get('camera') == 'front' and v['after'].endswith('_front_after.webp')
        )
        p.text(28, y, key + '   ' + title + '     ·     ' + row['method'], 21, True)
        p.text(1135, y, shot['label'], 18)
        ims = group(folder, key, 'oblique')
        for x, im in zip(xs, ims):
            set_image(p, im, x, y + 35, 350, 145)
        set_image(p, photo(ATLAS / shot['after']), xs[-1], y + 35, 350, 145)
        q = checks[key]['views']['oblique']
        p.text(
            30,
            y + 195,
            f'{row["gaussians"]:,} 个高斯  ·  在 {q["shared_opaque_front_pixels"]:,} 个共同像素上，深度绝对差中位数 {q["gaussian_expected_vs_glb_depth_mm_median"]:.2f} mm，95 分位 {q["gaussian_expected_vs_glb_depth_mm_p95"]:.1f} mm。',
            17,
        )
    y = 176 + len(picks) * 237
    legend(p, 28, y + 9)
    p.text(
        28,
        y + 55,
        '底色表示未计入统计的像素。网格取首个正面交点，高斯取透明度加权的期望深度。井口周边路面在独立高斯资产中有透明过渡。',
        17,
    )
    p.save(out)


def shaft(folder, checks, out):
    key = 'M11'
    p = Plate(1440, 1144)
    p.text(26, 23, '从三个角度看同一口矩形敞井', 43, True)
    p.text(
        27,
        91,
        'M11 的井圈、井壁和井底都有实体网格与高斯表示。改变机位后，井内表面和周边过渡带的深度差分别有多大？',
        21,
    )
    xs = [26, 497, 968]
    for x, t in zip(xs, ['同机位网格', '原始高斯资产', '共同像素的光轴深度绝对差']):
        p.text(x, 146, t, 21, True)
    for i, (mode, label) in enumerate([('top', '俯视'), ('oblique', '斜视'), ('low', '低角度')]):
        y = 196 + i * 263
        p.text(26, y, label, 22, True)
        for x, im in zip(xs, group(folder, key, mode)):
            set_image(p, im, x, y + 33, 440, 166)
        q = checks[key]['views'][mode]['depth_by_first_mesh_part']
        p.text(
            26,
            y + 214,
            f'按网格首次命中部件分组，绝对差中位数为    井圈  {q["rim"]["median_mm"]:.2f} mm       井壁  {q["wall"]["median_mm"]:.2f} mm       周边路面  {q["apron"]["median_mm"]:.2f} mm',
            19,
        )
    p.draw.line((26, 997, 1408, 997), fill='#bdc6b8', width=2)
    legend(p, 26, 1016)
    p.text(
        26,
        1064,
        '低角度时，独立 PLY 的井外路面逐渐透明，部分射线继续穿到井壁。迁入实拍道路后的外观可在图谱中与该独立渲染对照。',
        18,
    )
    p.save(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', type=Path, default=ROOT / 'validation/atlas-gaussians.json')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'runs/atlas-gaussian-checks')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/figures')
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    checks = json.loads(a.checks.read_text())['browser_vs_cuda_and_glb']
    rows = json.loads((ATLAS / 'catalog.js').read_text().removeprefix('window.roadAssets=')[:-2])
    portfolio(a.evidence, checks, {r['id']: r for r in rows}, a.out / 'gaussian-asset-comparison.jpg')
    shaft(a.evidence, checks, a.out / 'gaussian-shaft-depth.jpg')
