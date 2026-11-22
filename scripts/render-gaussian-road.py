"""Make a same-camera plate from the actual WebGL road and underground-colour passes."""

import argparse
from importlib import import_module
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Plate = import_module('render-pit-survey').Plate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=ROOT / 'runs/atlas-host-road')
    parser.add_argument('--checks', type=Path, default=ROOT / 'validation/atlas-host-road.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/figures/gaussian-road-openings.jpg')
    args = parser.parse_args()
    checked = json.loads(args.checks.read_text())['audited_assets']
    cases = [
        ('M11', 'low', '矩形敞井', '生成 RGB'),
        ('M10', 'oblique', '圆形敞井', '生成 RGB'),
        ('A117', 'oblique', '断条雨水篦', '实拍 RGB'),
        ('A119', 'low', '穿孔圆盖掀开', '摄影测量盖面'),
        ('A125', 'oblique', '完整铸铁盖落座', '摄影测量盖面'),
        ('A128', 'oblique', '破损盖板与井旁断片', '摄影测量盖面'),
    ]
    width, pad, gap, image_w, image_h = 1536, 24, 12, 363, 180
    xs = [pad + i * (image_w + gap) for i in range(4)]
    row_height, top = 269, 169
    plate = Plate(width, top + len(cases) * row_height + 72)
    plate.text(pad, 24, '铺上带开孔的路面后，哪些地下结构还看得见？', 41, True)
    plate.text(pad, 84, '每行固定相机。洋红色标出低于路面 7.5 cm 的高斯，其位置、大小和透明度保持原值。', 22)
    for x, title in zip(xs, ['独立高斯资产', '标出地下结构', '铺路后标出地下结构', '恢复原始颜色']):
        plate.text(x, 131, title, 22, True)
    for i, (asset, mode, title, source) in enumerate(cases):
        y = top + i * row_height
        data = checked[asset]['views'][mode]
        plate.text(
            pad,
            y,
            f'{asset}    {title}    ·    {source}    ·    ' + ('低角度' if mode == 'low' else '斜视'),
            21,
            True,
        )
        for x, kind in zip(xs, ['gaussian', 'probe', 'road-probe', 'road']):
            plate.picture(args.captures / f'{asset}-{mode}-{kind}.png', x, y + 32, image_w, image_h)
        plate.text(
            pad,
            y + 224,
            f'井口外响应像素由 {data["exterior_raw_probe_pixels"]:,} 降至 {data["exterior_host_probe_pixels"]:,}。'
            f'  井口内 {data["interior_pixels"]:,} 个核验像素，铺路前后 RGB 通道最大差为 {data["interior_max_rgb_change"]} / 255。',
            18,
        )
    y = top + len(cases) * row_height
    plate.draw.line((pad, y, width - pad, y), fill='#bdc6b8', width=2)
    plate.text(
        pad,
        y + 23,
        '开孔来自各自 GLB 的铺面内缘。沥青采用 Poly Haven 的 3 米扫描材质。统计由原始路面三角形独立求交，剔除两像素边界。',
        18,
    )
    plate.save(args.out)


if __name__ == '__main__':
    main()
