"""Compare the five double-hatch states at identical recorded exposures and magnification."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--views', type=Path, default=ROOT/'outputs/twin-hatch')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/twin-hatch/comparison.jpg')
    args = ap.parse_args()
    cases = [('A228','双盖齐平'), ('A229','右盖移位'), ('M229','左盖缺失'), ('M230','右盖缺失'), ('M231','双盖缺失')]
    page = Image.new('RGB', (1530, 782), '#f3f2ed')
    draw = ImageDraw.Draw(page)
    fontfile = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(fontfile, 20, index=2)
    title = ImageFont.truetype(fontfile, 32, index=2)
    draw.text((20, 9), '双联检查井在三段实车道路中的外观', font=title, fill='#1a302a')
    for col,label in enumerate(['原路面', *[key + '  ' + name for key,name in cases]]):
        draw.text((20 + col*252, 76), label, font=font, fill='#1e3933')
    for i,(scene,sensor,caption) in enumerate([
        ('a2d2-frontleft','rear_center','Audi A2D2 后视鱼眼'),
        ('kitti360-0000','image_03','KITTI-360 0000 右侧鱼眼'),
        ('kitti360-0003-right','image_00','KITTI-360 0003 前视针孔')
    ]):
        base = args.views / scene
        index = read_json(base/'index.json')
        views = [index[key]['views'][sensor] for key,_ in cases]
        if len({r['source_rgb_sha256'] for r in views}) != 1:
            raise ValueError('States were recorded at different exposures')
        layers, rects = [], []
        for (key,_), row in zip(cases,views):
            folder = base/key/sensor
            xy = np.asarray(row['crop_xy'])
            box = np.asarray(Image.open(folder/'part.png').getbbox()) + np.tile(xy,2)
            rects.append(box)
            if not layers:
                layers.append((Image.open(folder/'before.png'), xy))
            layers.append((Image.open(folder/'after.png'), xy))
        bounds = np.array(rects)
        lo, hi = bounds[:,:2].min(0), bounds[:,2:].max(0)
        pad = max(10, round(float((hi-lo).max()) * .10))
        lo = np.maximum(lo - pad, np.max([xy for _,xy in layers], axis=0))
        hi = np.minimum(hi + pad, np.min([xy + im.size for im,xy in layers], axis=0))
        for col, (im,xy) in enumerate(layers):
            panel = ImageOps.contain(im.crop((*list(lo-xy), *list(hi-xy))), (242, 166), Image.Resampling.LANCZOS)
            page.paste(panel, (16 + 252*col + (242-panel.width)//2, 112 + 217*i + (166-panel.height)//2))
        draw.text((20, 280 + 217*i), f"{caption}   ·   实际曝光 {views[0]['frame']}", font=font, fill='#28443b')
    args.out.parent.mkdir(exist_ok=True, parents=True)
    page.save(args.out, quality=94)


if __name__ == '__main__':
    main()
