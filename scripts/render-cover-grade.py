"""Render common-scale GLB height sections and a matched driving-camera plate for the grade family."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'gen2hole/downstream/sensors'))
from road_geometry import Surface
from road_camera_frames import driving_frame

INK = '#18302d'
MUTED = '#45635e'
PAPER = '#f5f4ef'
FONTS = Path('/usr/share/fonts/opentype/noto')
RAMP = np.array([[40, 89, 128], [141, 196, 206], [236, 234, 214], [230, 174, 102], [160, 73, 45]])
LIMITS = [-70, -35, 0, 20, 40]


def words(draw, xy, value, size, bold=False, color=INK):
    font = ImageFont.truetype(str(FONTS/('NotoSansCJK-'+('Bold' if bold else 'Regular')+'.ttc')), size, index=2)
    draw.text(xy, value, font=font, fill=color, anchor='lt')


def read(path):
    return json.loads(path.read_text())


def height(mesh, xy):
    rays = np.c_[xy, np.ones(len(xy))]
    dist, _, _ = mesh.cast(rays, np.broadcast_to([0., 0., -1.], rays.shape))
    return (1-dist)*1000


def colors(values):
    rgb = np.stack([np.interp(values, LIMITS, channel) for channel in RAMP.T], -1)
    rgb[values < -80] = [39, 52, 66]
    return np.uint8(rgb)


def section(mesh, dest, title, metrics, span):
    image = Image.new('RGB', (940, 360), PAPER)
    draw = ImageDraw.Draw(image)
    words(draw, (18, 13), title, 25, True)
    axis = np.linspace(-span/2, span/2, 260)
    x, y = np.meshgrid(axis, -axis)
    z = height(mesh, np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    image.paste(Image.fromarray(colors(z)), (18, 56))
    direction = np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
    index = np.array([147.5, 185.5])
    line = direction*[1, -1]*129.5
    draw.line([tuple(index-line), tuple(index+line)], fill='#263632', width=2)
    words(draw, (18, 327), '俯视首交高程 · 黑线为剖面', 17, color=MUTED)
    left, right, top, bottom = 358, 910, 84, 260
    def graph_y(h):
        return top + (44-h)/120 * (bottom-top)
    for h in [-70, -35, 0, 35]:
        py = graph_y(h)
        draw.line((left, py, right, py), fill='#69857d' if h == 0 else '#cfdbd3', width=2 if h == 0 else 1)
        words(draw, (left-44, py-9), str(h), 15)
    words(draw, (left, 52), '相对路面高度（毫米）', 17, color=MUTED)
    xs = np.linspace(-span/2, span/2, 1101)
    hs = height(mesh, xs[:, None] * direction)
    px = np.linspace(left, right, len(xs))
    # Terminate the line at deep openings. Drawing over them would imply a supporting surface.
    visible = np.flatnonzero(np.isfinite(hs) & (hs >= -76))
    for run in np.split(visible, np.flatnonzero(np.diff(visible) != 1)+1):
        if len(run) >= 2:
            draw.line(list(zip(px[run], graph_y(hs[run]))), fill='#aa532e', width=4)
    for at, value in [(left, -span/2), ((left+right)/2, 0), (right-28, span/2)]:
        words(draw, (at-8, 267), f'{value:.2f}', 15)
    words(draw, (left, 301), '剖面距离（米） · 曲线断开处低于显示范围', 17, color=MUTED)
    low, high = np.array(metrics['rim_range_m'])*1000
    words(draw, (left, 328), f'井框外缘 {low:+.1f} 至 {high:+.1f} 毫米   /   井底 {metrics["floor_z_m"]:.2f} 米', 17)
    image.save(dest)


def full_driving_frames(run, cases, raw):
    directory = run/'evaluation'/'camera'
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, case in cases.items():
        folder = run/'sensors/matched_camera'/case
        records[key] = []
        for shot in read(folder/'views.json'):
            if not shot['available']:
                continue
            site, camera, number = shot['site'], shot['camera'], shot['frame']
            original = raw/site/'camera'/camera/f'{number:02d}.jpg'
            rect = shot['crop_xyxy']
            with Image.open(original) as im, Image.open(folder/shot['before_stem']) as before:
                frame = im.convert('RGB')
                if list(frame.size) != shot['calibration_size'] or not np.array_equal(frame.crop(rect), before):
                    raise ValueError('The camera patch does not match the original vehicle image')
            before_file = directory/f'{site}-{camera}-{number}.jpg'
            shutil.copyfile(original, before_file)
            with Image.open(folder/shot['after_stem']) as after:
                expected = np.asarray(after.convert('RGB'))
                if frame.crop(rect).size != after.size:
                    raise ValueError('The calibrated insertion changed crop size')
                frame.paste(after, rect[:2])
            after_file = directory/f'{key}-{site}-{camera}-{number}.webp'
            frame.save(after_file, lossless=True, method=2)
            with Image.open(after_file) as saved:
                if not np.array_equal(saved, frame) or not np.array_equal(saved.crop(rect), expected):
                    raise ValueError('The full vehicle frame changed after lossless encoding')
            box = np.array(shot['xyxy_target_in_crop']) + [rect[0],rect[1]]*2
            records[key].append(dict(camera=camera, site=site, frame=number, roi=rect, bbox=box.tolist(),
                before=str((folder/shot['before_stem']).relative_to(run)), after=str((folder/shot['after_stem']).relative_to(run)),
                full_before=str(before_file.relative_to(run)), full_after=str(after_file.relative_to(run)),
                original_sha256=hashlib.sha256(before_file.read_bytes()).hexdigest(),
                full_after_sha256=hashlib.sha256(after_file.read_bytes()).hexdigest()))
    (directory/'frames.json').write_text(json.dumps(records, indent=2)+'\n')
    return records


def render(run, report, output, config, site):
    bundle = run/'series'
    spec = read(bundle/'calibration/survey.json')
    states = spec['states']
    evidence = read(report)
    for state in states:
        key = state['id']
        folder = bundle/'assets'/('missing' if state.get('absent') else 'raised')/key
        mesh = Surface(folder/'asset.glb')
        section(mesh, run/'evaluation'/key/'grade.png', state['label'], evidence['states'][key]['grade'], 1.0)
    if output is None:
        return
    transfers = evidence['placements']
    cases = {s['asset']: case for case, s in transfers.items() if s['site'] == site}
    frames = full_driving_frames(run, cases, Path(read(config)['pandaset_raw']))
    width, margin, gap = 1640, 38, 12
    cell = (width-2*margin-4*gap)//5
    page = Image.new('RGB', (width, 1554), PAPER)
    draw = ImageDraw.Draw(page)
    words(draw, (margin, 27), '井盖仍在座圈上时，路面能产生多大的高差？', 41, True)
    words(draw, (margin, 92), '同一份摄影测量盖面与 24 个通孔。改变井框和盖板的安装高度，保留外侧路面与井底。', 23, color=MUTED)
    def panel(src, box, fit=True):
        x, y, w, h = box
        with Image.open(src) as im:
            im = im.convert('RGB')
            tile = ImageOps.fit(im, (w,h), Image.Resampling.LANCZOS) if fit else ImageOps.contain(im, (w,h), Image.Resampling.LANCZOS)
            page.paste(tile, (x+(w-tile.width)//2,y+(h-tile.height)//2))
    for i, state in enumerate(states):
        k = state['id']
        x = margin + i*(cell+gap)
        data = evidence['states'][k]['grade']
        lo, hi = np.array(data['rim_range_m'])*1000
        grade = f'{lo:+.0f} mm' if abs(hi-lo)<1 else f'{lo:+.0f} 至 {hi:+.0f} mm'
        words(draw, (x, 151), k+'   '+state['label'], 23, True)
        words(draw, (x, 189), '井框外缘  '+grade, 19, color=MUTED)
        panel(bundle/'views'/(k+'_reverse.jpg'), (x,225,cell,190))
        p = run/'evaluation'/k/'grade.png'
        with Image.open(p) as im:
            tile = im.crop((18,56,278,316)).resize((cell,cell),Image.Resampling.LANCZOS)
        page.paste(tile, (x, 436))
    words(draw,(margin, 436+cell+14), '俯视高程（毫米）', 19, True)
    for i,(color,t) in enumerate(zip([*RAMP, [39,52,66]], ['−70','−35','0','+20','+40','深井 / 通孔'])):
        x=margin+248+108*i
        draw.rectangle((x,436+cell+14,x+26,436+cell+35),fill=tuple(color))
        words(draw,(x+34,436+cell+15),t,17)
    camera_name = 'front_camera'
    shots = []
    for state in states:
        folder = run/'sensors/matched_camera'/cases[state['id']]
        shot = next(v for v in read(folder/'views.json') if v['camera']==camera_name and v['available'])
        shots.append((folder, shot))
    if len({(s['frame'],tuple(s['crop_xyxy'])) for _,s in shots}) != 1:
        raise ValueError('Driving comparison needs a shared frame and crop')
    words(draw,(margin,798),f'同一辆车的前视机位  /  PandaSet {site} · 帧 {shots[0][1]["frame"]}',25,True)
    for i,(folder, shot) in enumerate(shots):
        panel(folder/shot['after_stem'],(margin+i*(cell+gap),846,cell,164),fit=False)
    tilted = next(s for s in states if s['installation_grade']['tilt_deg'] > 0)
    for i, shot in enumerate(frames[tilted['id']]):
        x = margin+i*(width//2-14)
        camera = '前视' if shot['camera']=='front_camera' else '后视'
        words(draw,(x,1042),f'{tilted["id"]}  {tilted["label"]}  /  {camera}全景 · 帧 {shot["frame"]}',23,True)
        with Image.open(run/shot['full_after']) as im:
            context = driving_frame(im.convert('RGB'), shot['bbox'])
        page.paste(context.resize((748,421),Image.Resampling.LANCZOS), (x,1084))
    words(draw,(margin,1518),'盖面平面大小按来源约 1 米图幅换算。起伏幅度、安装高差和 0.72 米井深为设计值。车载行显示模型迁入后的模拟画面。',18,color=MUTED)
    page.save(output, quality=93)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--site', default='109')
    parser.add_argument('--config', type=Path, default=ROOT/'gen2hole/config/local.json')
    args = parser.parse_args()
    render(args.run, args.report, args.out, args.config, args.site)
