"""Compare the four native state replays on identical calibrated vehicle pixels and sensor rows."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PARTS = {
    13: ('碎板顶面', '#409b89'),
    14: ('碎板断面', '#c76f45'),
    15: ('碎板底面', '#7b6491'),
    16: ('基层', '#907048'),
    17: ('坑槽侧壁', '#997ab5'),
}
STATES = {'P144': '原位', 'P145': '凸起', 'P146': '下沉', 'P147': '脱落'}
PAPER = '#f8f6ef'
INK = '#243b33'
MUTED = '#53665b'


def read(p):
    return json.loads(p.read_text())


def rgb(p):
    with Image.open(p) as im:
        return im.convert('RGB')


def text(d, x, y, s, size, bold=False, color=INK):
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-' + ('Bold' if bold else 'Regular') + '.ttc'
    d.text((x, y), s, font=ImageFont.truetype(font, size, index=2), fill=color, anchor='lt')


def reads_by_beam(file):
    with np.load(file) as f:
        frame = np.repeat(np.arange(80), np.diff(f['frame_offsets']))
        use = f['action'] == 1
        ids = list(zip(frame[use].tolist(), f['row'][use].tolist()))
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate native row')
        return {
            key: (int(s), int(p), float(b), float(a))
            for key, s, p, b, a in zip(
                ids, f['sensor'][use], f['part'][use], f['range_before_m'][use], f['range_after_m'][use]
            )
        }


def compare(run, out, report):
    sites = ['034', '109']
    fig = Image.new('RGB', (1662, 1180), PAPER)
    d = ImageDraw.Draw(fig)
    text(d, 36, 24, '碎板位置怎样改变行车相机看到的表面', 43, True)
    text(d, 38, 89, '两段实拍道路　/　每行共用前视帧、相机姿态、放置中心和像素范围', 24, color=MUTED)
    proof = {}
    for j, site in enumerate(sites):
        observations = {}
        scans = {}
        for key in STATES:
            case = site + '_' + key[1:]
            folder = run / 'sensors/camera' / case
            camera = {}
            for shot in read(folder / 'views.json'):
                if not shot['available']:
                    raise ValueError('Missing camera ' + case)
                with np.load(folder / shot['geometry']) as g:
                    parts = g['part']
                source = rgb(folder / shot['before_stem'])
                camera[shot['camera']] = (shot, source, rgb(folder / shot['after_stem']), parts)
            observations[key] = camera
            scans[key] = reads_by_beam(run / 'sensors/cases' / (case + '.npz'))
        proof[site] = {'camera': {}, 'lidar': {}}
        for cam in observations['P144']:
            base, raw, *_ = observations['P144'][cam]
            counts = {}
            for key in STATES:
                shot, before, _, parts = observations[key][cam]
                if any(
                    shot[n] != base[n] for n in ('frame', 'crop_xyxy', 'K_full', 'camera_to_world')
                ) or not np.array_equal(np.asarray(before), np.asarray(raw)):
                    raise ValueError('Vehicle images are not registered')
                counts[key] = {str(n): int((parts == n).sum()) for n in PARTS}
                if key == 'P147' and (np.isin(parts, [13, 14, 15]).any() or not (parts == 16).any()):
                    raise ValueError('Missing state does not expose its foundation')
            proof[site]['camera'][cam] = dict(
                frame=base['frame'], crop=base['crop_xyxy'], pixels_by_part=counts
            )
        for key, hits in scans.items():
            ref = scans['P144']
            shared = sorted(hits.keys() & ref.keys())
            if not shared:
                raise ValueError('No common hit rows')
            differences = []
            for ident in shared:
                sensor, _, before, after = hits[ident]
                orig = ref[ident]
                if orig[0] != sensor or abs(before - orig[2]) > 1e-6:
                    raise ValueError('Original beam differs')
                differences.append(after - orig[3])
            proof[site]['lidar'][key] = dict(
                simulated_hits=len(hits),
                by_part=dict(Counter(str(a[1]) for a in hits.values())),
                shared_with_in_place=len(shared),
                over_10mm=sum(abs(x) > 0.01 for x in differences),
                min_max_difference_mm=(1000 * np.array([min(differences), max(differences)])).tolist(),
            )
        # Use the registered photograph as the backdrop for the semantic overlay.
        cam = 'front_camera'
        base, raw, *_ = observations['P144'][cam]
        bbox = np.array(
            [Image.fromarray(np.uint8(v[cam][3] >= 13) * 255).getbbox() for v in observations.values()]
        )
        lo, hi = bbox[:, :2].min(0), bbox[:, 2:].max(0)
        span = np.ceil(max(hi[0] - lo[0] + 22, 1.8 * (hi[1] - lo[1] + 18))).astype(int)
        size = np.array([span, int(np.ceil(span / 1.8))])
        origin = np.clip((lo + hi - size) // 2, 0, np.array(raw.size) - size)
        box = tuple(np.r_[origin, origin + size])
        y0 = 154 + 480 * j
        text(d, 37, y0, f'PandaSet {site}　前视第 {base["frame"]} 帧', 30, True)
        for i, key in enumerate(['road', *STATES]):
            x = 36 + i * 321
            text(d, x, y0 + 59, '原始路面' if key == 'road' else STATES[key] + '　' + key, 23, True)
            im = raw if key == 'road' else observations[key][cam][2]
            fig.paste(im.crop(box).resize((303, 168), Image.Resampling.LANCZOS), (x, y0 + 99))
            if key == 'road':
                cfg = read(ROOT / 'gen2hole/config/local.json')
                context = rgb(
                    Path(cfg['pandaset_raw']) / site / 'camera/front_camera' / f"{base['frame']:02d}.jpg"
                )
                pos = (np.array(box) + base['crop_xyxy'][:2] * 2).tolist()
                ImageDraw.Draw(context).rectangle(pos, outline='#ffe17b', width=7)
                fig.paste(ImageOps.contain(context, (303, 167)), (x, y0 + 277))
                text(d, x, y0 + 450, '实拍全景与局部位置', 18, color=MUTED)
            else:
                parts = observations[key][cam][3]
                a = np.array(im)
                for code, (_, color) in PARTS.items():
                    mat = np.frombuffer(bytes.fromhex(color[1:]), dtype=np.uint8)
                    a[parts == code] = np.rint(0.25 * a[parts == code] + 0.75 * mat)
                fig.paste(
                    Image.fromarray(a).crop(box).resize((303, 168), Image.Resampling.LANCZOS), (x, y0 + 277)
                )
                code = 16 if key == 'P147' else 14 if key == 'P145' else 17 if key == 'P146' else 13
                text(d, x, y0 + 450, f'{PARTS[code][0]}　{(parts == code).sum():,} 个像素', 18, color=MUTED)
    for i, (code, (label, color)) in enumerate(PARTS.items()):
        x = 38 + i * 245
        d.rectangle((x, 1130, x + 23, 1153), fill=color)
        text(d, x + 32, 1130, label, 20)
    fig.save(out, quality=95, subsampling=0)
    report.write_text(json.dumps(proof, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    compare(a.run, a.out, a.report)
