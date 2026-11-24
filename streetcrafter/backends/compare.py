"""Compare saved renders at matching frame IDs and encode the driving replay."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity


def read(path, size):
    rgb = cv2.imread(str(path))
    if rgb is None or rgb.shape[1::-1] != tuple(size):
        raise ValueError(f'Missing or incorrectly sized frame {path}')
    return rgb


def metrics(ref, rgb):
    mse = np.square((rgb.astype(np.float32) - ref)/255).mean()
    return {'psnr_db': float(-10*np.log10(max(float(mse), 1e-12))),
            'ssim': float(structural_similarity(ref, rgb, channel_axis=2, data_range=255,
                                                gaussian_weights=True, use_sample_covariance=False))}


@lru_cache(maxsize=32)
def bar(text, width, height, header=False):
    canvas = Image.new('RGB', (width, height), '#1c1e20' if header else '#151618')
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 19, index=2)
    ImageDraw.Draw(canvas).text((13, height//2), text, font=font, anchor='lm',
                                fill='#bce1d7' if header else '#eeeeee')
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def tile(image, title, width=640, height=360):
    scale = min(width/image.shape[1], height/image.shape[0])
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = np.full((height+36, width, 3), (24, 22, 21), np.uint8)
    h, w = scaled.shape[:2]
    y, x = 36+(height-h)//2, (width-w)//2
    out[y:y+h, x:x+w] = scaled
    out[:36] = bar(title, width, 36)
    return out


def compare(root, scene, steps):
    bundle = root / 'work/streetcrafter-backends' / scene
    run = root / 'runs/streetcrafter-backends' / scene
    output = root / 'outputs/streetcrafter-backends' / scene / str(steps)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((bundle / 'all/split.json').read_text())
    gut, sc = run / f'3dgut-{steps}', run / f'sc-gs-{steps}'
    fish = (bundle / 'fish_all').exists()
    size = meta['resolution']
    methods = {'3DGUT': gut / 'frames', 'SC-GS': sc / 'frames'}
    if fish and (run / f'sc-gs-fisheye-{steps}').exists():
        sc = run / f'sc-gs-fisheye-{steps}'
        methods['SC-GS'] = sc / 'frames'
    if not fish:
        host = root / ('outputs/scene_A_056' if scene == '056' else 'outputs/scene_B_055') / 'frames'
    scores = {}
    names = sorted(meta['train'] + meta['test'], key=meta['times'].__getitem__)
    fps = round((len(names) - 1) / (meta['times'][names[-1]] - meta['times'][names[0]]), 6)
    label = meta.get('label', 'KITTI-360 0000 · 左侧鱼眼') if fish else 'PandaSet ' + scene + ' · 前视相机'
    with TemporaryDirectory(dir=output) as tmp:
        video = Path(tmp) / 'comparison.mp4'
        with subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                               '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1280x830',
                               '-framerate', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'fast',
                               '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
                              stdin=subprocess.PIPE) as enc:
            for name in names:
                images = {k: read(path/name, size) for k, path in methods.items()}
                gt = read(bundle / 'all/images' / name, size)
                if any(v.std() < 1 for v in images.values()) and gt.std() > 5:
                    raise ValueError(f'Blank reconstruction at {scene}/{name}')
                test = name in meta['test']
                if test:
                    scores[name] = {k: metrics(gt, v) for k, v in images.items()}
                if fish:
                    fish_size = meta.get('fisheye_resolution', (512, 512))
                    fish_gt = read(bundle / 'fish_all/images' / name, fish_size)
                    fish_rgb = read(gut / 'fisheye_frames' / name, fish_size)
                    if test:
                        scores[name]['3DGUT_fisheye'] = metrics(fish_gt, fish_rgb)
                    lens = meta.get('fisheye_label', '125° 等距鱼眼')
                    pin = f"{meta['perspective_horizontal_fov']}° 针孔"
                    sc_fish_dir = sc / 'fisheye_frames'
                    if sc_fish_dir.is_dir():
                        sc_fish = read(sc_fish_dir / name, fish_size)
                        if test:
                            scores[name]['SC-GS_fisheye'] = metrics(fish_gt, sc_fish)
                        top = [tile(fish_gt, '实拍 · ' + lens, 426), tile(fish_rgb, '3DGUT · ' + lens, 428),
                               tile(sc_fish, 'SC-GS · ' + lens, 426)]
                    else:
                        pattern = meta.get('raw_pattern', 'kitti360-0000-0200/image_02/{frame:010d}.png')
                        sensor = read(root / 'work/data' / pattern.format(frame=meta['source_frames'][name]),
                                      meta.get('raw_size', (1400, 1400)))
                        top = [tile(sensor, '原始鱼眼实拍', 426), tile(fish_gt, '实拍 · ' + lens, 428),
                               tile(fish_rgb, '3DGUT · ' + lens, 426)]
                    bottom = [tile(gt, '实拍 · ' + pin, 426), tile(images['3DGUT'], '3DGUT · ' + pin, 428),
                              tile(images['SC-GS'], 'SC-GS · ' + pin, 426)]
                else:
                    original = read(host / f'{int(Path(name).stem):03}.jpg', size)
                    if test:
                        scores[name]['StreetCrafter_JPEG'] = metrics(gt, original)
                    top = [tile(gt, '前视实拍'), tile(original, 'StreetCrafter · 7,000 步')]
                    bottom = [tile(images[k], f'{k} · {steps:,} 步') for k in methods]
                frame_id = meta.get('source_frames', {}).get(name, int(Path(name).stem))
                header = bar(f'{label}     /     第 {frame_id:03} 帧     /     ' + ('留出帧' if test else '训练帧'),
                             1280, 38, header=True)
                plate = np.vstack((header, np.hstack(top), np.hstack(bottom)))
                enc.stdin.write(plate.tobytes())
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError('Video encoding failed')
        mean = {key: {m: float(np.mean([row[key][m] for row in scores.values()]))
                      for m in ('psnr_db', 'ssim')} for key in next(iter(scores.values()))}
        report = dict(scene=scene, steps=steps, train=len(meta['train']), test=len(meta['test']),
                      fps=fps, frames=len(names), resolution=size, seed_points=meta['seed_points'],
                      mean=mean, held_out=scores)
        if fish:
            report.update(fisheye_resolution=meta.get('fisheye_resolution', [512,512]),
                          source_camera=meta['source_camera'],
                          scgs_training_views_per_frame=9 if sc.name.startswith('sc-gs-fisheye-') else 1)
        report_file = Path(tmp) / 'metrics.json'
        report_file.write_text(json.dumps(report, indent=2, allow_nan=False))
        video.replace(output / video.name)
        report_file.replace(output / report_file.name)
    print(json.dumps(mean, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('scene')
    parser.add_argument('steps', type=int)
    compare(**vars(parser.parse_args()))
