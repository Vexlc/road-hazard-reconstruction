"""Encode a camera-path sequence and two labeled held-out comparisons into one MP4."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate import rgb


WIDTH, HEIGHT, FPS = 1280, 864, 24
INK, MUTED, ACCENT = '#eef3f4', '#a4b6be', '#79dfce'
CHAPTERS = ('同轨迹漫游', '完整鱼眼对照', '共同针孔视场对照')
SCENES = {'dozer': '街边挖掘机', 'redwoods2': '红杉林'}
LABELS = {'GT': '实拍参考', '3dgut': '3DGUT', '3dgut-7000': '3DGUT', 'sc-gs': 'SC-GS', 'self-cali-gs': 'Self-Cali-GS'}


@lru_cache
def font(size):
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size, index=2)


def label(draw, position, message, size, color=INK, anchor='lm'):
    draw.text(position, message, font=font(size), fill=color, anchor=anchor)


def plate(scene, chapter):
    page = Image.new('RGB', (WIDTH, HEIGHT), '#101b23')
    draw = ImageDraw.Draw(page)
    label(draw, (28, 25), '鱼眼场景重建   /   ' + scene.upper(), 17, ACCENT)
    label(draw, (27, 69), SCENES.get(scene, scene), 37)
    label(draw, (WIDTH - 28, 66), f'0{chapter + 1}  /  03     {CHAPTERS[chapter]}', 25, anchor='rm')
    draw.line((28, 106, WIDTH - 28, 106), fill='#38515b', width=2)
    return page


def tile(page, rect, image, title, tag):
    left, top, width, height = rect
    draw = ImageDraw.Draw(page)
    label(draw, (left, top + 20), title, 23)
    label(draw, (left + width, top + 20), tag, 17, MUTED, 'rm')
    pixels = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    scale = min(width / pixels.width, (height - 45) / pixels.height)
    pixels = pixels.resize((round(pixels.width * scale), round(pixels.height * scale)), Image.Resampling.LANCZOS)
    page.paste(pixels, (left + (width - pixels.width) // 2, top + 44 + (height - 44 - pixels.height) // 2))


def path_diagram(poses):
    points = poses[:, :3, 3]
    centered = points - points.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ basis[:2].T
    coords *= min(300 / max(np.ptp(coords[:, 0]), 1e-7), 102 / max(np.ptp(coords[:, 1]), 1e-7))
    coords -= coords.min(axis=0)
    coords += [WIDTH - 36 - coords[:, 0].max(), 683]
    return [tuple(map(int, point)) for point in coords]


def frame_sequence(scene, runs, cache):
    split = json.loads((scene / 'split.json').read_text())
    report = json.loads((runs / 'evaluation.json').read_text())
    size = split['resolution']
    focal_x, focal_y, center_x, center_y, *coeffs = split['params_fisheye']
    matrix = np.array([[focal_x, 0, center_x - 0.5], [0, focal_y, center_y - 0.5], [0, 0, 1.]])
    maps = cv2.fisheye.initUndistortRectifyMap(matrix, np.array(coeffs), np.eye(3), matrix, tuple(size), cv2.CV_32FC1)
    path = path_diagram(np.load(cache / 'camera-path.npy'))
    base = plate(scene.name, 0)
    draw = ImageDraw.Draw(base)
    label(draw, (28, 670), '相机沿采集路线移动', 25)
    label(draw, (28, 720), '两侧使用相同位姿与针孔视场。左侧由鱼眼渲染重新投影。', 19, MUTED)
    label(draw, (28, 757), '中间机位从三维模型逐帧渲染，不是实拍或照片补帧。', 19, MUTED)
    draw.line(path, fill='#445864', width=3)
    label(draw, (WIDTH - 28, 814), '相机位置轨迹 · 平面投影', 16, MUTED, 'rm')
    for index, (pos_x, pos_y) in enumerate(path):
        page = base.copy()
        native = rgb(cache / f'gut/{index:04d}.png', size)
        rectified = cv2.remap(native, *maps, cv2.INTER_LINEAR)
        tile(page, (28, 145, 602, 458), rectified, '3DGUT', '7,000 步')
        tile(page, (650, 145, 602, 458), rgb(cache / f'sc/{index:04d}.png', size), 'SC-GS', '1,500 步')
        draw = ImageDraw.Draw(page)
        if index:
            draw.line(path[:index + 1], fill=ACCENT, width=3)
        draw.ellipse((pos_x - 5, pos_y - 5, pos_x + 5, pos_y + 5), fill=INK)
        draw.rectangle((28, 836, 28 + round(1224 * (index + 1) / len(path)), 838), fill=ACCENT)
        yield page, 1

    render_dirs = {}
    for method in ('3dgut', '3dgut-7000'):
        step = report['steps'][method]
        render_dirs[method], = (runs / method).glob(f'fit/*/ours_{step}/renders')
    if [row['source'] for row in report['frames']] != [split['sources'][name] for name in split['test']]:
        raise ValueError('Evaluation frames differ from the dataset split')
    for chapter, domain in [(1, 'full-fisheye'), (2, 'shared-perspective')]:
        base = plate(scene.name, chapter)
        for index, name in enumerate(split['test']):
            scores = report['frames'][index][domain]
            fish = {key: rgb(folder / f'{index:05d}.png', size) for key, folder in render_dirs.items()}
            lens = runs / 'self-cali-gs/training_val_1500/renderred' / f'distorted_{Path(name).stem}_test.png'
            fish['self-cali-gs'] = rgb(lens, size)
            if chapter == 1:
                images = {'GT': rgb(scene / 'fish/images' / name, size), **fish}
            else:
                images = {'GT': rgb(scene / 'images' / name, size),
                          '3dgut': cv2.remap(fish['3dgut'], *maps, cv2.INTER_LINEAR),
                          'sc-gs': rgb(runs / 'sc-gs/fit_node/heldout_1500' / name, size),
                          'self-cali-gs': cv2.remap(fish['self-cali-gs'], *maps, cv2.INTER_LINEAR)}
            page = base.copy()
            for panel, (key, pixels) in enumerate(images.items()):
                tag = split['sources'][name] if key == 'GT' else f"{report['steps'][key]:,} 步  ·  {scores[key]:.2f} dB"
                tile(page, (28 + 622 * (panel % 2), 114 + 342 * (panel // 2), 602, 338), pixels, LABELS[key], tag)
            draw = ImageDraw.Draw(page)
            label(draw, (28, 827), f"留出视角  {index + 1:02d} / {len(split['test']):02d}", 20, ACCENT)
            label(draw, (WIDTH - 28, 827), '同机位实拍对照  /  单帧 PSNR，越高越好', 18, MUTED, 'rm')
            yield page, 30


def encode(scene, runs, cache, output):
    executable = shutil.which('ffmpeg')
    if not executable:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / 'video.mp4'
        cmd = [executable, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(staged)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            for frame, repeats in frame_sequence(scene, runs, cache):
                data = frame.tobytes()
                for _ in range(repeats):
                    encoder.stdin.write(data)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Video encoder failed')
        staged.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    encode(args.scene, args.runs, args.cache, args.output)
