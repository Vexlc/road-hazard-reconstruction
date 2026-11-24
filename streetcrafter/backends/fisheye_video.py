"""Export a fisheye-only driving reel, including a displaced 3DGUT camera."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np

from compare import bar, read, tile


def make_video(root, scenes, steps):
    code = Path(__file__).resolve().parent
    official = Path(os.environ.get('OFFICIAL_ROOT', root.parent.parent/'01_official'))/'3dgrut'
    python = Path(os.environ.get('GUT_PYTHON', official/'.venv/bin/python'))
    env = dict(os.environ, PYTHONPATH=str(official), PATH=f'{python.parent}:{os.environ["PATH"]}')
    env.setdefault('CUDA_HOME', '/usr/local/cuda')
    env.setdefault('MAX_JOBS', '2')
    out = root/'outputs/streetcrafter-backends'
    out.mkdir(parents=True, exist_ok=True)
    fps = 30
    with TemporaryDirectory(dir=out) as tmp:
        target = Path(tmp)/'fisheye-roads.mp4'
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
               '-pixel_format', 'bgr24', '-video_size', '1200x976', '-framerate', str(fps), '-i', '-',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '17',
               '-movflags', '+faststart', str(target)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            for scene in scenes:
                bundle = root/'work/streetcrafter-backends'/scene
                meta = json.loads((bundle/'all/split.json').read_text())
                runs = root/'runs/streetcrafter-backends'/scene
                with (runs/'.backend.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    gut = runs/f'3dgut-{steps}'
                    sc = runs/f'sc-gs-fisheye-{steps}'
                    checkpoint, = (gut/'fit').glob('*/ckpt_last.pt')
                    shifted = gut/'fisheye_right_035m'
                    subprocess.run([str(python), str(code/'render_gut.py'), str(checkpoint),
                                    str(bundle/'fish_all'), str(shifted), '--offset-right', '.35'],
                                   cwd=official, env=env, check=True)
                    size = meta.get('fisheye_resolution', (512, 512))
                    label = meta.get('label', 'KITTI-360 0000 · 左侧鱼眼')
                    lens = meta.get('fisheye_label', '125° 等距鱼眼')
                    times = meta['times']
                    names = sorted(times, key=times.__getitem__)
                    clock = np.array([times[n] for n in names])
                    ticks = np.rint((np.r_[clock, clock[-1]+np.median(np.diff(clock))]-clock[0])*fps).astype(int)
                    for i, name in enumerate(names):
                        header = bar(f"{label}   /   {lens}   /   第 {meta['source_frames'][name]} 帧   /   " +
                                     ('留出帧' if name in meta['test'] else '训练帧'), 1200, 40, header=True)
                        panels = [tile(read(folder/name, size), title, 600, 432) for folder, title in [
                            (bundle/'fish_all/images', '鱼眼实拍'), (gut/'fisheye_frames', '3DGUT · 实拍机位'),
                            (sc/'fisheye_frames', 'SC-GS · 实拍机位'), (shifted, '3DGUT · 相机向右平移 35 厘米')]]
                        plate = np.vstack((header, np.hstack(panels[:2]), np.hstack(panels[2:])))
                        pixels = plate.tobytes()
                        for _ in range(ticks[i+1]-ticks[i]):
                            encoder.stdin.write(pixels)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Failed to encode the fisheye reel')
        target.replace(out/target.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    make_video(Path(__file__).resolve().parents[2], args.scenes, args.steps)
