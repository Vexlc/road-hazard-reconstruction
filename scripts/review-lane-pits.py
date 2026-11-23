"""Compare four native assets using the same PandaSet camera frames and crop."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen2hole.large_pits.audit import camera_summary, check_movie, read, review, sha256

KEYS = ('P198', 'P199', 'P200', 'P201')
PANEL = (752, 298)
SIZE = (1552, 838)


def inspect(bundle, folder, key):
    row, = read(bundle / 'assets/asset_manifest.json')
    record = read(folder / 'placement.json')
    checks = read(bundle / 'checks.json')[key]
    assets = (bundle / 'assets' / row['glb']).parent
    for name, digest in checks['sha256'].items():
        if sha256(assets / name) != digest:
            raise ValueError(f'{key}: asset digest mismatch for {name}')
    for kind in ('ply', 'glb'):
        if record[kind + '_sha256'] != checks['sha256']['asset.' + kind]:
            raise ValueError(f'{key}: scene uses a different {kind}')
    checkpoint = bundle / 'scenes' / record['case'] / 'trained_model/iteration_7000.pth'
    if sha256(checkpoint) != record['source_checkpoint_sha256']:
        raise ValueError(f'{key}: different placed scene checkpoint')
    cameras = camera_summary(record)
    if (row['id'] != key or record['asset'] != key or record['size_multiplier'] != 1
            or record['reload_max_pixel_error'] or any(cam['protected_edits'] or cam['outside_edits'] for cam in cameras.values())):
        raise ValueError(f'{key}: failed scene registration or preservation checks')
    dest = folder.parent / 'reference'
    dest.mkdir(exist_ok=True)
    source = review.source_check(bundle, row, dest, context=False, detailed=True)
    fields = ('label', 'reference_kind', 'oriented_extent_m', 'outline_area_m2', 'depth_p95_m', 'scale_kind')
    result = {field: row[field] for field in fields}
    for field in ('design_long_axis_m', 'design_bed_depth_p95_m'):
        if field in row:
            result[field] = row[field]
    result.update(source_check=source, geometry_sha256=checks['geometry_sha256'], native_hashes=checks['sha256'],
                  movie=check_movie(folder / 'drive.mp4', record), cameras=cameras)
    return record, result


def matched_frame_numbers(records, camera):
    available = [{f['frame'] for f in r['frames'][camera]} for r in records.values()]
    numbers = sorted(set.intersection(*available))
    if not numbers or numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise ValueError(f'Missing or discontinuous matched frames in {camera}')
    return numbers


def decode(captures, records, camera, frame):
    pictures, boxes = [], []
    for key, cap in captures.items():
        rows = records[key]['frames'][camera]
        i = next(i for i, row in enumerate(rows) if row['frame'] == frame)
        offset = 0 if camera == 'front_camera' else len(records[key]['frames']['front_camera']) + 33
        if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != offset + i:
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset + i)
        ok, pixels = cap.read()
        if not ok:
            raise ValueError(f'{key}: missing {camera} frame {frame}')
        pictures.append(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
        if rows[i]['bbox']:
            boxes.append(rows[i]['bbox'])
    return pictures, .75 * np.asarray(boxes)


def near_crop(boxes):
    x0, y0 = boxes[:, :2].min(0)
    x1, y1 = boxes[:, 2:].max(0)
    width = min(1440, max((x1-x0)*1.28, (y1-y0)*PANEL[0]/PANEL[1]*1.3, 390))
    height = width*PANEL[1]/PANEL[0]
    left = np.clip((x0+x1-width)/2, 0, 1440-width)
    top = np.clip((y0+y1-height)/2, 70, 810-height)
    return np.array([left, top, left+width, top+height])


def layout(measurements):
    page = Image.new('RGB', SIZE, '#f5f3ed')
    review.label(page, (20, 18), '车道上的四种坑洞', 34, True)
    review.label(page, (20, 62), 'PandaSet 109   /   同一位置、同一机位与裁切', 19)
    for i, (key, data) in enumerate(measurements.items()):
        x, y = 18 + i%2*764, 107 + i//2*347
        kind = '实拍参考' if data['reference_kind'] == 'photograph' else '生成参考'
        review.label(page, (x, y), f"{key}   {data['label']}", 23, True)
        review.label(page, (x+510, y+3), f"{kind}   ·   {data['outline_area_m2']:.2f} m²", 19)
    review.label(page, (18, 808), 'P198 参考照片 © Jennifer Petrie / Geograph · CC BY-SA 2.0      P199–P201 参考图由 ImageGen 生成', 16)
    return page


def compose(template, pictures, crop, caption):
    page = template.copy()
    box = tuple(np.rint(crop).astype(int))
    for i, picture in enumerate(pictures):
        tile = picture.crop(box).resize(PANEL, Image.Resampling.LANCZOS)
        page.paste(tile, (18+i%2*764, 143+i//2*347))
    review.label(page, (1120, 34), caption, 22, True)
    return page


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('runs/lane-pits-batch'))
    p.add_argument('--out', type=Path, default=Path('outputs/photo-large-potholes/lane-batch'))
    args = p.parse_args()
    records, measurements, movies = {}, {}, {}
    for key in KEYS:
        folder = args.out / key / 'scene-109'
        records[key], measurements[key] = inspect(args.runs/key/'series', folder, key)
        movies[key] = folder/'drive.mp4'
    for field in ('centre_world', 'basis_columns', 'site'):
        if any(record[field] != records[KEYS[0]][field] for record in records.values()):
            raise ValueError(f'Mismatched scene {field}')
    if len({m['geometry_sha256'] for m in measurements.values()}) != len(KEYS):
        raise ValueError('Repeated geometry')
    template = layout(measurements)
    captures = {key: cv2.VideoCapture(str(path)) for key, path in movies.items()}
    frames_written = 0
    stage = args.out/'comparison.stage.mp4'
    full = np.array([0, 810-1440*PANEL[1]/PANEL[0], 1440, 810])
    paths = {}
    try:
        with imageio.get_writer(str(stage), fps=10, codec='libx264', quality=8, macro_block_size=2,
                                output_params=['-threads', '3', '-movflags', '+faststart']) as writer:
            for camera, title in [('front_camera', '前视'), ('back_camera', '后视')]:
                numbers = matched_frame_numbers(records, camera)
                paths[camera] = numbers
                for number in numbers:
                    pictures, bounds = decode(captures, records, camera, number)
                    near = number == (numbers[-1] if camera == 'front_camera' else numbers[0])
                    if near and len(bounds) != len(KEYS):
                        raise ValueError('A comparison subject is occluded in the close-up')
                    close = near_crop(bounds) if near else full
                    # Keep the approach wide, pause for inspection, then pull back on departure.
                    zooms = [0]
                    if near:
                        if camera == 'front_camera':
                            zooms += list(np.linspace(.05, 1, 12)) + [1]*15
                        else:
                            zooms = [1]*12 + list(np.linspace(1, 0, 12))
                    for zoom in zooms:
                        page = compose(template, pictures, full*(1-zoom)+close*zoom, f'{title}   /   第 {number:02d} 帧')
                        writer.append_data(np.asarray(page))
                        frames_written += 1
                    if near and camera == 'front_camera':
                        page.save(args.out/'overview.jpg', quality=96, subsampling=0)
    finally:
        for cap in captures.values():
            cap.release()
    cap = cv2.VideoCapture(str(stage))
    count = 0
    while cap.grab():
        ok, frame = cap.retrieve()
        if not ok or tuple(frame.shape[1::-1]) != SIZE:
            raise ValueError('Unreadable comparison frame')
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if count != frames_written or fps != 10:
        raise ValueError('Incomplete comparison movie')
    stage.replace(args.out/'comparison.mp4')
    report = dict(site='109', shared_placement=True, size_multiplier=1, checkpoint_reload_max_pixel_error=0,
                  matched_camera_frames=paths, movie=dict(frames=count, fps=fps, size=SIZE), assets=measurements)
    (args.out/'measurements.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
