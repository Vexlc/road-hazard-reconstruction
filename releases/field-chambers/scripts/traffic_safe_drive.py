"""Export only traffic-clear exposures from native PandaSet before/after drives."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream'))
from traffic_clearance import TrafficSurvey, world_bounds


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def archived_inputs(plan):
    registry = read(ROOT/'catalog/roadgs-releases.json')
    identity = digest(plan)
    for release in registry['releases'].values():
        entries = {name:checksum for checksum, name in (line.split('  ', 1) for line in
                   (ROOT/release['checksums']).read_text().splitlines())}
        for name, checksum in entries.items():
            if name.endswith('/scenes/placement_plan.json') and checksum == identity:
                prefix = name[:-len('scenes/placement_plan.json')]
                return {key[len(prefix):]:value for key,value in entries.items() if key.startswith(prefix)}
    return {}


@lru_cache(maxsize=256)
def verify_asset(path, expected):
    if not expected or digest(path) != expected:
        raise ValueError(f'Unknown or changed placement geometry: {path}')


@lru_cache(maxsize=8)
def pandaset(raw):
    lidar = raw / 'lidar'
    camera = raw / 'camera/front_camera'
    times, poses = read(lidar / 'timestamps.json'), read(lidar / 'poses.json')
    shots, cameras = read(camera / 'timestamps.json'), read(camera / 'poses.json')
    if not len(times) == len(poses) == len(shots) == len(cameras) == 80:
        raise ValueError(f'Incomplete PandaSet sequence: {raw}')
    if not np.isfinite(shots).all() or not np.all(np.diff(shots) > 0) or np.max(np.abs(np.subtract(times, shots))) > .1:
        raise ValueError('Camera exposures do not match this LiDAR sequence')
    samples, persistent = [], None
    for index, (time, pose) in enumerate(zip(times, poses)):
        position = pose['position']
        actors = [dict(id='ego', xy_m=[position['x'], position['y']], radius_m=5.5, max_speed_mps=55.)]
        path = raw / 'annotations/cuboids' / f'{index:02d}.pkl.gz'
        complete = False
        fixed = {}
        if path.is_file():
            table = pd.read_pickle(path)
            complete = not table.empty
            columns = ['uuid', 'label', 'position.x', 'position.y', 'dimensions.x', 'dimensions.y', 'yaw', 'stationary']
            if table['uuid'].isna().any() or table['uuid'].duplicated().any():
                raise ValueError(f'Invalid cuboid identities: {path}')
            for key, label, east, north, width, length, yaw, stationary in table[columns].itertuples(index=False, name=None):
                if label in ('Signs', 'Construction Signs', 'Cones', 'Pylons', 'Road Barriers', 'Temporary Construction Barriers'):
                    continue
                if not np.isfinite([width, length]).all() or min(width, length) <= 0:
                    raise ValueError(f'Invalid cuboid dimensions: {path}/{key}')
                actors.append(dict(id=key, xy_m=[east, north], radius_m=np.hypot(width, length)/2,
                                   max_speed_mps=12. if label in ('Pedestrian', 'Pedestrian with Object') else 55.))
                if stationary is True and np.isfinite(yaw):
                    fixed[key] = (east, north, width, length, yaw)
        persistent = fixed if persistent is None else {key:value for key,value in persistent.items() if fixed.get(key) == value}
        samples.append(dict(time_s=time, complete=complete, actors=actors))
    fixed_boxes = {}
    for key, (east, north, width, length, yaw) in persistent.items():
        cosine, sine = abs(np.cos(yaw)), abs(np.sin(yaw))
        half = np.array([cosine*width + sine*length, sine*width + cosine*length])/2
        fixed_boxes[key] = [[east, north] - half, [east, north] + half]
    for sample in samples:
        sample['actors'] = [actor for actor in sample['actors'] if actor['id'] not in fixed_boxes]
    survey = TrafficSurvey(dict(schema='traffic-world-xy-v1', scene=raw.name, samples=samples, fixed_boxes_m=fixed_boxes))
    return survey, shots, cameras, read(camera / 'intrinsics.json')


@lru_cache(maxsize=256)
def native_bounds(path):
    verts = PlyData.read(str(path))['vertex']
    return np.array([[reduce(verts[axis]) for axis in ('x', 'y', 'z')] for reduce in (np.min, np.max)])


def check_native_plan(raw, cases, records, models, margin, start, *, frames_only=False):
    host = pandaset(raw)
    survey, shots, _, _ = host
    reports = {}
    for case in cases:
        if case['site'] != survey.scene or len(case['assets']) != len(records):
            raise ValueError('Traffic survey does not match placement site or slots')
        boxes = {key: world_bounds(anchor, native_bounds(Path(models[key]['ply'])))
                 for key, anchor in zip(case['assets'], records)}
        if frames_only:
            targets = [row['center_world'] for key, row in zip(case['assets'], records)
                       if models[key]['category'] in ('pothole', 'stereo', 'rgbd')]
            report = select_frames(host, boxes, targets or [row['center_world'] for row in records], start, margin)
            if not report['kept_source_frames']:
                raise ValueError(f"{case['id']}: no traffic-clear exposures with the target in view")
            reports[case['id']] = report
            continue
        try:
            reports[case['id']] = survey.require_clear(boxes, margin)
        except ValueError as error:
            raise ValueError(f"{case['id']}: choose another surveyed road site; {error}") from error
        frames = [start, start + 39, *(row['source_reference_frame'] for row in records)]
        if any(not 0 <= frame < len(shots) or not survey.times[0] <= shots[frame] <= survey.times[-1] for frame in frames):
            raise ValueError(f"{case['id']}: requested camera time is outside traffic survey")
    return reports


def in_front(center, camera, lens):
    rotation = Rotation.from_quat([camera['heading'][axis] for axis in 'xyzw'])
    position = [camera['position'][axis] for axis in 'xyz']
    right, down, depth = rotation.inv().apply(np.subtract(center, position))
    return (depth > 0 and 0 <= lens['fx'] * right / depth + lens['cx'] < 1920
            and 0 <= lens['fy'] * down / depth + lens['cy'] < 1080)


def select_frames(host, boxes, targets, start, margin):
    survey, shots, cameras, lens = host
    intervals = survey.evaluate(boxes, margin)
    if not 0 <= start <= len(shots) - 40:
        raise ValueError('Invalid native drive start')
    rows = []
    for frame in range(start, start + 40):
        decision = survey.exposure(intervals, shots[frame])
        visible = any(in_front(center, cameras[frame], lens) for center in targets)
        rows.append(dict(source_frame=frame, camera_time_s=shots[frame], **decision,
                         target_in_front_view=visible, kept=decision['clear'] and visible))
    return dict(scene=survey.scene, margin_m=margin, world_xy_bounds_m={key: box.tolist() for key, box in boxes.items()},
                kept_source_frames=[row['source_frame'] for row in rows if row['kept']], frames=rows)


def frame_hashes(path):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
           '-map', '0:v:0', '-fps_mode', 'passthrough', '-f', 'framemd5', '-']
    lines = subprocess.check_output(cmd, text=True).splitlines()
    return [line.split(',')[-1].strip() for line in lines if line and not line.startswith('#')]


def encode(source, destination, keep):
    hashes = frame_hashes(source)
    if len(hashes) != 40:
        raise ValueError(f'Expected the native 40-exposure drive, got {len(hashes)}: {source}')
    if not keep:
        return
    select = '+'.join(f'eq(n\\,{index})' for index in keep)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-nostdin', '-n', '-threads', '2',
                    '-i', str(source), '-map', '0:v:0', '-vf', f'select={select},setpts=N/(10*TB)', '-r', '10',
                    '-c:v', 'libx264', '-threads', '3', '-preset', 'veryfast', '-qp', '0', '-movflags', '+faststart',
                    str(destination)], check=True)
    if frame_hashes(destination) != [hashes[index] for index in keep]:
        raise ValueError(f'Decoded frames differ from selected source exposures: {source}')


def compilation(stage, report):
    clips = [item['video'] for item in report if item['video']]
    if not clips:
        return
    playlist = stage/'playlist.ffconcat'
    playlist.write_text('ffconcat version 1.0\n' + ''.join(f"file '{path}'\n" for path in clips))
    movie = stage/'safe-drives.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-nostdin', '-n', '-f', 'concat',
                    '-safe', '1', '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart', str(movie)], check=True)
    if frame_hashes(movie) != [value for path in clips for value in frame_hashes(stage/path)]:
        raise ValueError('Compilation does not exactly match the approved clips')


def export_case(folder, dest, models, start, raw_root, margin, archive):
    scene = read(folder / 'scene.json')
    source = folder / 'global_before_after.mp4'
    receipt = read(folder / 'complete.json')['files']
    for path in (source, folder / 'scene.json'):
        if digest(path) != receipt[path.name]:
            raise ValueError(f'Native migration receipt mismatch: {path}')
    if 'drive_source_frames' in scene:
        if scene['drive_source_frames'] != list(range(start, start + 40)):
            raise ValueError('Drive plan and recorded exposure IDs disagree')
    elif any(archive.get(f'scenes/{folder.name}/{path}') != receipt[path] for path in (source.name, 'scene.json')):
        raise ValueError('Unknown source-frame mapping: expected a verified release drive or native drive_source_frames')
    if (scene['id'] != folder.name or folder.name.split('_')[0] != scene['site']
            or [row['asset_id'] for row in scene['placements']] != scene['assets']
            or len(set(scene['assets'])) != len(scene['assets'])):
        raise ValueError(f'Scene/placement identity mismatch: {folder}')
    host = pandaset(raw_root / scene['site'])
    boxes, pits, centers = {}, [], []
    for row in scene['placements']:
        key = row['asset_id']
        if row['size_multiplier'] != 1.:
            raise ValueError(f'Not a native metre-scale release placement: {key}')
        verify_asset(models[key], row.get('asset_ply_sha256') or archive.get(models[key].relative_to(folder.parents[1]).as_posix()))
        boxes[key] = world_bounds(row, native_bounds(models[key]))
        centers.append(row['center_world'])
        if row['category'] == 'pothole':
            pits.append(row['center_world'])
    report = select_frames(host, boxes, pits or centers, start, margin)
    keep = [frame - start for frame in report['kept_source_frames']]
    dest.mkdir()
    encode(source, dest / 'safe.mp4', keep)
    write(dest / 'frames.json', dict(case=scene['id'], source_sha256=receipt[source.name], **report))
    return dict(case=scene['id'], input_frames=len(report['frames']), kept=len(keep),
                video=f'{scene["id"]}/safe.mp4' if keep else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path, help='Native PandaSet package containing scenes/ and assets/')
    parser.add_argument('out', type=Path, help='New directory; existing or historical outputs are never overwritten')
    parser.add_argument('--case', action='append', help='Case ID (repeatable); default: every case')
    parser.add_argument('--raw-root', type=Path, default=ROOT/'tools/multiscene/data/raw', help='Trusted PandaSet raw data')
    parser.add_argument('--margin', type=float, default=.25, help='Extra clearance in metres')
    args = parser.parse_args()
    bundle = args.bundle
    models = {row['id']: bundle/'assets'/row['ply'] for row in read(bundle/'assets/asset_manifest.json')}
    plan_path = bundle/'scenes/placement_plan.json'
    plan = read(plan_path)
    archive = archived_inputs(plan_path)
    folders = {path.parent.name: path.parent for path in (bundle/'scenes').glob('*/scene.json')}
    chosen = sorted(set(args.case or folders))
    if not chosen or not set(chosen) <= folders.keys() or args.out.exists():
        parser.error('Provide existing case(s) and a new destination')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp)/'clean'
        stage.mkdir()
        report = []
        for key in chosen:
            site = key.split('_')[0]
            item = export_case(folders[key], stage/key, models, plan['drive_start_by_site'][site], args.raw_root, args.margin, archive)
            report.append(item)
            print(f'{key}: {item["kept"]}/40', flush=True)
        write(stage/'index.json', report)
        compilation(stage, report)
        stage.rename(args.out)


if __name__ == '__main__':
    main()
