"""Publish a full calibrated driver view with a detail of the same single-asset road instance."""

import hashlib
import json
import shutil

import numpy as np

from PIL import Image, ImageDraw, ImageOps


def read(p):
    return json.loads(p.read_text())


def driving_frame(image, bounds):
    """Keep the driver's full field of view and enlarge the same marked patch."""
    w, h = image.size
    left, top, right, bottom = bounds
    half_width = max(28, int((right - left) * 0.63), int((bottom - top) * 1.40))
    half_height = max(14, round(half_width / 2.2))
    cx = min(max((left + right) // 2, half_width), w - half_width)
    cy = min(max((top + bottom) // 2, half_height), h - half_height)
    region = (cx - half_width, cy - half_height, cx + half_width, cy + half_height)
    detail = ImageOps.fit(image.crop(region), (308, 140), method=Image.Resampling.LANCZOS)
    full = image.resize((1280, 720), Image.Resampling.LANCZOS)
    border = '#ffe181'
    ImageDraw.Draw(full).rectangle(
        tuple(round(v * 1280 / w) if i % 2 == 0 else round(v * 720 / h) for i, v in enumerate(region)),
        outline=border,
        width=2,
    )
    full.paste(ImageOps.expand(detail, border=2, fill=border), (950, 18))
    return full


def camera_frames(run, processed):
    folder = run / 'evaluation/streets'
    folder.mkdir(parents=True, exist_ok=True)
    frames = {}
    for path in sorted((run / 'series/sensor_views').glob('*/frames.json')):
        sequence = read(path)
        case = sequence['case']
        scene = read(run / 'series/scenes' / case / 'scene.json')
        (placement,) = scene['placements']
        site, category = scene['site'], placement['category']
        channel = next(int(code) - 1 for code, name in sequence['mask_values'].items() if name == category)
        if sequence['assets'] != [placement['asset_id']]:
            raise ValueError('Camera and placed asset disagree in ' + case)
        example = run / 'series/migration' / case / 'sensor_examples' / case
        best = read(example / (category + '.json'))
        frame, box = best['frame'], best['bbox']
        row = next(f for f in sequence['frames'] if f['frame'] == frame)
        if row['class_bboxes_xyxy'][channel] != box or row['class_pixels'][channel] != best['count']:
            raise ValueError('Mismatched source frame or asset bounds in ' + case)
        with Image.open(example / (category + '_after.jpg')) as after:
            if list(after.size) != sequence['size']:
                raise ValueError('Mismatched full driving frame in ' + case)
            driving_frame(after, box).save(folder / (case + '-after.jpg'), quality=94)
        with Image.open(processed / site / 'images' / f'{frame:03d}_0.jpg') as before:
            if list(before.size) != sequence['size']:
                raise ValueError('Mismatched camera image size in ' + case)
            driving_frame(before, box).save(folder / (case + '-before.jpg'), quality=94)
        frames[case] = dict(site=site, frame=frame, bbox=box, visible_pixels=row['class_pixels'][channel])
    (folder / 'frames.json').write_text(json.dumps(frames, indent=2) + '\n')
    return frames


def full_driving_frames(run, cases, raw):
    directory = run/'evaluation'/'camera'
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, case in cases:
        folder = run/'sensors/matched_camera'/case
        records.setdefault(key, [])
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
