"""Publish a full calibrated driver view with a detail of the same single-asset road instance."""

import json

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
