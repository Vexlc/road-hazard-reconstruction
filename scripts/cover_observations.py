"""Compare two exported states on the same original laser rows and calibrated image pixels."""

import json

import numpy as np
from PIL import Image, ImageDraw


def laser_rows(path):
    with np.load(path, allow_pickle=False) as data:
        frames = np.repeat(np.arange(len(data['frame_offsets']) - 1), np.diff(data['frame_offsets']))
        rows = {
            (int(f), int(r)): (int(s), float(b), float(a), int(p), xyz)
            for f, r, s, b, a, p, xyz in zip(
                frames,
                data['row'],
                data['sensor'],
                data['range_before_m'],
                data['range_after_m'],
                data['part'],
                data['xyz'],
            )
        }
        if len(rows) != len(frames):
            raise ValueError('Duplicate original beam in ' + str(path))
        return rows


def laser_difference(reference, alternative):
    keys = reference.keys() | alternative.keys()
    total = 0
    no_return_switch = 0
    by_sensor = [0, 0]
    frames = set()
    changed_parts = 0
    deltas = []
    paired_hits = []
    for key in keys:
        sample = reference.get(key) or alternative[key]
        sensor, before = sample[:2]
        original = (sensor, before, before, 0, None)
        a = reference.get(key, original)
        b = alternative.get(key, original)
        if a[:2] != b[:2]:
            raise ValueError('Raw beam mismatch ' + str(key))
        changed_parts += int(a[3] != b[3])
        if np.isfinite(a[2]) != np.isfinite(b[2]):
            no_return_switch += 1
        elif np.isfinite(a[2]):
            distance = abs(a[2] - b[2])
            deltas.append(distance)
            if distance > 0.01:
                total += 1
                by_sensor[sensor] += 1
                frames.add(key[0])
                if a[4] is not None and b[4] is not None:
                    paired_hits.append((distance, key, sensor, a, b))
    strongest = [
        dict(
            frame=key[0],
            row=key[1],
            sensor=sensor,
            delta_m=distance,
            reference_world=a[4].tolist(),
            alternative_world=b[4].tolist(),
            reference_part=a[3],
            alternative_part=b[3],
        )
        for distance, key, sensor, a, b in sorted(paired_hits, key=lambda hit: (-hit[0], hit[1]))[:2]
    ]
    return dict(
        compared_edited_rows=len(keys),
        range_change_over_1cm=total,
        by_sensor_over_1cm=by_sensor,
        frames_over_1cm=len(frames),
        max_range_change_m=max(deltas, default=0),
        hit_status_changes=no_return_switch,
        annotated_part_changes=changed_parts,
        strongest_paired_hits=strongest,
    )


def draw_break_hits(run, site, placement, meta, laser):
    hits = laser['strongest_paired_hits']
    if not hits or not meta.get('break_polygon_uv') or not np.allclose(meta['lid_transform'], np.eye(4)):
        return
    size = np.array(meta['panel_size_m'])
    density = 1600
    color = run / 'series/references/lid_ortho.png'
    with Image.open(color) as photo:
        top = photo.convert('RGB').resize(
            tuple(np.rint(size * density).astype(int)), Image.Resampling.LANCZOS
        )
    draw = ImageDraw.Draw(top)

    def point(world):
        xy = ((np.array(world) - placement['center_world']) @ placement['basis_columns'])[:2]
        return tuple(((xy / [size[0], -size[1]] + 0.5) * top.size).round().astype(int))

    polygon = [tuple(np.rint(np.array(p) * top.size).astype(int)) for p in meta['break_polygon_uv']]
    draw.line(polygon + [polygon[0]], fill='#f3d97f', width=2)
    points = list(polygon)
    for hit in hits:
        a, b = map(point, [hit['reference_world'], hit['alternative_world']])
        draw.line((a, b), fill='white', width=3)
        for (x, y), fill in [(a, '#28e5d3'), (b, '#ff824a')]:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill, outline='black')
        points.extend([a, b])
    lo = np.maximum(0, np.min(points, axis=0) - [85, 35])
    hi = np.minimum(top.size, np.max(points, axis=0) + [85, 35])
    path = run / 'evaluation' / meta['id'] / (site + '-lidar-break.png')
    top.crop((*lo, *hi)).save(path)
    laser['plan_view'] = path.name


def matched_camera(ref, alt, folder, dest):
    for field in ('site', 'camera', 'frame', 'camera_timestamp', 'K_full', 'camera_to_world'):
        if ref[field] != alt[field]:
            raise ValueError('Unmatched camera ' + field)
    boxes = np.array([ref['crop_xyxy'], alt['crop_xyxy']])
    lo = boxes[:, :2].max(0)
    hi = boxes[:, 2:].min(0)
    if (hi <= lo).any():
        raise ValueError('Disjoint calibrated crops')
    parts = []
    depth = []
    photos = []
    protected = []
    originals = []
    for record, box in zip([ref, alt], boxes):
        x, y = lo - box[:2]
        w, h = hi - lo
        roi = np.s_[y : y + h, x : x + w]
        directory = folder / record['case']
        with np.load(directory / record['geometry']) as maps:
            parts.append(maps['part'][roi])
            depth.append(maps['depth_z_m'][roi])
            protected.append(maps['protected'][roi] != 0)
        for images, name in [(photos, 'after_stem'), (originals, 'before_stem')]:
            with Image.open(directory / record[name]) as im:
                images.append(np.array(im.convert('RGB'))[roi])
    if not np.array_equal(*originals):
        raise ValueError('Camera comparison does not share the original pixels')
    valid = ~(protected[0] | protected[1])
    common = valid & (parts[0] != 0) & (parts[1] != 0)
    support = valid & ((parts[0] != 0) | (parts[1] != 0))
    changed_part = valid & (parts[0] != parts[1])
    changed_depth = common & (abs(depth[0] - depth[1]) > 0.01)
    signal = changed_part | changed_depth
    ys, xs = np.where(support)
    if not len(xs):
        raise ValueError('No visible surface in matched crops')
    low = np.maximum(0, [xs.min() - 6, ys.min() - 6])
    high = np.minimum(hi - lo, [xs.max() + 7, ys.max() + 7])
    overlay = photos[1].copy()
    overlay[signal] = np.rint(0.30 * overlay[signal] + 0.70 * np.array([255, 206, 54]))
    tiles = []
    for image in (photos[0], overlay):
        tile = Image.fromarray(image).crop((*low, *high))
        tiles.append(tile.resize((tile.width * 3, tile.height * 3), Image.Resampling.NEAREST))
    width, height = tiles[0].size
    pair = Image.new("RGB", (2 * width + 6, height), '#f2f1eb')
    pair.paste(tiles[0])
    pair.paste(tiles[1], (width + 6, 0))
    pair.save(dest)
    return dict(
        frame=ref['frame'],
        crop_intersection_xyxy=[*lo.tolist(), *hi.tolist()],
        visible_asset_union_pixels=int(support.sum()),
        part_changed_pixels=int(changed_part.sum()),
        shared_surface_depth_change_over_1cm_pixels=int(changed_depth.sum()),
        changed_pixels=int(signal.sum()),
        image=dest.name,
    )


def compare_cases(run, site, base, other):
    sensors = run / 'sensors'
    source = sensors / 'camera'
    rows = [json.loads((source / case / 'views.json').read_text()) for case in (base, other)]
    reference = {v['camera']: v for v in rows[0] if v['available']}
    camera = {}
    for shot in rows[1]:
        name = shot['camera']
        ref = reference.get(name)
        if not shot['available'] or not ref or shot['frame'] != ref['frame']:
            continue
        dest = run / 'evaluation' / shot['asset_id'] / (site + '-' + name + '-difference.png')
        camera[name] = matched_camera(ref, shot, source, dest)
    amounts = [laser_rows(sensors / 'cases' / (case + '.npz')) for case in (base, other)]
    laser = laser_difference(*amounts)
    (placement,) = json.loads((run / 'series/scenes' / other / 'scene.json').read_text())['placements']
    meta = json.loads(
        (
            run / 'series/assets' / placement['category'] / placement['asset_id'] / 'metadata.json'
        ).read_text()
    )
    draw_break_hits(run, site, placement, meta, laser)
    return dict(reference_case=base, alternative_case=other, cameras=camera, lidar=laser)
