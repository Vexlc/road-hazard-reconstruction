"""Audit through-openings, solid topology, measured edges, rigid poses and matched road instances."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from cover_observations import compare_cases

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream/sensors"))
from road_geometry import Surface, glb_triangles, read_json, sha256 as file_hash
from cover_fragments import check_fragments
from cover_grade import check_grade, check_source_relief


def require(ok, why):
    if not ok:
        raise ValueError(why)


def topology(parts):
    v = []
    f = []
    start = 0
    for label, points, faces in parts:
        v.append(points)
        f.append(faces + start)
        start += len(points)
    vertices, inv = np.unique(np.vstack(v), axis=0, return_inverse=True)
    faces = inv[np.vstack(f)]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs, idx, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(idx, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    require((counts == 2).all() and not balance.any(), "Unpaired or reversed edges in the cover")
    graph = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(vertices), len(vertices)))
    components = connected_components(graph, directed=False, return_labels=False)
    require(components == 1, "Cover has unsupported detached pieces")
    chi = len(vertices) - len(pairs) + len(faces)
    triangles = vertices[faces].astype('f8')
    volume = float(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    )
    require(volume > 0 and chi % 2 == 0, "Invalid oriented metal volume")
    return dict(
        through_holes=1 - chi // 2,
        euler_characteristic=chi,
        connected_solids=components,
        closed_oriented_mesh=True,
        metal_volume_m3=volume,
    )


def surface_hits(scene, origins, direction):
    return scene.cast(origins, np.broadcast_to(direction, origins.shape))[:2]


def photo_check(bundle, profile, trace, meta, lid, dest):
    photo = np.array(Image.open(bundle / "references/frame.png").convert("RGB"))
    mask = np.array(Image.open(bundle / "references" / profile["mask"]).convert("L")) > 127
    c = np.float32(profile["corners_xy"])
    # Invert the original four photograph coordinates independently of the exported UV map.
    uv_transform = cv2.getPerspectiveTransform(c, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    h, w = mask.shape
    y, x = np.indices((h, w), dtype='f4')
    uv = cv2.perspectiveTransform(np.stack([x, y], -1), uv_transform)
    pos = (uv - 0.5) * [meta['panel_size_m'][0], -meta['panel_size_m'][1]]
    inside = (abs(uv - 0.5) < 0.499).all(2)
    if profile['shape'] == 'round':
        inside &= np.linalg.norm(uv - 0.5, axis=2) < 0.499
    if meta.get('break_polygon_uv'):
        roi = cv2.perspectiveTransform(
            np.float32(meta['break_polygon_uv'])[None], np.linalg.inv(uv_transform)
        )[0]
        edited = np.uint8(mask)
        cv2.fillPoly(edited, [np.rint(roi).astype('i4')], 1)
        mask = edited > 0
    pixels = np.column_stack([pos[inside], np.ones(inside.sum())])
    pose = np.array(meta['lid_transform'])
    origin = pixels @ pose[:3, :3].T + pose[:3, 3]
    distance, _ = surface_hits(lid, origin, -pose[:3, 2])
    openings = np.zeros(mask.shape, bool)
    openings[inside] = ~np.isfinite(distance)
    mask &= inside
    union = (openings | mask).sum()
    iou = (openings & mask).sum() / union
    holes, n = cv2.connectedComponents(mask.astype('u1'))[:2]
    recall = [float(openings[n == i].mean()) for i in range(1, holes)]
    require(iou > 0.93 and min(recall) > 0.85, "A labelled opening was lost on the exported cover")
    overlay = photo.copy()
    for binary, color in [(mask, (255, 212, 56)), (openings, (38, 217, 205))]:
        outlines, _ = cv2.findContours(binary.astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, outlines, -1, color, 1, cv2.LINE_AA)
    if meta.get('break_polygon_uv'):
        cv2.polylines(overlay, [np.rint(roi).astype('i4')], True, (255, 100, 82), 2, cv2.LINE_AA)
    lo, hi = (
        np.maximum(0, c.min(0).astype(int) - 20),
        np.minimum([w, h], np.ceil(c.max(0)).astype(int) + 21),
    )
    Image.fromarray(overlay[lo[1] : hi[1], lo[0] : hi[0]]).save(dest / 'openings.jpg', quality=96)
    return dict(
        source_holes=holes - 1,
        opening_iou=float(iou),
        minimum_opening_recall=min(recall),
        tested_panel_pixels=int(inside.sum()),
        source_open_pixels=int(mask.sum()),
        model_open_pixels=int(openings.sum()),
        design_break_included=bool(meta.get('break_polygon_uv')),
    )


def cross_section(scene, meta, angle, dest):
    # The principal ray is orthogonal to the image plane. Camera and frame are identical for all states.
    direction = np.array([0.52, -0.854, 0.0]) * np.sin(np.deg2rad(angle)) + [
        0,
        0,
        np.cos(np.deg2rad(angle)),
    ]
    direction /= np.linalg.norm(direction)
    right = np.array([0.854, 0.52, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    x = np.linspace(-0.70, 0.70, 680) * max(meta['panel_size_m'])
    y = np.linspace(0.68, -0.68, 580) * max(meta['panel_size_m'])
    xx, yy = np.meshgrid(x, y)
    center = np.array([0, 0, 0.06])
    rays = (center + xx[:, :, None] * right + yy[:, :, None] * up + 2 * direction).reshape(-1, 3)
    _, ids = surface_hits(scene, rays, -direction)
    ids = ids.reshape(yy.shape)
    # Fill rays that enter the finite mesh from below its outer road boundary.
    at_road = rays - rays[:, 2, None] / direction[2] * direction
    outside = (
        ((at_road[:, :2] < scene.low[:2]) | (at_road[:, :2] > scene.high[:2])).any(1).reshape(ids.shape)
    )
    ids[np.isin(ids, [6, 7]) & outside] = 1
    palette = np.uint8(
        [
            [240, 239, 232],
            [208, 210, 199],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [122, 126, 104],
            [64, 123, 134],
            [40, 75, 98],
            [187, 171, 145],
            [225, 169, 86],
            [146, 95, 79],
            [168, 113, 58],
            [102, 108, 85],
        ]
    )
    Image.fromarray(palette[ids]).save(dest / f'rays-{angle}.png')
    # Track which rays aimed *at the road-plane opening* can see into the shaft.
    w, l = meta['width_m'], meta['length_m']
    sx, sy = np.meshgrid(np.linspace(-0.498, 0.498, 233) * w, np.linspace(-0.498, 0.498, 233) * l)
    inside = (
        (sx / (0.5 * w)) ** 2 + (sy / (0.5 * l)) ** 2 < 1
        if meta['exponent'] == 2
        else np.ones(sx.shape, bool)
    )
    target = np.c_[sx[inside], sy[inside], np.zeros(inside.sum())] + 2 * direction
    _, labels = surface_hits(scene, target, -direction)
    return dict(
        angle_from_normal_deg=angle,
        rays=len(labels),
        visible_well_fraction=float(np.isin(labels, [6, 7]).mean()),
        face_fraction=float((labels == 9).mean()),
        aperture_or_outer_edge_fraction=float((labels == 11).mean()),
    )


def measured_side(bundle, profile, trace, meta, lid_parts, lid, dest):
    camera = trace['source_camera']
    k = np.array(camera['camera_K'])
    pose = np.array(camera['camera_from_cover'])
    w, l = meta['panel_size_m']
    local = np.array([[-w / 2, l / 2], [w / 2, l / 2], [w / 2, -l / 2], [-w / 2, -l / 2]])
    lower = next(v for label, v, _ in lid_parts if label == 10)
    ndx = profile['edge_calibration']['check_edge_corners']
    vertices = lower[[np.argmin(np.linalg.norm(lower[:, :2] - local[i], axis=1)) for i in ndx]]
    homogeneous = (vertices @ pose[:3, :3].T + pose[:3, 3]) @ k.T
    line = homogeneous[:, :2] / homogeneous[:, 2:]
    marked = np.array(profile['edge_calibration']['check_lower_edge_xy'])
    error = np.abs(np.cross(line[1] - line[0], marked - line[0])) / np.linalg.norm(line[1] - line[0])
    fraction = (marked - line[0]) @ (line[1] - line[0]) / np.sum((line[1] - line[0]) ** 2)
    require(((fraction >= 0) & (fraction <= 1)).all(), "Validation marks outside lower edge")
    photo = np.array(Image.open(bundle / "references/frame.png").convert("RGB"))
    h, w = photo.shape[:2]
    yy, xx = np.indices((h, w))
    directions = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)], 1) @ np.linalg.inv(k).T @ pose[:3, :3]
    eye = -pose[:3, 3] @ pose[:3, :3]
    _, ids, _ = lid.cast(np.broadcast_to(eye, directions.shape), directions)
    material = ids.reshape(h, w)
    cv2.line(photo, *np.rint(line).astype(int), (38, 217, 205), 2, cv2.LINE_AA)
    for xy in marked:
        cv2.circle(photo, tuple(np.rint(xy).astype(int)), 3, (255, 214, 52), 1, cv2.LINE_AA)
    fit = profile['edge_calibration']['bottom_corner_xy']
    cv2.drawMarker(photo, tuple(fit), (242, 147, 81), cv2.MARKER_CROSS, 9, 2)
    points = np.vstack([line, profile['corners_xy']])
    lo = np.maximum(0, points.min(0).astype(int) - 18)
    hi = np.ceil(points.max(0)).astype(int) + 19
    Image.fromarray(photo[lo[1] : hi[1], lo[0] : hi[0]]).save(dest / "side-fit.jpg", quality=96)
    projection = photo.copy()
    for index, rgb in [(9, [240, 186, 101]), (11, [96, 172, 190]), (10, [155, 126, 99])]:
        region = material == index
        projection[region] = np.rint(0.45 * photo[region] + 0.55 * np.asarray(rgb))
    Image.fromarray(projection[lo[1] : hi[1], lo[0] : hi[0]]).save(dest / "source-pose.jpg", quality=96)
    return dict(
        reference_width_m=camera['reference_width_m'],
        length_m=l,
        thickness_m=float(-vertices[:, 2].mean()),
        estimated_focal_px=float(k[0, 0]),
        fit_corner_error_px=camera['lower_corner_fit_error_px'],
        reserved_edge_pixels=len(error),
        edge_distances_px=error.tolist(),
        median_px=float(np.median(error)),
        max_px=float(error.max()),
        optical_camera_visible_face_pixels=int((material == 9).sum()),
        optical_camera_visible_wall_pixels=int((material == 11).sum()),
    )


def broken_web(intact, changed, meta, polygon, dest):
    # Trace identical vertical rays on both exported solids in the cover's local coordinates.
    n = 960
    y, x = np.indices((n, n))
    uv = (np.stack([x, y], -1) + 0.5) / n
    xy = (uv - 0.5) * [meta['panel_size_m'][0], -meta['panel_size_m'][1]]
    origins = np.c_[xy.reshape(-1, 2), np.ones(n * n)]
    pose = np.asarray(meta['lid_transform'])
    cast_from = origins @ pose[:3, :3].T + pose[:3, 3]
    metal = []
    for surface, points, direction in ((intact, origins, [0, 0, -1]), (changed, cast_from, -pose[:3, 2])):
        distance, _ = surface_hits(surface, points, direction)
        metal.append(np.isfinite(distance).reshape(n, n))
    removed = metal[0] & ~metal[1]
    region = np.zeros((n, n), 'u1')
    outline = np.rint(np.asarray(polygon) * n - 0.5).astype('i4')
    cv2.fillPoly(region, [outline], 1)
    # Vectorizing the adjoining marked openings can shift their edges by a pixel.
    guard = cv2.dilate(region, np.ones((9, 9), 'u1')).astype(bool)
    changed_inside = int((removed & guard).sum())
    changed_outside = int((removed & ~guard).sum())
    require(
        changed_inside > 0 and changed_inside / removed.sum() > 0.95,
        "Break is not localized at the marked web",
    )
    lo = np.maximum(0, outline.min(0) - [65, 28])
    hi = np.minimum([n, n], outline.max(0) + [65, 28])
    crop = np.s_[lo[1] : hi[1], lo[0] : hi[0]]
    tiles = []
    width = round(
        280 * (hi[0] - lo[0]) * meta['panel_size_m'][0] / ((hi[1] - lo[1]) * meta['panel_size_m'][1])
    )
    for mask in metal:
        tile = np.where(mask[crop][..., None], np.uint8([225, 169, 86]), np.uint8([40, 75, 98]))
        tiles.append(Image.fromarray(tile).resize((width, 280), Image.Resampling.NEAREST))
    joined = Image.new('RGB', (2 * width + 6, 280), '#f2f1eb')
    joined.paste(tiles[0])
    joined.paste(tiles[1], (width + 6, 0))
    joined.save(dest / 'broken-web.png')
    return dict(
        removed_metal_area_m2=float(removed.mean() * np.prod(meta['panel_size_m'])),
        removed_rays=int(removed.sum()),
        outside_guard_rays=changed_outside,
        grid=n,
        guard_pixels=4,
    )


def check(run, with_sensors):
    bundle = run / 'series'
    spec = read_json(bundle / 'calibration/survey.json')
    profile = spec['perforations']
    trace = read_json(bundle / 'calibration/grate.json')
    checks = read_json(bundle / 'checks.json')
    results = {}
    stock = {}
    intact = {}
    report = dict(family=spec['family'], source=spec['source_url'], states=results)
    baseline = None
    baseline_id = None
    changes = []
    reference_ring = None
    for state in spec['states']:
        key = state['id']
        absent = state.get('absent', False)
        folder = bundle / 'assets' / ('missing' if absent else 'raised') / key
        glb = folder / 'asset.glb'
        meta = read_json(folder / 'metadata.json')
        dest = run / 'evaluation' / key
        dest.mkdir(parents=True, exist_ok=True)
        require(
            checks[key]['sha256']['asset.glb'] == file_hash(glb)
            and checks[key]['sha256']['asset.ply'] == file_hash(folder / 'asset.ply'),
            key + ' asset identity',
        )
        all_parts = list(glb_triangles(glb))
        lids = []
        known = set()
        for part, vertices, faces in all_parts:
            if part in (9, 10, 11):
                lids.append((part, vertices, faces))
            else:
                sig = sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
                if 'installation_grade' not in state or part == 7:
                    require(stock.setdefault(part, sig) == sig, key + ' changed fixed shaft')
                known.add(part)
        fixed = known - {8} if 'installation_grade' in state else known
        require(fixed == {1, 5, 6, 7} and (not lids) == absent, key + ' unexpected cover or shaft')
        result = dict(
            glb_sha256=checks[key]['sha256']['asset.glb'], ply_sha256=checks[key]['sha256']['asset.ply']
        )
        if 'installation_grade' in state:
            if reference_ring is None:
                neutral = next((s for s in spec['states'] if not s.get('absent')
                                and s.get('installation_grade', {}).get('center_z_m') == 0
                                and s['installation_grade']['tilt_deg'] == 0), None)
                require(neutral is not None, 'A grade comparison requires a flush seated reference')
                reference_glb = bundle/'assets'/'raised'/neutral['id']/'asset.glb'
                report['grade_reference'] = neutral['id']
                reference_ring = next((v, f) for part, v, f in glb_triangles(reference_glb) if part == 5)
            result['grade'] = check_grade(glb, all_parts, folder/'surface.npz', reference_ring, profile, state)
            if key == report['grade_reference'] and 'displacement' in profile:
                report['grade_source_relief'] = check_source_relief(bundle, glb, profile, trace, dest)
        if state.get('fragment'):
            require(lids and not absent and meta['fragment_location'] == state['fragment']['location'],
                    key + ' missing fractured cover')
        elif lids:
            solid = topology(lids)
            require(solid['through_holes'] == meta['through_holes'], key + ' topology vs annotations')
            result['solid'] = solid
            lid = Surface(glb, exclude=(1, 5, 6, 7, 8))
            result['source_footprint'] = photo_check(bundle, profile, trace, meta, lid, dest)
            if any(meta['lid_offset_m']):
                fixed = Surface(glb, exclude=(6, 7, 9, 10, 11))
                bottom = next(v for part, v, _ in lids if part == 10)
                vertices = np.unique(bottom, axis=0).astype(float)
                above = vertices.copy()
                above[:, 2] = max(fixed.high[2], vertices[:, 2].max()) + 1
                distance, _ = surface_hits(fixed, above, [0, 0, -1])
                hit = np.isfinite(distance)
                gap = vertices[hit, 2] - above[hit, 2] + distance[hit]
                require(
                    len(gap) and abs(gap.min()) < 0.0002, key + ' cover loses contact with road or frame'
                )
                result['support'] = dict(
                    underside_vertices=len(vertices),
                    over_road_or_frame=len(gap),
                    contacts_within_0_2mm=int((abs(gap) < 0.0002).sum()),
                    minimum_signed_gap_m=float(gap.min()),
                )
            require(
                solid['through_holes'] == result['source_footprint']['source_holes'],
                key + ' wrong opening count',
            )
            if not state.get('break_polygon_uv'):
                transform = np.asarray(meta['lid_transform'])
                residual = 0.0
                for part, vertices, faces in lids:
                    xyz = (vertices - transform[:3, 3]) @ transform[:3, :3]
                    ref, reference_faces = intact.setdefault(part, (xyz, faces))
                    require(np.array_equal(reference_faces, faces), key + ' rigid topology changed')
                    residual = max(residual, float(np.abs(xyz - ref).max()))
                require(residual < 2e-6, key + ' cover shape changed with pose')
                result['rigid_pose_max_residual_m'] = residual
            if (
                trace.get('source_camera')
                and state.get('angle_deg') == 0
                and not state.get('break_polygon_uv')
                and not any(state['offset_m'])
            ):
                report['same_photo_side_check'] = measured_side(
                    bundle, profile, trace, meta, lids, lid, dest
                )
            if state.get('break_polygon_uv'):
                changes.append((key, lid, meta, dest))
            elif np.allclose(meta['lid_transform'], np.eye(4)):
                baseline = lid
                baseline_id = key
        else:
            require(
                meta['through_holes'] == 0 and meta['reference_through_holes'] == len(trace['openings_uv']),
                key + ' missing lid metadata',
            )
        full = Surface(glb)
        if (dest/'cutaway.json').is_file():
            inspection=read_json(dest/'cutaway.json')
            require(inspection['source_glb_sha256']==result['glb_sha256']
                    and inspection['cutaway_glb_sha256']==file_hash(dest/'cutaway.glb')
                    and inspection['preserved_parts'], key+' stale inspection copy')
            result['sectioned_model']=inspection
        result['views'] = [cross_section(full, meta, a, dest) for a in (0, 55)]
        results[key] = result
    if baseline is not None:
        for key, lid, meta, dest in changes:
            results[key]['break_comparison'] = broken_web(
                baseline, lid, meta, meta['break_polygon_uv'], dest
            )
    if 'fracture_path_uv' in profile:
        require(baseline_id is not None, 'A fracture comparison needs its intact reference')
        check_fragments(run, baseline_id, results, topology)
    report['fixed_geometry_sha256_by_part'] = stock
    if with_sensors:
        sensors = run / 'sensors/checks'
        laser = read_json(sensors / 'scan_audit.json')
        cameras = read_json(sensors / 'camera_audit.json')
        geoms = read_json(run / 'sensors/geometry/inventory.json')
        anchors = {}
        placed = {}
        for path in sorted((bundle / 'scenes').glob('*/scene.json')):
            scene = read_json(path)
            (placement,) = scene['placements']
            key = placement['asset_id']
            site = scene['site']
            require(
                scene['checkpoint_reloaded']
                and placement['size_multiplier'] == 1
                and placement['reload_max_pixel_error'] == 0
                and placement['asset_ply_sha256'] == results[key]['ply_sha256'],
                key + ' native road instance',
            )
            require(geoms[key]['mesh_sha256'] == results[key]['glb_sha256'], key + ' sensor mesh')
            anchor = [
                placement[f] for f in ('center_world', 'basis_columns', 'source_reference_frame', 'slot')
            ]
            require(anchors.setdefault(site, anchor) == anchor, site + ' unmatched placement')
            case = read_json(run / 'sensors/cases' / (scene['id'] + '.json'))
            require(case['scene_sha256'] == file_hash(path), scene['id'] + ' ray frame identity')
            require(
                case['delta_sha256'] == file_hash(run / 'sensors/cases' / (scene['id'] + '.npz')),
                scene['id'] + ' stored beam edits',
            )
            (one,) = case['assets']
            require(
                one['asset_id'] == key and min(one['by_sensor']) > 0,
                scene['id'] + ' sensor instance or lidar coverage',
            )
            placed[scene['id']] = dict(
                asset=key,
                site=site,
                frames=case['frames'],
                returns_by_lidar=one['by_sensor'],
                part_counts=one['part_counts'],
            )
        require(set(a['asset'] for a in placed.values()) == set(results), "Missing transferred state")
        comparisons = {}
        camera_folder = 'matched_camera' if 'fracture_path_uv' in profile or reference_ring is not None else 'camera'
        for base, base_info in placed.items():
            if base_info['asset'] != baseline_id:
                continue
            for case, other in placed.items():
                if case != base and other['site'] == base_info['site']:
                    comparisons[case] = compare_cases(run, other['site'], base, case, camera_folder=camera_folder)
        if reference_ring is not None:
            # Isolate removal from grade: keep the very same sunk ring in both sensor replays.
            grade_by_id = {s['id']: s for s in spec['states']}
            removal = {}
            for case, entry in placed.items():
                missing = grade_by_id[entry['asset']]
                if not missing.get('absent'):
                    continue
                for base, other in placed.items():
                    seated = grade_by_id[other['asset']]
                    if (other['site'] == entry['site'] and not seated.get('absent') and
                            seated['installation_grade'] == missing['installation_grade']):
                        pair = compare_cases(run, entry['site'], base, case, image_suffix='same-grade-removal',
                                             camera_folder=camera_folder)
                        removal[case] = pair
            report['removal_at_same_grade'] = removal
        if 'fracture_path_uv' in profile:
            lost = {s['id'] for s in spec['states'] if s.get('fragment', {}).get('location') == 'missing'}
            relocated = {s['id'] for s in spec['states'] if s.get('fragment', {}).get('location') in ('floor','road')}
            report['relative_to_missing_fragment'] = {
                case: compare_cases(run, item['site'], base, case, image_suffix='fragment-location', camera_folder=camera_folder)
                for base, missing in placed.items() if missing['asset'] in lost
                for case, item in placed.items() if item['site'] == missing['site'] and item['asset'] in relocated
            }
        report.update(
            placements=placed,
            relative_to_seated=comparisons,
            same_road_anchors=anchors,
            scan_audit=laser,
            camera_crops=cameras['camera_counts'],
            independent_geometry=read_json(sensors / 'independent_geometry.json'),
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-sensors", action="store_true")
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(check(args.run, args.with_sensors), indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    )
