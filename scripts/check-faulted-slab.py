"""Independently measure the delivered slab, its scan registration and its support in each state."""

import argparse
from io import BytesIO
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, read_json, save_json, sha256, Surface  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLOCK = {'Detached pavement top', 'Detached pavement edges', 'Detached pavement underside'}


def require(value, message):
    if not value:
        raise ValueError(message)


def cast(scene, xy):
    origins = np.column_stack([xy, np.full(len(xy), 0.3)]).astype('f4')
    directions = np.zeros_like(origins)
    directions[:, 2] = -1
    distance, part, _ = scene.cast(origins, directions)
    require(np.isfinite(distance).all(), 'Vertical rays missed a delivered surface')
    return 0.3 - distance, part


def closed_solid(parts):
    triangles = np.concatenate([p[0][p[1]].astype('f8') for name, p in parts.items() if name in BLOCK])
    verts, ids = np.unique(triangles.reshape(-1, 3), return_inverse=True, axis=0)
    faces = ids.reshape(-1, 3)
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    reverse = np.unique(unique[:, ::-1], axis=0)
    require(
        np.all(counts == 1)
        and np.array_equal(unique, reverse)
        and len(verts) - len(unique) // 2 + len(faces) == 2,
        'Slab is not a closed, consistently oriented solid',
    )
    signed = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6
    volume = float(signed.sum())
    require(volume > 0, 'Inverted slab')
    centroid = (triangles.sum(axis=1) * signed[:, None]).sum(axis=0) / (4 * volume)
    return dict(
        vertices=len(verts), triangles=len(faces), volume_m3=volume, uniform_mass_center_m=centroid.tolist()
    )


def check(run, inputs, parent, survey):
    bundle = run / 'series'
    specs = read_json(survey)
    parent_id = specs['parent']
    parent_mask = np.asarray(Image.open(parent / 'calibration' / (parent_id + '_outline.png'))) > 0
    cx, cy, n = specs['crop_col_row_intervals']
    outline = parent_mask[cy : cy + n + 1, cx : cx + n + 1]
    sx, sy = specs['seed_col_row_in_crop']
    labs, _ = ndimage.label(~outline)
    island = ndimage.binary_fill_holes(labs == labs[sy, sx])
    require(not island[[0, -1], :].any() and not island[:, [0, -1]].any(), 'Open source boundary')
    records = {}
    reference = None
    for state in sorted(specs['states'], key=lambda s: s['offset_m'] != 0):
        key = state['id']
        offset = state['offset_m']
        folder = bundle / 'assets/pothole' / key
        meta = read_json(folder / 'metadata.json')
        slab = meta['slab']
        mesh = folder / 'asset.glb'
        parts = {
            name: (v, f, uv, bytes(color))
            for name, v, f, uv, color in glb_triangles(mesh, names=True, appearance=True)
        }
        pitch = meta['source_pitch_m']
        step = meta['step']
        span = n * pitch
        raw = {}
        for role in ('Diffuse', 'Displacement'):
            spec = meta['source_files'][role]
            file = inputs / meta['material'] / spec['url'].split('/')[-1]
            require(sha256(file) == spec['sha256'], 'Original scan hash mismatch')
            raw[role] = file
        x0, y0 = meta['origin_col_row']
        crop = np.s_[y0 : y0 + n + 1, x0 : x0 + n + 1]
        source = np.asarray(Image.open(raw['Diffuse']).convert('RGB'))[crop]
        image = Image.open(BytesIO(parts['Road adjoining the fracture'][3]))
        require(np.array_equal(np.asarray(image.convert('RGB')), source), 'Upper surface colour changed')
        heights = cv2.imread(str(raw['Displacement']), cv2.IMREAD_UNCHANGED)
        if heights.ndim == 3:
            heights = heights[:, :, 0]
        require(heights.dtype == np.uint16, '16-bit scan required')
        rr, cc = np.indices(outline.shape)
        mat = meta['material_scale']
        a, b, c = meta['road_plane_pixel']
        true_z = (heights[crop].astype(float) / 65535.0 - mat['midlevel']) * mat['scale_m'] - (
            a * (cc + x0) + b * (rr + y0) + c
        )
        scene = Surface(mesh)
        xy = np.c_[((cc.ravel() - n / 2) * pitch), ((n / 2 - rr.ravel()) * pitch)]
        z, hit = cast(scene, xy[(rr.ravel() > 0) & (cc.ravel() > 0) & (rr.ravel() < n) & (cc.ravel() < n)])
        z = z.reshape(n - 1, n - 1)
        hit = hit.reshape(z.shape)
        rows, cols = rr[1:-1, 1:-1], cc[1:-1, 1:-1]
        # Avoid ambiguous intersections on grid-aligned vertical sides.
        inside_cell = (rows % step != 0) & (cols % step != 0)
        truth_mask = island[1:-1, 1:-1]
        if offset is not None:
            predicted = hit == 13
            meet = (predicted & truth_mask & inside_cell).sum()
            union = ((predicted | truth_mask) & inside_cell).sum()
            iou = float(meet / union)
            require(iou > 0.97, 'Block footprint diverged from optical boundary')
            truth = true_z[1:-1, 1:-1][predicted & truth_mask & inside_cell] + offset
            residual = abs(z[predicted & truth_mask & inside_cell] - truth) * 1000
            solid = closed_solid(parts)
            require(abs(solid['volume_m3'] - slab['volume_m3']) < 1e-8, 'Independent volume mismatch')
            solid.update(
                source_iou=iou,
                heldout_texels=len(residual),
                abs_error_mm_median=float(np.median(residual)),
                abs_error_mm_p95=float(np.percentile(residual, 95)),
            )
            require(solid['abs_error_mm_median'] < 0.5, 'Block top disagrees with withheld scan texels')
            v, faces, uv, color = parts['Detached pavement top']
            require(
                color == parts['Road adjoining the fracture'][3]
                and np.max(abs(uv - np.c_[0.5 + v[:, 0] / span, 0.5 - v[:, 1] / span])) < 1e-6,
                'Top texture no longer registered',
            )
        else:
            require(
                not BLOCK.intersection(parts) and 'Subgrade bearing' not in parts,
                'Missing state still contains slab or bearing',
            )
            core = ndimage.binary_erosion(truth_mask, iterations=2 * step) & inside_cell
            require(
                (hit[core] == 16).all() and np.max(abs(z[core] - specs['base_z_m'])) < 0.00001,
                'Missing slab exposes incorrect base',
            )
            solid = dict(exposed_base_texels=int(core.sum()), exposed_base_z_m=float(np.median(z[core])))
        if reference is None:
            require(offset == 0, 'Family needs an in-place reference')
            reference = (parts, z, hit)
        previous, baseline, baseline_hit = reference
        for name in {'Road adjoining the fracture', 'Fracture lip and exposed bed', 'Slab socket wall'}:
            require(
                all(np.array_equal(p, q) for p, q in zip(parts[name], previous[name])),
                'Fixed pavement changed: ' + name,
            )
        if offset is not None:
            for name in BLOCK:
                v, f, uv, color = parts[name]
                v0, f0, uv0, color0 = previous[name]
                require(
                    np.array_equal(f, f0)
                    and color == color0
                    and np.max(abs(uv - uv0)) < 2e-6
                    and np.max(abs(v - v0 - [0, 0, offset])) < 2e-7,
                    'Slab shape, texture or texture coordinates changed: ' + name,
                )
            selected = (baseline_hit == 13) & (hit == 13) & inside_cell
            measured = z[selected] - baseline[selected]
            require(np.max(abs(measured - offset)) < 1e-5, 'Elevation difference differs from rigid offset')
            solid.update(
                matched_top_texels=int(selected.sum()),
                vertical_difference_mm=float(np.median(measured) * 1000),
            )
            # Intersect each underside face centre with the subgrade after removing the entire plate.
            bottom, bottom_faces, *_ = parts['Detached pavement underside']
            underside = bottom[bottom_faces].astype('f8')
            centres = underside.mean(axis=1)
            ground, material = cast(Surface(mesh, exclude=(13, 14, 15)), centres[:, :2])
            gap = centres[:, 2] - ground
            require((material == 16).all() and (gap >= -0.00001).all(), 'Plate penetrates the subgrade')
            contact = abs(gap) < 0.00001
            require(contact.any() and not contact.all(), 'Expected both an inset bearing and a free edge')
            area = 0.5 * np.abs(
                np.cross(underside[:, 1] - underside[:, 0], underside[:, 2] - underside[:, 0])[:, 2]
            )
            hull = ConvexHull(centres[contact, :2])
            centre_xy = np.array(solid['uniform_mass_center_m'][:2])
            margin = float(-(hull.equations[:, :2] @ centre_xy + hull.equations[:, 2]).max())
            require(margin > 0, 'Uniform-density centre of mass projects outside the support polygon')
            solid.update(
                support_top_m=float(np.median(ground[contact])),
                underside_faces_checked=len(gap),
                supported_faces=int(contact.sum()),
                contact_area_m2=float(area[contact].sum()),
                support_margin_m=margin,
                max_penetration_m=max(0.0, -float(gap.min())),
            )
        records[key] = dict(
            glb_sha256=sha256(mesh),
            ply_sha256=sha256(folder / 'asset.ply'),
            source_sha256=sha256(raw['Diffuse']),
            source_displacement_sha256=sha256(raw['Displacement']),
            source_area_m2=float(island.sum() * pitch**2),
            source_xy_extent_m=slab['grid_xy_extent_m'],
            designed_offset_m=offset,
            **solid,
        )
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--repeat', type=Path, help='Independently prepared and rebuilt run for bytewise comparison')
    p.add_argument('--inputs', type=Path, default=ROOT / 'work/reconstruction-inputs/scanned-pavement')
    p.add_argument('--survey', type=Path, default=ROOT / 'gen2hole/examples/faulted-slab/survey.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = check(args.run, args.inputs, args.parent, args.survey)
    if args.repeat:
        require(args.repeat.resolve() != args.run.resolve(), 'Choose an independently rebuilt run')
        for key, record in report.items():
            rel = Path('series/assets/pothole') / key
            hashes = {}
            for name in ('asset.glb', 'asset.ply', 'splat_material.npz', 'surface.npz', 'metadata.json'):
                original, repeated = args.run / rel / name, args.repeat / rel / name
                require(not original.samefile(repeated), 'Rebuild shares a source file: ' + str(rel / name))
                hashes[name] = sha256(repeated)
                require(hashes[name] == sha256(original), 'Rebuild differs: ' + str(rel / name))
            record['independent_rebuild_sha256'] = hashes
    save_json(args.out, report)
    print({k: v.get('vertical_difference_mm', v.get('exposed_base_z_m')) for k, v in report.items()})
