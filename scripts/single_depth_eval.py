"""Verify reserved native range samples against a one-frame mesh and its labelled surfaces."""

import numpy as np
from PIL import Image

from fuse_pit import fuse
from pit_captures import read_views
from road_geometry import glb_triangles, read_json, sha256, Surface
from single_depth_surfaces import split_depth, surface_regions, annotated_fragments, paver_labels, target_view


def error_stats(error):
    values = np.abs(error) * 1000
    return dict(
        samples=len(error), median_mm=float(np.median(values)), p95_mm=float(np.percentile(values, 95))
    )


def comparison_meshes(glb, dest, regions):
    def parts(path):
        return {
            name: (xyz, faces, uv, bytes(image))
            for name, xyz, faces, uv, image in glb_triangles(path, names=True, appearance=True)
        }

    base = parts(glb)
    results = {}
    for mode in ('labels', 'isolated'):
        path = dest / (mode + '.glb')
        current = parts(path)
        expected = set(regions) if mode == 'isolated' else set(base)
        if set(current) != expected:
            raise ValueError('Unexpected labelled mesh nodes in ' + str(path))
        for name, (xyz, faces, uv, image) in current.items():
            ref = base[name]
            if not all(np.array_equal(a, b) for a, b in zip((xyz, faces, uv), ref[:3])):
                raise ValueError('Region comparison changed triangles or UVs in ' + name)
            if mode == 'isolated' and image != ref[3]:
                raise ValueError('Isolated surface has a different base color')
        results[mode] = dict(sha256=sha256(path), triangles=sum(len(v[1]) for v in current.values()))
    return results


def evaluate(args, spec):
    key = spec['id']
    bundle = args.run / 'series'
    cal = bundle / 'calibration'
    capture = read_json(cal / (key + '_capture.json'))
    views = read_views(spec, args.source, args.catalog, args.files_base or cal)
    (name,) = spec['fit']
    view = target_view(spec.get('surface_regions', {}), views[name])
    if view.sources != capture['inputs'][name]['files'] or spec['holdout']:
        raise ValueError('Wrong single-camera source')
    train, test = split_depth(view)
    field, _, frame = fuse([train], {name: np.eye(4)})
    with np.load(cal / (key + '.npz')) as saved:
        for axis in ('height', 'defect_mask', 'view_count'):
            if not np.array_equal(saved[axis], field[axis]):
                raise ValueError('Reserved ranges affect the published fit: ' + axis)
        names = []
        if 'surface_regions' in spec:
            labels, names = surface_regions(spec['surface_regions'], train, field, frame)
            if (
                not np.array_equal(labels, saved['region_ids'])
                or names != capture['geometry']['region_names']
            ):
                raise ValueError('Semantic regions differ from their source')
    glb = bundle / 'assets/pothole' / key / 'asset.glb'
    meshes = comparison_meshes(glb, bundle / 'comparison' / key, names) if names else {}
    model = Surface(glb)
    basis = np.array(frame['local_axes_in_anchor_camera'])
    eye = -np.array(frame['local_origin_in_anchor_m']) @ basis
    y, x = np.where(test & view.mask)
    if len(x) != capture['reserved_pit_depth_pixels'] or train.depth[y, x].any():
        raise ValueError('Incorrect held-out rays')
    ray = view.xyz(np.c_[x, y], np.ones(len(x))) @ basis
    truth = view.depth[y, x]
    actual, part, _ = model.cast(np.broadcast_to(eye, ray.shape), ray)
    valid = np.isfinite(actual)
    flat = -eye[2] / ray[:, 2]
    if valid.mean() < 0.95:
        raise ValueError('Mesh misses too many reserved pit depth samples')
    whole = error_stats(actual[valid] - truth[valid])
    baseline = error_stats(flat[valid] - truth[valid])
    result = {}
    regions = np.zeros_like(view.mask, 'u1')
    reference = (
        annotated_fragments(spec['surface_regions'], view)
        if 'fragment_polygons_px' in spec.get('surface_regions', {})
        else None
    )
    if 'paver' in spec.get('surface_regions', {}):
        reference = paver_labels(spec['surface_regions'], view)
    for i, n in enumerate(names, 1):
        surface = Surface(glb, include_names={n})
        d, _, _ = surface.cast(np.broadcast_to(eye, ray.shape), ray)
        hit = np.isfinite(d) & valid & (abs(d - actual) < 0.0001)
        if hit.sum() < 5:
            raise ValueError('Too few reserved observations for ' + n)
        regions[y[hit], x[hit]] = i
        z = eye[2] + ray[hit, 2] * truth[hit]
        rec = dict(
            **error_stats(actual[hit] - truth[hit]), observed_median_elevation_mm=float(np.median(z) * 1000)
        )
        if reference is not None:
            target = reference[y, x] == i
            rec['source_annotation_iou_on_reserved_rays'] = float((hit & target).sum() / (hit | target).sum())
        result[n] = rec
    folder = args.run / 'evaluation' / key
    folder.mkdir(parents=True, exist_ok=True)
    shape = view.mask.shape
    core = view.rgb.copy()
    core[~view.mask] = (0.4 * core[~view.mask] + 65).astype('u1')
    core[y, x] = (239, 177, 61)
    Image.fromarray(core).save(folder / 'reserved.jpg', quality=94)
    yr, xr = np.where(view.mask)
    all_ray = view.xyz(np.c_[xr, yr], np.ones(len(xr))) @ basis
    dist, parts, _ = model.cast(np.broadcast_to(eye, all_ray.shape), all_ray)
    cls = np.zeros(shape, 'u1')
    cls[yr, xr] = parts
    pic = view.rgb.copy()
    for code, hue in [(13, (29, 209, 182)), (14, (235, 153, 61)), (18, (29, 209, 182)), (19, (249, 179, 57))]:
        mask = cls == code
        pic[mask] = (0.4 * pic[mask] + 0.6 * np.array(hue)).astype('u1')
    Image.fromarray(pic).save(folder / 'labels.jpg', quality=95)
    zmap = np.full(shape, np.nan, 'f4')
    zmap[yr, xr] = dist
    valid_test = test & view.mask & np.isfinite(zmap)
    np.savez_compressed(
        folder / 'heldout.npz',
        test=valid_test,
        region=regions,
        part=cls,
        optical_m=zmap,
        measured_m=np.where(valid_test, view.depth, np.nan),
        K=view.K,
        eye=eye,
        axes=basis,
    )
    return dict(
        glb_sha256=sha256(glb),
        source=spec['source'],
        fit=[name],
        holdout_type='same_capture_depth_blocks',
        reserved_pit_pixels=len(truth),
        meshed_fraction=float(valid.mean()),
        fitted_valid_pit_pixels=int((train.mask & (train.depth > 0)).sum()),
        withheld_mesh=whole,
        withheld_road_plane=baseline,
        region_results=result,
        comparison_meshes=meshes,
        fit_rebuilt_without_reserved_ranges=True,
    )
