"""Independently measure water storage and dry-bed identity from the delivered GLBs."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_propagation, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import Surface, glb_triangles, read_json, save_json, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def depths(solid, points):
    start = points.copy()
    start[:, 2] += 0.3
    direction = np.broadcast_to([0.0, 0.0, -1.0], start.shape)
    distance, _, _ = solid.cast(start, direction)
    require(np.isfinite(distance).all(), 'A water triangle extends beyond the bed')
    return distance.astype('f8') - 0.3


def compare_beds(dry, wet):
    original = dict((name, rest) for name, *rest in glb_triangles(dry, names=True, appearance=True))
    actual = dict((name, rest) for name, *rest in glb_triangles(wet, names=True, appearance=True))
    water, faces, _, _ = actual.pop('Water surface')
    require(set(original) == set(actual), 'Different dry parts')
    for name, attrs in actual.items():
        for a, b in zip(original[name], attrs):
            require(np.array_equal(a, b), name + ' differs in geometry, indices, UVs or JPEG')
    return water.astype('f8'), faces


def spill_path(field, up, water):
    h = field['height']
    # Propagate *inward* from every boundary edge, testing the measured spill
    # elevation on either side of a ten-micron perturbation.
    y, x = np.indices(h.shape)
    w = float(field['width'])
    z = up[0] * w * (x / (h.shape[1] - 1) - 0.5) + up[1] * w * (0.5 - y / (len(h) - 1)) + up[2] * h
    perimeter = np.zeros(h.shape, bool)
    perimeter[[0, -1], :] = True
    perimeter[:, [0, -1]] = True
    seed = tuple(water['basin_seed_rc'])
    exits = []
    for delta in (-0.00001, 0.00001):
        open_cells = z < water['spill_elevation_m'] + delta
        reached = binary_propagation(
            perimeter & open_cells, [[0, 1, 1], [1, 1, 1], [1, 1, 0]], mask=open_cells
        )
        exits.append(bool(reached[seed]))
    require(exits == [False, True], 'Spill level does not bracket an open path to the patch boundary')


def check_one(bundle, base, key):
    folder = bundle / 'assets/pothole' / key
    meta = read_json(folder / 'metadata.json')
    water = meta['water']
    bed = meta['dry_source_id']
    dry = base / 'assets/pothole' / bed
    mesh = folder / 'asset.glb'
    require(sha256(dry / 'asset.glb') == meta['dry_glb_sha256'], 'Dry mesh replaced')
    xyz, faces = compare_beds(dry / 'asset.glb', mesh)
    solid = Surface(dry / 'asset.glb')
    up = np.asarray(water['up_local'])
    tri = xyz[faces]
    normal_area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * 0.5
    projected = normal_area[:, 2]
    require((projected > 0).all(), 'Folded or degenerate water triangle')
    flatness = float(np.abs(xyz @ up - water['free_surface_m']).max())
    require(flatness < 1e-6, 'Non-horizontal free surface')
    d = depths(solid, tri.mean(axis=1))
    require(d.min() > -1e-6, 'Water surface passes under bed')
    volume = float(projected @ d * 1000)
    require(
        abs(volume - water['retained_liters']) < max(0.00001, volume * 1e-5),
        'Volume differs from independent GLB rays',
    )
    require(
        abs(water['input_liters'] - volume - water['spilled_liters']) < 0.0001, 'Unbalanced liquid volume'
    )
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    require(counts.max() == 2, 'Nonmanifold shoreline')
    graph = coo_matrix((np.ones(len(edges), bool), edges.T), shape=(len(xyz), len(xyz))).tocsr()
    require(connected_components(graph, directed=False, return_labels=False) == 1, 'Detached water patches')
    shoreline = np.unique(edges[counts == 1])
    edge_error = float(np.abs(depths(solid, xyz[shoreline])).max())
    require(edge_error < 1e-5, 'Shoreline separates from bed')
    with np.load(dry / 'surface.npz') as original, np.load(folder / 'surface.npz') as field:
        for name in ['height', 'defect_mask', 'alpha', 'width', 'length']:
            require(np.array_equal(original[name], field[name]), 'Dry field changed: ' + name)
        spill_path(field, up, water)
        mask = field['water_mask'] > 0.5
        require(np.array_equal(mask, field['water_depth_local_z'] > 0), 'Crossed dry shoreline')
        components, _ = label(field['defect_mask'] > 0.5)
        labels, counts = np.unique(components[components > 0], return_counts=True)
        component_hits = [
            int(np.count_nonzero(mask & (components == i))) for i in labels[np.argsort(-counts)]
        ]
        water_observed = float(
            (field['rgb_supported' if 'rgb_supported' in field else 'view_count'][mask] > 0).mean()
        )
    source_eval = read_json(base.parent / 'evaluation/measurements.json')[bed]
    require(source_eval['glb_sha256'] == meta['dry_glb_sha256'], 'Different bed in source-camera evaluation')
    if 'heldout' in source_eval:
        obs = {k: (v['exported_glb'], 'heldout RGB-D frame') for k, v in source_eval['heldout'].items()}
    else:
        obs = {k: (v, 'depth withheld from RGB fit') for k, v in source_eval['frames'].items()}
    measurements = {
        k: dict(
            role=role,
            valid_pit_depth_pixels=v['visible_pit_depth_samples'],
            depth_abs_median_mm=v['depth_abs_median_mm'],
        )
        for k, (v, role) in obs.items()
    }
    return dict(
        glb_sha256=sha256(mesh),
        dry_source=bed,
        dry_glb_sha256=meta['dry_glb_sha256'],
        dry_source_depth_checks=measurements,
        wet_vertices_by_pit_largest_first=component_hits,
        dry_part_geometry_uv_texture_identical=True,
        surface_triangles=len(faces),
        shoreline_vertices=len(shoreline),
        max_shoreline_gap_m=edge_error,
        max_surface_level_error_m=flatness,
        integrated_liters=volume,
        input_liters=water['input_liters'],
        retained_liters=water['retained_liters'],
        spilled_liters=water['spilled_liters'],
        capacity_liters=water['capacity_liters'],
        depth_max_mm=water['max_water_depth_m'] * 1000,
        surface_area_m2=water['surface_area_m2'],
        wet_vertices_direct_source_support_fraction=water_observed,
        site=water['site'],
        road_grade_percent=float(100 * np.linalg.norm(up[:2]) / up[2]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=ROOT / 'runs')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    bundle = args.run / 'series'
    spec = read_json(bundle / 'calibration/survey.json')
    result = {}
    for state in spec['states']:
        key = state['id']
        base = args.runs / spec['beds'][state['bed']]['run'] / 'series'
        row = check_one(bundle, base, key)
        result[key] = row
        print(
            f'{key}  retained {row["integrated_liters"]:.4f} L  overflow {row["spilled_liters"]:.4f} L',
            flush=True,
        )
    save_json(args.out, result)


if __name__ == '__main__':
    main()
