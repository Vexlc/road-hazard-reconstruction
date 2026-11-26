"""Conservative, world-XY traffic clearance for placements and recorded exposures."""

from itertools import product

import numpy as np


def world_bounds(anchor, bounds, world_transform=None):
    bounds = np.asarray(bounds, dtype=float)
    basis = np.asarray(anchor['basis_columns'], dtype=float)
    center = np.asarray(anchor['center_world'], dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0])
            or center.shape != (3,) or not np.isfinite(center).all() or basis.shape != (3, 3)
            or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5)):
        raise ValueError('Invalid asset bounds or metric placement')
    world = np.array(list(product(*bounds.T))) @ basis.T + center
    if world_transform is not None:
        matrix = np.asarray(world_transform, dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)):
            raise ValueError('Expected rigid registration to the surveyed world')
        world = world @ matrix[:3, :3].T + matrix[:3, 3]
    return np.stack((world[:, :2].min(axis=0), world[:, :2].max(axis=0)))


def mesh_boxes(mesh, anchor, key, world_transform=None):
    if hasattr(mesh, 'meshes'):
        return {seat['id']: world_bounds(seat['anchor'], mesh.meshes[seat['asset']].bounds, world_transform)
                for seat in anchor['instances']}
    return {key: world_bounds(anchor, mesh.bounds, world_transform)}


class TrafficSurvey:
    def __init__(self, packet):
        if packet['schema'] != 'traffic-world-xy-v1' or not packet['scene']:
            raise ValueError('Expected a named, metric world-XY traffic survey')
        self.scene = packet['scene']
        self.rows = packet['samples']
        self.times = np.array([row['time_s'] for row in self.rows], dtype=float)
        if (self.times.ndim != 1 or len(self.times) < 2 or not np.isfinite(self.times).all()
                or not np.all(np.diff(self.times) > 0)):
            raise ValueError('Need at least two strictly increasing finite survey times')
        self.tracks = []
        self.fixed = {key: np.asarray(value, dtype=float) for key, value in packet.get('fixed_boxes_m', {}).items()}
        if any(not key or key == 'ego' or box.shape != (2, 2) or not np.isfinite(box).all()
               or np.any(box[1] <= box[0]) for key, box in self.fixed.items()):
            raise ValueError('Invalid stationary world-XY bounds')
        for row in self.rows:
            actors = {}
            if row['complete'] is True:
                for actor in row['actors']:
                    key = actor['id']
                    if not isinstance(key, str) or not key or key in actors or key in self.fixed:
                        raise ValueError('Missing or duplicate traffic identity')
                    state = np.array([*actor['xy_m'], actor['radius_m'], actor['max_speed_mps']], dtype=float)
                    if state.shape != (4,) or not np.isfinite(state).all() or min(state[2:]) <= 0:
                        raise ValueError('Expected finite XY, positive bounding radius and speed bound')
                    actors[key] = state
                if 'ego' not in actors:
                    raise ValueError('Complete samples must include the recording vehicle (ego)')
            self.tracks.append(actors)

    def evaluate(self, assets, margin_m=.25):
        keys = list(assets)
        boxes = np.asarray(list(assets.values()), dtype=float)
        if (boxes.shape != (len(keys), 2, 2) or not keys or not np.isfinite(boxes).all()
                or np.any(boxes[:, 1] <= boxes[:, 0]) or not np.isfinite(margin_m) or margin_m < 0):
            raise ValueError('Expected non-empty metric asset footprints and nonnegative margin')
        intervals = []
        fixed_min = None
        for key, fixed in self.fixed.items():
            gaps = np.maximum(np.maximum(boxes[:, 0] - fixed[1], fixed[0] - boxes[:, 1]), 0.)
            distances = np.linalg.norm(gaps, axis=1)
            closest = distances.argmin()
            candidate = (float(distances[closest]), key, keys[closest])
            if fixed_min is None or candidate < fixed_min:
                fixed_min = candidate
        for index, duration in enumerate(np.diff(self.times)):
            first, last = self.tracks[index:index + 2]
            if not first or not last or duration > .20:
                intervals.append({'clear': False, 'reason': 'unknown_coverage'})
                continue
            ids = sorted(first.keys() | last.keys())
            left = np.array([first.get(key, last.get(key)) for key in ids])
            right = np.array([last.get(key, first.get(key)) for key in ids])
            radius = np.maximum(left[:, 2], right[:, 2])
            travel = np.maximum(left[:, 3], right[:, 3]) * duration
            if np.any(np.linalg.norm(left[:, :2] - right[:, :2], axis=1) > travel + 1e-6):
                intervals.append({'clear': False, 'reason': 'track_speed_exceeded'})
                continue

            def separation(states):
                xy = states[:, None, :2]
                delta = np.maximum(np.maximum(boxes[:, 0] - xy, xy - boxes[:, 1]), 0.)
                return np.linalg.norm(delta, axis=2) - radius[:, None]

            paired = np.array([key in first and key in last for key in ids])
            lower = (separation(left) + separation(right)) / 2 - travel[:, None] * np.where(paired, .5, 1.)[:, None]
            actor, asset = np.unravel_index(lower.argmin(), lower.shape)
            nearest = (float(lower[actor, asset]), ids[actor], keys[asset])
            minimum, actor_id, asset_id = min(nearest, fixed_min) if fixed_min is not None else nearest
            intervals.append(dict(clear=minimum > margin_m, actor=actor_id, asset=asset_id,
                                  lower_bound_m=minimum, reason='clear' if minimum > margin_m else 'traffic'))
        return intervals

    def exposure(self, intervals, time_s):
        if not np.isfinite(time_s) or not self.times[0] <= time_s <= self.times[-1]:
            return {'clear': False, 'reason': 'outside_survey'}
        index = min(np.searchsorted(self.times, time_s, side='right') - 1, len(intervals) - 1)
        return dict(intervals[index], survey_interval=int(index))

    def require_window(self, assets, start, end, margin_m):
        if not (np.isfinite([start, end]).all() and self.times[0] <= start <= end <= self.times[-1]):
            raise ValueError('Exposure window is outside the traffic survey')
        rows = self.evaluate(assets, margin_m)
        first, last = (self.exposure(rows, time)['survey_interval'] for time in (start, end))
        for index in range(first, last + 1):
            if not rows[index]['clear']:
                raise ValueError(f'Traffic clearance refused at survey interval {index}: {rows[index]}')
        return dict(scene=self.scene, exposure_start_s=start, exposure_end_s=end, margin_m=margin_m,
                    minimum_bound_m=min(row['lower_bound_m'] for row in rows[first:last+1]))

    def require_clear(self, assets, margin_m):
        rows = self.evaluate(assets, margin_m)
        for index, row in enumerate(rows):
            if not row['clear']:
                raise ValueError(f'Traffic clearance refused at survey interval {index}: {row}')
        return dict(scene=self.scene, survey_start_s=self.times[0], survey_end_s=self.times[-1],
                    intervals=len(rows), margin_m=margin_m, minimum_bound_m=min(row['lower_bound_m'] for row in rows))
