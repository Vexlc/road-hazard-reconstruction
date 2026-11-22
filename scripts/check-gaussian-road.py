"""Test hosted pit and shaft visibility using original GLB rays and browser colour probes."""

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage

from atlas_gaussians import ATLAS, ROOT, catalog, projection
from pit_road import PitBed

sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, read_json, save_json, sha256

PLANE_Z = -0.003


class PitReference(PitBed):
    def __init__(self, path, profile):
        super().__init__(path)
        if self.kind != profile['kind'] or getattr(self, 'annotation_sha', None) != profile.get(
            'annotation_sha256'
        ):
            raise ValueError('Changed bed or source annotation: ' + str(path))
        with Image.open(BytesIO(base64.b64decode(profile['mask'].split(',')[1]))) as image:
            size = np.array(image.size)
        self.origin = np.asarray(profile['bounds'][:2])
        self.pitch = np.asarray(profile['bounds'][2:]) / size
        axes = [self.origin[i] + (np.arange(n) + 0.5) * self.pitch[i] for i, n in enumerate(size)]
        xy = np.stack(np.meshgrid(*axes), axis=-1)
        z, label = self.top(xy)
        mask = self.footprint(z, label, profile['roadHeight'])
        self.clearance = ndimage.distance_transform_edt(~mask, sampling=self.pitch[::-1])
        self.blend_width = profile['blendWidth']
        self.grid = mask

    def regions(self, xy, valid, plane):
        z, labeled = self.top(xy)
        exact = valid & self.footprint(z, labeled, plane)
        h, w = self.grid.shape
        # Certify interior/exterior only where the independent point ray and
        # all four adjacent physical texel centres agree. Score IoU over every pixel.
        ij = np.floor((xy - self.origin) / self.pitch - 0.5).astype('i8')
        cols, rows = ij[..., 0], ij[..., 1]
        inside = ndimage.binary_erosion(exact, iterations=2)
        outside = ndimage.binary_erosion(~exact, iterations=2)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            col = np.clip(cols + du, 0, w - 1)
            row = np.clip(rows + dv, 0, h - 1)
            occupied = valid & self.grid[row, col]
            inside &= occupied
            outside &= ~occupied
        far = self.clearance[
            np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)
        ] > self.blend_width + 3 * np.linalg.norm(self.pitch)
        # Count the surround within the photographed mesh's footprint; sky and
        # empty far-field pixels could otherwise make a trivial test pass.
        far &= valid & (rows >= 0) & (rows < h - 1) & (cols >= 0) & (cols < w - 1) & np.isfinite(z)
        return inside, outside, exact, far


def mesh_road(path):
    for name, verts, triangles in glb_triangles(path, names=True):
        if name == 'Road around the aperture':
            scene = o3d.t.geometry.RaycastingScene(nthreads=3)
            scene.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(triangles))
            return scene, verts.min(0), verts.max(0)
    raise ValueError(f'Missing pavement in {path}')


def road_pixels(info, road):
    w, h = info['width'], info['height']
    view, intrinsics = projection(info['camera'], w, h)
    pose = np.linalg.inv(view)
    y, x = np.mgrid[:h, :w]
    rays = (
        np.stack(
            [(x + 0.5 - w / 2) / intrinsics[0, 0], (y + 0.5 - h / 2) / intrinsics[1, 1], np.ones_like(x)],
            axis=2,
        )
        @ pose[:3, :3].T
    )
    eye = pose[:3, 3]
    valid = rays[:, :, 2] < -1e-6
    distance = (info['road_height_m'] - eye[2]) / np.where(valid, rays[:, :, 2], -1.0)
    xy = eye[:2] + distance[:, :, None] * rays[:, :, :2]
    if isinstance(road, PitReference):
        return road.regions(xy, valid, info['road_height_m'])
    # Cast vertically against the full original triangles. A miss at the centre
    # identifies the opening. A pavement hit or a point beyond the native mesh
    # bounds certifies the exterior, even when the camera sees past the horizon.
    scene, low, high = road
    test = np.zeros((h, w, 6), 'f4')
    test[:, :, :2] = xy
    test[:, :, 2] = high[2] + 1.0
    test[:, :, 5] = -1.0
    hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(test), nthreads=3)['t_hit'].numpy())
    beyond = np.any((xy < low[:2]) | (xy > high[:2]), axis=2)
    central = np.linalg.norm(xy, axis=2) < 0.9 * min(*high[:2], *(-low[:2]))
    inside = ndimage.binary_erosion(valid & central & ~hit, iterations=2)
    outside = ndimage.binary_erosion(~valid | beyond | hit, iterations=2)
    return inside, outside, None, None


def inspect(path, surface):
    info = read_json(path)
    inside, outside, exact, far = road_pixels(info, surface)

    def image(kind):
        data = np.asarray(
            Image.open(path.with_name(path.stem + '-' + kind + '.png')).convert('RGB'), dtype='i2'
        )
        if data.shape[:2] != outside.shape:
            raise ValueError(f'Camera or image dimensions mismatch in {path}')
        return data

    raw, probe, road, road_probe = [image(k) for k in ('gaussian', 'probe', 'road', 'road-probe')]
    response = np.max(abs(raw - probe), axis=2) > 8
    remaining = np.max(abs(road - road_probe), axis=2) > 8
    if not inside.any() or not outside.any():
        raise ValueError('Camera must observe both the opening and the ground')
    result = dict(
        camera=info['camera'],
        width=info['width'],
        height=info['height'],
        exterior_pixels=int(outside.sum()),
        exterior_raw_probe_pixels=int(response[outside].sum()),
        exterior_host_probe_pixels=int(remaining[outside].sum()),
        interior_pixels=int(inside.sum()),
        interior_raw_probe_pixels=int(response[inside].sum()),
        interior_host_probe_pixels=int(remaining[inside].sum()),
        interior_max_rgb_change=int(abs(raw[inside] - road[inside]).max()),
        interior_probe_max_rgb_change=int(abs(probe[inside] - road_probe[inside]).max()),
    )
    if exact is not None:
        actual = image('footprint')[:, :, 0] > 127
        common = int((actual & exact).sum())
        base, hard = image('base'), image('unblended')
        apron_before = (np.max(abs(hard - base), axis=2) > 8) & far
        apron_after = (np.max(abs(road - base), axis=2) > 8) & far
        result.update(
            native_opening_pixels=int(exact.sum()),
            browser_opening_pixels=int(actual.sum()),
            footprint_iou=common / int((actual | exact).sum()),
            opening_extra_in_exterior=int(actual[outside].sum()),
            opening_missing_in_interior=int((~actual[inside]).sum()),
            far_mesh_pixels=int(far.sum()),
            apron_before_pixels=int(apron_before.sum()),
            apron_after_pixels=int(apron_after.sum()),
        )
        if result['opening_extra_in_exterior'] or result['opening_missing_in_interior'] or apron_after.any():
            raise ValueError(
                f'{path.stem}: browser cutout differs from native GLB beyond the contour tolerance: {result}'
            )
    if (
        result['exterior_host_probe_pixels']
        or max(result[k] for k in ('interior_max_rgb_change', 'interior_probe_max_rgb_change')) > 1
    ):
        raise ValueError(f'{path.stem}: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=ROOT / 'runs/atlas-host-road')
    parser.add_argument('--out', type=Path, default=ROOT / 'validation/atlas-host-road.json')
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()
    rows = catalog()
    result = {}
    for path in sorted(args.captures.glob('*.json')):
        info = read_json(path)
        key = info['key']
        if args.ids and key not in args.ids:
            continue
        result.setdefault(key, []).append(path)
    if not result or (args.ids and set(args.ids) != result.keys()):
        raise ValueError('Missing browser captures')
    checks = {}
    for key, paths in result.items():
        row = rows[key]
        sources = {kind: sha256(ATLAS / row[kind]) for kind in ('glb', 'ply')}
        plane, probe = PLANE_Z, -0.075
        if row.get('pitRoad'):
            text = (ATLAS / row['pitRoad']).read_text()
            profile = json.loads(text.removeprefix('document.currentScript.roadProfile=')[:-2])
            if profile['glb'] != sources['glb']:
                raise ValueError(key + ' cutout belongs to a different mesh')
            plane, probe = profile['roadHeight'], profile['probeHeight']
        captures = [read_json(path) for path in paths]
        if any(
            c['sources'] != sources or c['road_height_m'] != plane or c.get('probe_height_m', -0.075) != probe
            for c in captures
        ):
            raise ValueError(key + ' browser sources or road plane differ from the verification inputs')
        if len({round(c['camera']['phi'], 4) for c in captures}) != 3:
            raise ValueError(key + ' requires three different camera elevations')
        surface = (
            PitReference(ATLAS / row['glb'], profile) if row.get('pitRoad') else mesh_road(ATLAS / row['glb'])
        )
        views = {read_json(p)['mode']: inspect(p, surface) for p in paths}
        if set(views) != {'oblique', 'low', 'top'}:
            raise ValueError(key + ' requires all three camera views')
        checks[key] = dict(sources=sources, views=views)
        if row.get('pitRoad'):
            checks[key].update(
                road_height_m=plane,
                tracer_below_m=probe,
                contact=profile['kind'],
                blend_width_m=profile['blendWidth'],
            )
        print(
            key,
            {k: (v['exterior_raw_probe_pixels'], v['exterior_host_probe_pixels']) for k, v in views.items()},
        )
    if not any(v['interior_host_probe_pixels'] for a in checks.values() for v in a['views'].values()):
        raise ValueError('No underground structures visible within any opening')
    save_json(
        args.out,
        dict(
            **(
                {}
                if any('road_height_m' in a for a in checks.values())
                else dict(road_height_m=PLANE_Z, tracer_below_m=-0.075)
            ),
            response_threshold_0_255=8,
            excluded_contour_px=2,
            audited_assets=checks,
        ),
    )


if __name__ == '__main__':
    main()
