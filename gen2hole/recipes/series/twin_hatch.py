"""Rebuild the two individually removable scanned leaves of a double service hatch."""

import json

import numpy as np
from PIL import Image

from mesh_io import Mesh, ROOT, smooth
from perforated_cover import cover_mesh, digest, pixels, sample_surface, shaft_mesh


def add_crossbeam(mesh, profile, center_x):
    width, depth = profile['beam_width_m'], profile['beam_depth_m']
    size_x, size_y = profile['size_m']
    thickness, seat = profile['thickness_m'], profile['seat_overlap_m']
    if not 0 < width < size_x / 2 or not 0 < depth < profile['shaft_depth_m'] - thickness:
        raise ValueError('Invalid centre bearing dimensions')
    # The designed bar bears on the two perimeter shelves, with one shelf half-width of overlap.
    dx, dy = width / 2, (size_y - seat) / 2
    xy = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]) + [center_x, 0]
    xyz = np.vstack([np.c_[xy, np.full(4, -thickness)], np.c_[xy, np.full(4, -thickness - depth)]])
    quads = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
    ids = np.arange(24).reshape(6, 4)
    tris = np.vstack([ids[:, [0, 1, 2]], ids[:, [0, 2, 3]]])
    uv = np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1))
    mesh.add('Central bearing beam', xyz[quads], tris, uv, 'Fixed frame and seat')


def build():
    cal = ROOT / 'calibration'
    cfg = json.loads((cal / 'survey.json').read_text())
    trace = json.loads((cal / 'grate.json').read_text())
    p = cfg['perforations']
    seam = np.asarray(p['twin_seam_uv'])
    size = np.asarray(p['size_m'])
    if p['shape'] != 'rect' or not p['solid_face'] or not 0 < seam[0] < seam[1] < 1:
        raise ValueError('Expected two solid rectangular leaves and a photographed seam')
    center_x = (seam.mean() - 0.5) * size[0]
    if (seam[1] - seam[0]) * size[0] >= p['beam_width_m']:
        raise ValueError('Both leaves must bear on the central beam')
    stock, field = shaft_mesh(p, trace)
    add_crossbeam(stock, p, center_x)
    substrate = sample_surface(stock.parts, field)
    leaves = {side: cover_mesh(p, [], panel_interval=bounds)[0] for side, bounds in
              [('left', (0, seam[0])), ('right', (seam[1], 1))]}
    n = len(substrate)
    axis = np.linspace(-field['width'] / 2, field['width'] / 2, n)
    x, y = np.meshgrid(axis, -axis)
    material = 'Scanned concrete and metal face'
    for state in cfg['states']:
        panels = state['panels']
        missing = sorted(set(leaves) - set(panels))
        dx = float(state.get('right_outward_m', 0))
        if (len(set(panels)) != len(panels) or set(panels) - set(leaves) or
                not 0 <= dx < size[0] / 4 or (dx and 'right' not in panels) or
                bool(state.get('absent')) != (len(panels) == 0)):
            raise ValueError('Invalid independent leaf configuration')
        mesh = Mesh()
        mesh.materials.update(stock.materials)
        mesh.parts.extend(stock.parts)
        result = dict(field, substrate_height=substrate)
        placements = {}
        for side in panels:
            shift = dx if side == 'right' else 0
            # The outer edge of this rigid leaf crosses the 2 mm radial clearance onto the fixed
            # raised flange. The full north and south rims support it along both long edges.
            if shift and shift <= 0.002:
                raise ValueError('Displaced leaf must reach the fixed flange')
            dz = p['thickness_m'] + 0.0015 if shift else 0
            placements[side] = [shift, 0., dz]
            leaf = leaves[side]
            face = leaf.parts[0][4]
            mesh.materials[material] = leaf.materials[face]
            mesh.orm_maps[material] = pixels('lid_orm.png')
            for name, verts, faces, uv, mat, road in leaf.parts:
                if mat != face:
                    mesh.materials[mat] = leaf.materials[mat]
                mesh.parts.append((side.title() + ' ' + name.lower(), np.float32(verts + placements[side]),
                                   faces, uv, material if mat == face else mat, road))
            if shift:
                distance = np.maximum.reduce([x - (size[0] / 2 + shift), center_x + shift - x, abs(y) - size[1] / 2])
                if np.any(distance[[0, -1]] <= 0) or np.any(distance[:, [0, -1]] <= 0):
                    raise ValueError('Displaced leaf extends beyond the road field')
                result['alpha'] = np.maximum(result['alpha'], smooth((0.055 - distance) / 0.05))
        result['height'] = sample_surface(mesh.parts, field) if panels else substrate
        key = state['id']
        cat = 'missing' if missing else 'raised'
        meta = dict(id=key, label=state['label'], state=state['label'], family=cfg['family'], category=cat,
                    recipe_version='double-hatch-v1', scale_kind=p['scale_source'], source_url=cfg['source_url'],
                    credit=cfg['credit'], source_image='references/frame.png',
                    source_sha256=digest(ROOT / 'references/frame.png'), mask_sha256=digest(ROOT / 'references' / p['mask']),
                    geometry_source='Registered scan face, seam and relative relief. Designed seat, beam, substrate and installation dimensions.',
                    panel_size_m=size.tolist(), missing_panels=missing, panel_translations_m=placements,
                    beam_center_x_m=center_x, beam_width_m=p['beam_width_m'], lid_thickness_m=p['thickness_m'],
                    depth_design_m=p['shaft_depth_m'], width_m=float(size[0] - 2 * p['seat_overlap_m']),
                    length_m=float(size[1] - 2 * p['seat_overlap_m']), exponent=16., seed=504031,
                    road_projection=[239.5, 239.5, 479 / field['width'], 479 / field['length'], 768])
        cache = ROOT / 'mesh_cache'
        cache.mkdir(exist_ok=True)
        Image.fromarray(pixels('pavement.jpg')).save(cache / (key + '_ground.jpg'), quality=94)
        mesh.save(ROOT / 'assets' / cat / key, cache, meta, result)
        print(key, placements, flush=True)


if __name__ == '__main__':
    build()
