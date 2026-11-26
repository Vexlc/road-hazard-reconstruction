"""Audit a fitted cover and its fractured solids in the exported GLBs."""

from hashlib import sha256
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, cKDTree

from road_geometry import Surface, glb_triangles, part_id


def require(ok, message):
    if not ok:
        raise ValueError(message)


def face_data(glb, name):
    for label, points, faces, uv, image in glb_triangles(glb, names=True, appearance=True):
        if label == name:
            return points[faces], uv[faces], sha256(image).hexdigest()
    raise ValueError('Missing model face ' + name)


def match_faces(scene, name, transform, original, tree):
    xyz, uv, rgb = face_data(scene, name)
    xyz = (xyz - transform[:3, 3]) @ transform[:3, :3]
    points, coords, ref_rgb = original
    require(rgb == ref_rgb, 'Fragment texture differs from the intact lid')
    _, ids = tree.query(xyz.mean(1))
    require(len(np.unique(ids)) == len(ids), 'Duplicate fractured face')
    ref = points[ids]
    # GLB permits cyclic shifts of the same oriented triangle indices.
    shifts = np.argmin(
        [np.linalg.norm(xyz - np.roll(ref, i, axis=1), axis=2).max(1) for i in range(3)], axis=0
    )
    order = (np.arange(3)[None, :] - shifts[:, None]) % 3
    error = np.linalg.norm(xyz - ref[np.arange(len(ids))[:, None], order], axis=2)
    texerr = abs(uv - coords[ids[:, None], order])
    require(error.max() < 2e-6 and texerr.max() < 2e-6, 'Fracture changed source shape or UV')
    return ids, float(error.max()), float(texerr.max())


def bearing(parts, name, fixed, supports):
    vertices = np.unique(next(v for n, v, _ in parts if n == name), axis=0)
    origin = vertices.copy()
    # For a floor fragment test against the fixed floor alone. The independent shaft-clearance test
    # below rules out contact with the shaft walls, including the insertion path from above.
    origin[:, 2] = 1
    dist, kind, _ = fixed.cast(origin, np.broadcast_to([0, 0, -1], origin.shape))
    gap = vertices[:, 2] - (1 - dist)
    hit = np.isin(kind, supports) & (abs(gap) < 0.0002)
    require(
        hit.sum() >= 3 and not (gap[np.isfinite(gap)] < -0.0002).any(), 'Penetration or missing support'
    )
    fragment = name.startswith('Lid fragment ')
    solid = np.concatenate(
        [v[f] for n, v, f in parts if n.startswith('Lid ') and n.startswith('Lid fragment ') == fragment]
    )
    origin = vertices.mean(0)
    solid = solid.astype(float) - origin
    # Signed tetrahedron volumes yield the centre of mass of a uniform-density solid.
    weights = np.einsum('ij,ij->i', solid[:, 0], np.cross(solid[:, 1], solid[:, 2]))
    center = origin + np.average(solid.sum(1) / 4, axis=0, weights=weights)
    eq = ConvexHull(vertices[hit, :2]).equations
    margin = -float((eq[:, :2] @ center[:2] + eq[:, 2]).max())
    require(margin > 0, 'The projected centre of mass lies outside the supporting contacts')
    return dict(
        contacts_within_0_2mm=int(hit.sum()),
        minimum_signed_gap_m=float(gap[np.isfinite(gap)].min()),
        uniform_density_centre_of_mass_m=center.tolist(),
        centre_inside_support_margin_m=margin,
    )


def reference_outline(bundle, dest, profile):
    im = Image.open(bundle / 'references/lid_ortho.png').convert('RGB')
    pix = np.rint(np.asarray(profile['fracture_path_uv']) * (im.width - 1)).astype(int)
    ImageDraw.Draw(im).line([tuple(p) for p in pix], width=4, fill='#3ee1ce')
    im.save(dest / 'fracture-line.jpg', quality=96)


def section_map(glb, meta, dest, span_x, key):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    parts = list(glb_triangles(glb, names=True))
    groups = {
        name: (
            'fragment'
            if name.startswith('Lid fragment ')
            else 'lid'
            if name.startswith('Lid ')
            else 'shaft'
            if part_id(name) in (6, 7)
            else 'seat'
            if part_id(name) == 5
            else 'road'
        )
        for name, _, _ in parts
    }
    colors = {
        'road': '#b8b7ad',
        'seat': '#7b7867',
        'shaft': '#3b6473',
        'lid': '#ac7a37',
        'fragment': '#d55440',
    }
    unit = meta['panel_size_m'][0]
    left, right = span_x
    y0, y1 = -0.64 * unit, 0.64 * unit
    ycut = unit * 0.0173
    xs = np.linspace(left, right, 720)
    ys = np.linspace(y1, y0, round(720 * (y1 - y0) / (right - left)))
    x, y = np.meshgrid(xs, ys)
    origin = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    direction = np.broadcast_to([0, 0, -1], origin.shape)
    full = Surface(glb)
    depth, material, _ = full.cast(origin, direction)
    topcolor = np.array(
        [
            [245, 244, 238],
            [199, 199, 184],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [124, 121, 104],
            [69, 115, 130],
            [59, 100, 115],
            [0, 0, 0],
            [202, 155, 84],
            [141, 102, 69],
            [171, 119, 62],
        ],
        'u1',
    )[material]
    loose = {name for name, label in groups.items() if label == 'fragment'}
    visible = np.zeros(len(origin), bool)
    if loose:
        single = Surface(glb, include_names=loose)
        ld, _, _ = single.cast(origin, direction)
        finite = np.isfinite(ld) & np.isfinite(depth)
        visible[finite] = abs(ld[finite] - depth[finite]) < 0.0002
        topcolor[visible] = [211, 87, 65]
    fig, (plan, cut) = plt.subplots(1, 2, figsize=(11, 3.9), dpi=135, layout='constrained')
    fig.patch.set_facecolor('#f4f3ec')
    plan.imshow(topcolor.reshape(y.shape + (3,)), extent=[left, right, y0, y1], aspect='equal')
    plan.axhline(ycut, color='#fff4b7', lw=1.3, ls='--')
    plan.set_axis_off()
    plan.set_title(key + '  俯视首个可见表面', fontproperties=font)
    for name, v, f in parts:
        tri = v[f]
        # Intersect the actual GLB triangles with a fixed vertical plane.
        choose = (tri[:, :, 1].min(1) < ycut) & (tri[:, :, 1].max(1) > ycut)
        tri = tri[choose]
        if not len(tri):
            continue
        pts = []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            va, vb = tri[:, a], tri[:, b]
            ok = (va[:, 1] > ycut) != (vb[:, 1] > ycut)
            q = va[ok] + (vb[ok] - va[ok]) * ((ycut - va[ok, 1]) / (vb[ok, 1] - va[ok, 1]))[:, None]
            pts.extend(zip(np.flatnonzero(ok), q[:, [0, 2]]))
        pts.sort(key=lambda item: item[0])
        segments = np.asarray([p for _, p in pts]).reshape(-1, 2, 2)
        cut.add_collection(LineCollection(segments, colors=colors[groups[name]], linewidths=2))
    cut.set_xlim(left, right)
    cut.set_ylim(-meta['depth_design_m'] - 0.05, 0.076)
    cut.set_aspect('equal', adjustable='box')
    cut.axhline(0, color='#99968c', lw=0.7, ls=':', zorder=0)
    cut.set_xlabel('沿截线距离（米）', fontproperties=font)
    cut.set_ylabel('相对路面高度（米）', fontproperties=font)
    cut.set_title('沿黄线剖开 · 截面来自模型三角面', fontproperties=font)
    cut.spines[['top', 'right']].set_visible(False)
    cut.grid(axis='y', alpha=0.14)
    fig.savefig(dest / 'fracture-section.png')
    plt.close(fig)
    return dict(
        section_y_m=ycut,
        map_extent_xy_m=[left, right, y0, y1],
        fragment_visible_surface_pixels=int(visible.sum()),
        ray_grid_size=list(x.shape[::-1]),
    )


def check_fragments(run, baseline, states, topology):
    bundle = run / 'series'
    profile = json.loads((bundle / 'calibration/survey.json').read_text())['perforations']
    main = 'Lid face'
    frag = 'Lid fragment face'

    def glb(k):
        return bundle / 'assets/raised' / k / 'asset.glb'

    reference = face_data(glb(baseline), main)
    tree = cKDTree(reference[0].mean(1))
    original_volume = states[baseline]['solid']['metal_volume_m3']
    other_ids = [
        k
        for k in states
        if k != baseline
        and (glb(k).parent / 'metadata.json').is_file()
        and 'fragment_location' in json.loads((glb(k).parent / 'metadata.json').read_text())
    ]
    metas = {k: json.loads((glb(k).parent / 'metadata.json').read_text()) for k in [baseline, *other_ids]}
    scale = metas[baseline]['panel_size_m'][0]
    right = 0.67 * scale
    for k in other_ids:
        vertices = [v for n, v, _ in glb_triangles(glb(k), names=True) if n == frag]
        if vertices:
            right = max(right, float(vertices[0][:, 0].max()) + 0.08)
    seen = None
    reference_outline(bundle, run / 'evaluation' / baseline, profile)
    for k, meta in metas.items():
        dest = run / 'evaluation' / k
        states[k]['section'] = section_map(glb(k), meta, dest, (-0.67 * scale, right), k)
        if k == baseline:
            continue
        named = list(glb_triangles(glb(k), names=True))
        ids, err, uv = match_faces(glb(k), main, np.eye(4), reference, tree)
        if seen is None:
            seen = set(ids)
        require(set(ids) == seen, 'Retained faces differ between states')
        fixed = Surface(glb(k), exclude=(9, 10, 11))
        resident = [
            (part_id(n), v, f)
            for n, v, f in named
            if n.startswith('Lid ') and not n.startswith('Lid fragment ')
        ]
        solid = topology(resident)
        require(
            solid['through_holes'] == meta['through_holes'], 'Retained hole count disagrees with topology'
        )
        stats = dict(
            retained_solid=solid,
            retained_face_triangles=len(ids),
            max_retained_vertex_error_m=err,
            max_retained_uv_error=uv,
            retained_bearing=bearing(named, 'Lid underside', fixed, [5]),
            fragment_location=meta['fragment_location'],
        )
        other = [(part_id(n), v, f) for n, v, f in named if n.startswith('Lid fragment ')]
        if other:
            pose = np.array(meta['fragment_transform'])
            require(
                np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3)) and np.linalg.det(pose[:3, :3]) > 0,
                'Invalid rigid fragment pose',
            )
            fid, e, u = match_faces(glb(k), frag, pose, reference, tree)
            require(
                not seen.intersection(fid) and len(seen) + len(fid) == len(reference[0]),
                'Split does not partition the intact top',
            )
            loose = topology(other)
            err_volume = abs(loose['metal_volume_m3'] + solid['metal_volume_m3'] - original_volume)
            require(err_volume < 1.0e-7, 'Fragment volumes do not sum to the original cover')
            floor = meta['fragment_location'] == 'floor'
            contact = Surface(
                glb(k), include_names={'Shaft bottom'} if floor else {'Road around the aperture'}
            )
            support = bearing(named, 'Lid fragment underside', contact, [7] if floor else [1])
            vertices = next(v for n, v, _ in named if n == frag)
            if floor:
                extent = np.asarray(meta['panel_size_m']) / 2 - profile['seat_overlap_m']
                norm = abs(vertices[:, :2] / extent)
                radial = np.linalg.norm(norm, axis=1) if profile['shape'] == 'round' else norm.max(1)
                require(radial.max() < 1, 'Detached fragment intersects the shaft')
            require(
                states[k]['section']['fragment_visible_surface_pixels'] > 0,
                'Detached piece disappears in plan',
            )
            stats.update(
                fragment_solid=loose,
                fragment_face_triangles=len(fid),
                max_fragment_vertex_error_m=e,
                max_fragment_uv_error=u,
                volume_sum_error_m3=err_volume,
                fragment_bearing=support,
                fragment_bottom_z_m=float(
                    next(v for n, v, _ in named if n == 'Lid fragment underside')[:, 2].min()
                ),
            )
        else:
            require(meta['fragment_location'] == 'missing', 'Detached fragment was lost')
        stats['retained_metal_volume_fraction'] = solid['metal_volume_m3'] / original_volume
        stats['source_top_triangles'] = len(reference[0])
        states[k]['fracture'] = stats
