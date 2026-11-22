"""Fit a solid perforated cover, recessed seat and open shaft from a registered photograph."""

import hashlib
import json

import bpy  # noqa: F401 -- registers the embedded mathutils module.
import numpy as np
from mathutils.geometry import delaunay_2d_cdt
from PIL import Image
from scipy.ndimage import map_coordinates

from chambers import contour, sheet
from mesh_io import Mesh, ROOT, smooth
from survey_chamber import lid_pose


def pixels(name):
    return np.array(Image.open(ROOT / "references" / name).convert("RGB"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cross_area(points):
    return np.cross(points, np.roll(points, -1, axis=0)).sum() / 2


def fracture_polygon(profile):
    """The marked path runs from above the panel to below it in photograph UV."""
    path = np.array(profile['fracture_path_uv'], dtype=float)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all()
            or path[0, 1] >= 0 or path[-1, 1] <= 1 or (np.diff(path[:, 1]) <= 0).any()
            or ((path[:, 0] <= 0) | (path[:, 0] >= 1)).any()):
        raise ValueError('Trace a top-to-bottom fracture with increasing image y')
    uv = np.vstack([path, [1.1, path[-1, 1]], [1.1, path[0, 1]]])
    w, h = profile['size_m']
    return (uv - 0.5) * [w, -h]


def cover_mesh(profile, openings, piece=None):
    size = np.array(profile["size_m"])
    hole_loops = [(np.array(c) - 0.5) * [size[0], -size[1]] for c in openings]
    shape = profile["shape"]
    if shape == "round":
        t = np.arange(256) * 2 * np.pi / 256
        outer = np.column_stack([np.cos(t), np.sin(t)]) * size / 2
        if any((np.linalg.norm(loop / (size / 2), axis=1) >= 1).any() for loop in hole_loops):
            raise ValueError("An opening crosses the round lid's edge")
    elif shape == "rect":
        outer = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
    else:
        raise ValueError("Supported cover outlines are round and rect")
    loops = [outer, *hole_loops]
    cut = fracture_polygon(profile) if 'fracture_path_uv' in profile else None
    if cut is not None:
        loops.append(cut)
    vertices = np.vstack(loops)
    edges = []
    start = 0
    for loop in loops:
        indices = np.arange(len(loop)) + start
        edges.extend(zip(indices.tolist(), np.roll(indices, -1).tolist()))
        start += len(loop)
    relief_path = ROOT / "calibration/relief.npy"
    if relief_path.exists():
        # Interior vertices follow the photographed relief; constraints keep triangles out of holes.
        axis = np.linspace(-0.5, 0.5, 111)[1:-1]
        gx, gy = np.meshgrid(axis * size[0], axis * size[1])
        vertices = np.vstack([vertices, np.column_stack([gx.ravel(), gy.ravel()])])
    # Request the constrained triangulation of the convex hull. Classify triangle centroids
    # in the original loops, so even multiple nonconvex openings stay open.
    points, _, faces, *_ = delaunay_2d_cdt(vertices.tolist(), edges, [], 0, 1e-8, False)
    xy = np.asarray(points)
    triangles = np.asarray(faces, dtype="i4")
    centroids = xy[triangles].mean(axis=1)

    def inside(poly):
        # Closed polygon parity, evaluated only in its bounding box.
        near = ((centroids >= poly.min(0)) & (centroids <= poly.max(0))).all(1)
        active = centroids[near]
        hit = np.zeros(len(active), bool)
        for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0)):
            if ay != by:
                hit ^= ((ay > active[:, 1]) != (by > active[:, 1])) & (
                    active[:, 0] < ax + (active[:, 1] - ay) * (bx - ax) / (by - ay)
                )
        result = np.zeros(len(near), bool)
        result[near] = hit
        return result

    keep = inside(outer)
    for poly in hole_loops:
        keep &= ~inside(poly)
    tri = triangles[keep]
    if not len(tri):
        raise ValueError("No metal remains inside the silhouette")
    winding = np.cross(xy[tri[:, 1]] - xy[tri[:, 0]], xy[tri[:, 2]] - xy[tri[:, 0]])
    tri[winding < 0] = tri[winding < 0, ::-1]
    uv = xy / size + 0.5
    height = np.zeros(len(xy))
    if relief_path.exists():
        img = np.load(relief_path)
        measure = map_coordinates(
            img, [(1 - uv[:, 1]) * (len(img) - 1), uv[:, 0] * (img.shape[1] - 1)], order=1, mode="nearest"
        )
        active = measure[np.unique(tri)]
        low, high = np.percentile(active, [3, 97])
        if high <= low:
            raise ValueError("No usable cover relief in the displacement map")
        height = (np.clip((measure - low) / (high - low), 0, 1) - 0.5) * float(profile["relief_span_m"])
    top = np.column_stack([xy, height])
    projected_area = None
    if cut is not None:
        if piece:
            in_fragment = inside(cut)[keep]
            tri = tri[in_fragment if piece == 'fragment' else ~in_fragment]
            if not len(tri):
                raise ValueError('The marked fracture must divide the metal panel')
        projected_area = abs(np.cross(xy[tri[:, 1]] - xy[tri[:, 0]], xy[tri[:, 2]] - xy[tri[:, 0]])).sum()/2
    thickness = float(profile["thickness_m"])
    bottom = np.column_stack([xy, np.full(len(xy), -thickness)])
    mesh = Mesh()
    face = mesh.material("Photographed perforated face", pixels("lid_ortho.png"), 0.76, 0.48, 2.1)
    iron = mesh.material("Exposed cast and cut iron", pixels("iron.png"), 0.78, 0.48, 1.3)
    mesh.add("Lid face", top, tri, uv, face)
    mesh.add("Lid underside", bottom, tri[:, ::-1], uv, iron)
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    _, first, counts = np.unique(np.sort(pairs, axis=1), axis=0, return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError("Non-manifold metal boundary")
    a, b = pairs[first[counts == 1]].T
    quads = np.stack([top[a], bottom[a], bottom[b], top[b]], axis=1)
    sides = np.arange(quads.shape[0] * 4).reshape(-1, 4)
    length = np.linalg.norm(xy[b] - xy[a], axis=1)
    start = np.mod(xy[a].sum(axis=1) / size.min(), 0.75)
    end = np.minimum(1.0, start + length / size.min())
    coords = np.stack(
        [
            np.c_[start, np.ones(len(a))],
            np.c_[start, np.zeros(len(a))],
            np.c_[end, np.zeros(len(a))],
            np.c_[end, np.ones(len(a))],
        ],
        axis=1,
    )
    mesh.add(
        "Lid aperture walls and perimeter",
        quads,
        np.concatenate([sides[:, [0, 1, 2]], sides[:, [0, 2, 3]]]),
        coords,
        iron,
    )
    area = sum(abs(cross_area(c)) for c in hole_loops)
    stats = dict(
        through_holes=len(hole_loops),
        open_area_m2=float(area),
        open_face_fraction=float(area / abs(cross_area(outer))),
    )
    if cut is not None:
        stats['projected_metal_area_m2'] = float(projected_area)
    if piece:
        # For a connected planar triangulation the number of holes is 1 - (V - E + F).
        stats.update(through_holes=1 - len(np.unique(tri)) + len(counts) - len(tri),
                     open_area_m2=None, open_face_fraction=None)
    return mesh, stats


def fragment_pose(piece, state, profile):
    location = state['location']
    if location == 'missing':
        return None
    if location not in ('floor', 'road'):
        raise ValueError('Place the fragment on the floor or road, or mark it missing')
    _, v, f, *_ = piece.parts[0]
    tri = v[f, :2].astype(float)
    area = abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    center = np.average(tri.mean(axis=1), weights=area, axis=0)
    target = np.asarray(state['center_xy_m'], dtype=float)
    yaw = np.deg2rad(float(state.get('yaw_deg', 0)))
    if target.shape != (2,) or not np.isfinite(target).all() or not np.isfinite(yaw):
        raise ValueError('Invalid fragment centre or rotation')
    ca, sa = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:2, :2] = [[ca, -sa], [sa, ca]]
    pose[:2, 3] = target - pose[:2, :2] @ center
    pose[2, 3] = profile['thickness_m']
    xy = v[:, :2] @ pose[:2, :2].T + pose[:2, 3]
    radius = np.array(profile['size_m'])/2
    if location == 'floor':
        r = xy / (radius - profile['seat_overlap_m'])
        extent = np.linalg.norm(r, axis=1) if profile['shape'] == 'round' else abs(r).max(axis=1)
        if extent.max() >= 1:
            raise ValueError('The fallen fragment intersects the shaft wall')
        pose[2, 3] -= profile['shaft_depth_m']
    else:
        # Certify the whole fragment lies across a supporting tangent of the fixed frame.
        axis = target / max(np.linalg.norm(target), 1e-12)
        radius += profile['frame_band_m']
        limit = np.linalg.norm(axis * radius) if profile['shape'] == 'round' else abs(axis * radius).sum()
        if min(xy @ axis) <= limit:
            raise ValueError('Road fragment overlaps the frame or open shaft')
    return pose


def installation_pose(grade):
    z, tilt, heading = (float(grade[k]) for k in ('center_z_m', 'tilt_deg', 'uphill_deg'))
    if not np.isfinite([z, tilt, heading]).all() or not 0 <= tilt < 90:
        raise ValueError('Invalid installed frame height or tilt')
    angle, bearing = np.deg2rad([tilt, heading])
    direction = np.array([np.cos(bearing), np.sin(bearing)])
    pose = np.eye(4)
    pose[:2, :2] += (np.cos(angle) - 1) * np.outer(direction, direction)
    pose[2, :2] = np.sin(angle) * direction
    pose[:2, 2] = -pose[2, :2]
    pose[2, 2] = np.cos(angle)
    pose[2, 3] = z
    return pose


def sample_surface(parts, field):
    """First vertical hit on the delivered triangles. The substrate stays available under the lid."""
    from mathutils.bvhtree import BVHTree

    vertices, faces = [], []
    for _, v, f, *_ in parts:
        faces.extend((f + len(vertices)).tolist())
        vertices.extend(v.tolist())
    tree = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
    start = float(np.max(np.asarray(vertices)[:, 2]) + 1)
    axis = np.linspace(-field['width']/2, field['width']/2, len(field['height']))
    height = np.zeros_like(field['height'], dtype='f4')
    # The outermost cells coincide with the fixed pavement boundary at zero.
    for j, y in enumerate(-axis[1:-1], start=1):
        for i, x in enumerate(axis[1:-1], start=1):
            hit = tree.ray_cast((x, y, start), (0, 0, -1))[0]
            if hit is None:
                raise ValueError('Open seam in the installed cover or pavement')
            height[j, i] = hit.z
    return height


def shaft_mesh(profile, trace, pose=None):
    size = np.array(profile["size_m"])
    band, seat, t, depth = [
        float(profile[k]) for k in ("frame_band_m", "seat_overlap_m", "thickness_m", "shaft_depth_m")
    ]
    clear = 0.002
    if not (0 < seat < size.min() / 2 and 0 < clear < band and 0 < t < depth and depth > t + 0.04):
        raise ValueError("Invalid frame, seat, cover thickness or shaft depth")
    radius = size / 2
    span = trace["road_span_m"]
    ts = np.linspace(0, 2 * np.pi, 257)
    outline = np.stack(contour(ts, profile["shape"]), axis=-1)
    square = np.stack(contour(ts, "rect"), axis=-1)
    mesh = Mesh()
    pavement = mesh.material("Pavement", pixels("apron.png"), 0.91, 0, 1.8)
    iron = mesh.material("Fixed frame and seat", pixels("iron.png"), 0.76, 0.49, 1.3)
    liner = pixels(profile["lining"])
    gradient = np.linspace(0.70, 0.97, len(liner))[:, None, None]
    lining = np.tile(np.uint8(liner * gradient), (1, 3, 1))
    concrete = mesh.material("Designed concrete well", lining, 0.94, 0.0, 1.8)
    floor = mesh.material("Designed shaft floor", np.uint8(liner * 0.66), 0.97, 0.0, 1.5)

    def band_part(name, levels, material, texture="photo", road=False, transform=None):
        xy = np.stack([p for p, _ in levels])
        heights = np.stack([np.broadcast_to(z, (len(ts),)) for _, z in levels])
        xyz = np.dstack([xy, heights])
        if transform is not None:
            xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        if texture == "photo":
            uv = 0.5 + xy / span
        else:
            uv = np.stack(
                np.broadcast_arrays(ts[None, :] / (2 * np.pi), np.linspace(1, 0, len(levels))[:, None]), -1
            )
        sheet(mesh, name, xyz, uv, material, road=road)

    outer = outline * (radius + band)
    rim = outline * (radius + clear)
    shelf = outline * (radius - seat)
    top_edge = np.c_[outer, np.zeros(len(ts))]
    wall_edge = np.c_[shelf, np.full(len(ts), -t - 0.018)]
    if pose is not None:
        top_edge = top_edge @ pose[:3, :3].T + pose[:3, 3]
        wall_edge = wall_edge @ pose[:3, :3].T + pose[:3, 3]
        if wall_edge[:, 2].min() <= -depth or (abs(top_edge[:, :2]).max() >= span/2):
            raise ValueError('The installed frame exceeds the designed shaft or pavement bounds')
    levels = [((1 - f) * square * span / 2 + f * top_edge[:, :2], 0) for f in np.linspace(0, 1, 17)]
    band_part("Road around the aperture", levels, pavement, road=True)
    if np.any(top_edge[:, 2]):
        band_part("Chipped concrete collar", [(top_edge[:, :2], 0), (top_edge[:, :2], top_edge[:, 2])],
                  concrete, "strip")
    band_part(
        "Cast ring and cover seat",
        [
            (outer, 0),
            (outline * (radius + band * 0.88), 0.0015),
            (rim, 0.0015),
            (rim, -t),
            (shelf, -t),
            (shelf, -t - 0.018),
        ],
        pavement if profile["photographed_surround"] else iron,
        "photo" if profile["photographed_surround"] else "strip",
        transform=pose,
    )
    band_part("Inspection shaft", [(wall_edge[:, :2], wall_edge[:, 2]), (shelf, -depth)], concrete, "strip")
    xy = np.vstack([[0, 0], shelf[:-1]])
    idx = np.arange(1, len(xy))
    mesh.add(
        "Shaft bottom",
        np.c_[xy, np.full(len(xy), -depth)],
        np.c_[np.zeros(len(idx), int), idx, np.roll(idx, -1)],
        0.5 + xy / size,
        floor,
    )
    a = np.linspace(-span / 2, span / 2, 321)
    x, y = np.meshgrid(a, -a)

    def distance(r):
        return (
            np.hypot(x / r[0], y / r[1])
            if profile["shape"] == "round"
            else np.maximum(abs(x / r[0]), abs(y / r[1]))
        )

    opening = distance(radius - seat) < 1
    distance_to_frame = (distance(radius + band) - 1) * min(radius + band)
    alpha = smooth((0.085 - distance_to_frame) / 0.075)
    field = dict(
        height=np.where(opening, -depth, 0),
        alpha=alpha,
        defect_mask=distance_to_frame <= 0,
        opening_mask=opening,
        width=span,
        length=span,
    )
    return mesh, field


def build():
    cal = ROOT / "calibration"
    cfg = json.loads((cal / "survey.json").read_text())
    profile = cfg["perforations"]
    trace = json.loads((cal / "grate.json").read_text())
    profile = dict(profile, size_m=trace["size_m"], thickness_m=trace["thickness_m"])
    intact = cover_mesh(profile, trace["openings_uv"])
    split = None
    if 'fracture_path_uv' in profile:
        split = [cover_mesh(profile, trace['openings_uv'], piece) for piece in ('retained', 'fragment')]
    base_stock, base_field = shaft_mesh(profile, trace)
    mask_hash = digest(ROOT / "references" / profile["mask"])
    photo_hash = digest(ROOT / "references/frame.png")
    size = np.asarray(profile["size_m"])
    cache = ROOT / "mesh_cache"
    cache.mkdir(exist_ok=True)
    for state in cfg["states"]:
        key = state["id"]
        grade = state.get('installation_grade')
        installation = installation_pose(grade) if grade is not None else None
        stock, field = shaft_mesh(profile, trace, installation) if grade is not None else (base_stock, base_field)
        changed = trace["broken_openings_uv"].get(key)
        lid, stats = cover_mesh(profile, changed) if changed else intact
        absent = state.get("absent", False)
        angle = float(state.get("angle_deg", 0))
        offset = np.asarray(state.get("offset_m", [0, 0]), dtype=float)
        fragment = state.get('fragment')
        fracture = {}
        if installation is not None and (angle or offset.any() or fragment is not None):
            raise ValueError('Installation grade requires a seated or missing lid')
        if fragment is not None:
            if split is None or absent or changed or angle or offset.any():
                raise ValueError('A split cover requires a seated panel and a traced fracture')
            (lid, stats), (loose, loose_stats) = split
            transform = fragment_pose(loose, fragment, profile)
            fracture = dict(fragment_location=fragment['location'],
                            fragment_transform=transform.tolist() if transform is not None else None,
                            fragment_metal_area_m2=loose_stats['projected_metal_area_m2'],
                            original_metal_area_m2=intact[1]['projected_metal_area_m2'])
        if (
            not np.isfinite(angle)
            or not 0 <= angle <= 90
            or offset.shape != (2,)
            or not np.isfinite(offset).all()
            or (absent and (angle or offset.any()))
        ):
            raise ValueError("Invalid grate pose for " + key)
        mesh = Mesh()
        mesh.materials.update(stock.materials)
        mesh.parts.extend(stock.parts)
        result = dict(field)
        pose = None
        if not absent:
            pose = lid_pose(size[0] / 2, angle, offset, 0)
            if installation is not None:
                pose = installation @ pose
            if offset.any():
                # Compute first contact with the fixed road, flange or seat under the moved cover.
                from mathutils.bvhtree import BVHTree

                vertices = []
                faces = []
                for name, v, f, *_ in stock.parts:
                    if name in ("Cast ring and cover seat", "Road around the aperture"):
                        faces.extend((f + len(vertices)).tolist())
                        vertices.extend(v.tolist())
                tree = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
                bottom = (
                    next(p[1] for p in lid.parts if p[0] == "Lid underside") @ pose[:3, :3].T + pose[:3, 3]
                )
                contacts = []
                for x, y, z in bottom:
                    hit, *_ = tree.ray_cast((x, y, 1.0), (0, 0, -1.0))
                    if hit is not None:
                        contacts.append(hit.z - z)
                if not contacts:
                    raise ValueError("Displaced grate has no support")
                pose[2, 3] += max(contacts)
            mesh.materials.update(lid.materials)
            for name, v, f, uv, m, isroad in lid.parts:
                mesh.parts.append((name, (v @ pose[:3, :3].T + pose[:3, 3]).astype("f4"), f, uv, m, isroad))
            if offset.any():
                a = np.linspace(-field["width"] / 2, field["width"] / 2, field["height"].shape[0])
                x, y = np.meshgrid(a, -a)
                d = np.maximum(abs(x - offset[0]) - size[0] / 2, abs(y - offset[1]) - size[1] / 2)
                result["alpha"] = np.maximum(field["alpha"], smooth((0.065 - d) / 0.06))
        if fragment is not None and transform is not None:
            for name, v, f, uv, material, road in loose.parts:
                moved = (v @ transform[:3, :3].T + transform[:3, 3]).astype('f4')
                mesh.parts.append((name.replace('Lid ', 'Lid fragment ', 1), moved, f, uv, material, road))
            if fragment['location'] == 'road':
                # Preserve host road around the loose piece. Only its actual footprint needs an opaque apron.
                from scipy.spatial import ConvexHull
                from scipy.ndimage import distance_transform_edt
                from PIL import ImageDraw

                verts = next(v[:, :2] for name, v, *_ in mesh.parts if name == 'Lid fragment face')
                n = result['height'].shape[0]
                pos = (verts / [field['width'], -field['length']] + 0.5) * (n - 1)
                if (pos < 2).any() or (pos > n - 3).any():
                    raise ValueError('Detached piece extends beyond the road support grid')
                mask = Image.new('L', (n, n))
                ImageDraw.Draw(mask).polygon([tuple(v) for v in pos[ConvexHull(pos).vertices]], fill=255)
                distance = distance_transform_edt(np.asarray(mask) == 0) * field['width'] / (n - 1)
                result['alpha'] = np.maximum(field['alpha'], smooth((0.045 - distance) / 0.04))
        category = "missing" if absent else "raised"
        if installation is not None:
            result['substrate_height'] = sample_surface(stock.parts, field)
            result['height'] = result['substrate_height'] if absent else sample_surface(mesh.parts, field)
        if absent:
            stats = dict(
                reference_through_holes=stats["through_holes"],
                through_holes=0,
                open_area_m2=None,
                open_face_fraction=None,
            )
        meta = dict(
            id=key,
            label=state["label"],
            state=state["label"],
            family=cfg["family"],
            category=category,
            recipe_version="perforated-cover-v1",
            scale_kind=profile["scale_source"],
            break_polygon_uv=state.get("break_polygon_uv"),
            source_image="references/frame.png",
            source_sha256=photo_hash,
            mask_sha256=mask_hash,
            geometry_source="Rectified photographed face and individually marked through-openings. Designed recessed seat and hidden surfaces.",
            width_m=float(size[0] - 2 * profile["seat_overlap_m"]),
            length_m=float(size[1] - 2 * profile["seat_overlap_m"]),
            panel_size_m=size.tolist(),
            depth_design_m=profile["shaft_depth_m"],
            lid_thickness_m=profile["thickness_m"],
            lid_angle_degrees=None if absent else angle,
            lid_offset_m=None if absent else offset.tolist(),
            lid_transform=None if absent else pose.tolist(),
            exponent=2.0 if profile["shape"] == "round" else 16.0,
            road_projection=[239.5, 239.5, 479 / field["width"], 479 / field["length"], 768],
            seed=504031,
            **fracture,
            **stats,
        )
        if installation is not None:
            meta.update(installation_grade=grade, frame_transform=installation.tolist())
        Image.fromarray(pixels("pavement.jpg")).save(cache / (key + "_ground.jpg"), quality=94)
        mesh.save(ROOT / "assets" / category / key, cache, meta, result)
        print(key, stats, flush=True)


if __name__ == "__main__":
    build()
