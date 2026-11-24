"""Place a metric RoadGS mesh on surveyed pavement and raycast it into a calibrated host view."""

from io import BytesIO
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent / "sensors"))
from road_geometry import glb_triangles, sha256


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("A nonzero finite direction is required")
    return vector / length


def fit_road(points_world, seed_world, heading_world, *, up_world, radius_m):
    """Fit near an approximate ground point. Require local support around, not just beside, the anchor."""
    seed_world = np.asarray(seed_world, dtype=float)
    points_world = np.asarray(points_world, dtype=float)
    if (
        seed_world.shape != (3,)
        or not np.isfinite(seed_world).all()
        or points_world.ndim != 2
        or points_world.shape[1] != 3
        or not np.isfinite(radius_m)
        or radius_m <= 0
    ):
        raise ValueError("Expected a finite world seed, N × 3 points and a positive radius")
    up = unit(up_world)
    forward = unit(heading_world - up * np.dot(heading_world, up))
    axes = np.column_stack([forward, np.cross(up, forward), up])
    local = (np.asarray(points_world) - seed_world) @ axes
    local = local[(np.linalg.norm(local[:, :2], axis=1) < radius_m) & (abs(local[:, 2]) < 0.28)]
    if len(local) < 40 or radius_m <= 0:
        raise ValueError("Insufficient measured road near the anchor")
    x = np.column_stack([local[:, :2], np.ones(len(local))])
    # Fit to equally weighted cell medians so one dense scan line cannot dominate the plane.
    cells = np.floor(local[:, :2] / (radius_m / 5)).astype(int)
    _, bins = np.unique(cells, axis=0, return_inverse=True)
    reps = np.stack([np.median(local[bins == i], axis=0) for i in range(bins.max() + 1)])
    design = np.column_stack([reps[:, :2], np.ones(len(reps))])
    keep = np.ones(len(reps), bool)
    for tolerance in (0.10, 0.06, 0.04):
        if keep.sum() < 15:
            raise ValueError("Too few independent road cells")
        coef = np.linalg.lstsq(design[keep], reps[keep, 2], rcond=None)[0]
        keep = abs(design @ coef - reps[:, 2]) < tolerance
    support = reps[keep, :2]
    quadrants = (support[:, 0] > 0).astype(int) + 2 * (support[:, 1] > 0)
    error = abs(x @ coef - local[:, 2])
    inliers = error < 0.04
    surround = (
        len(support) >= 3
        and cv2.pointPolygonTest(cv2.convexHull(support.astype("f4")), (0.0, 0.0), True) >= 0.2 * radius_m
    )
    if (
        not surround
        or len(np.unique(quadrants)) != 4
        or len(support) < 15
        or keep.mean() < 0.65
        or inliers.sum() < 35
        or np.linalg.norm(coef[:2]) > 0.22
    ):
        raise ValueError("Road support is one-sided, discontinuous, or too steep")
    normal = unit(axes @ np.r_[-coef[:2], 1])
    forward = unit(forward - normal * np.dot(normal, forward))
    basis = np.column_stack([forward, np.cross(normal, forward), normal])
    return dict(
        center_world=(seed_world + coef[2] * up).tolist(),
        basis_columns=basis.tolist(),
        inliers=int(inliers.sum()),
        cells=len(support),
        rms_m=float(np.sqrt(np.mean(error[inliers] ** 2))),
    )


def fit_footprint(anchor, points_world, size_m):
    """Check the full contact rectangle against independent local cells and refine its plane in metres."""
    size = np.asarray(size_m, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Expected positive contact dimensions along heading and across the road")
    basis = np.asarray(anchor["basis_columns"])
    center = np.asarray(anchor["center_world"])
    local = (np.asarray(points_world) - center) @ basis
    half = size / 2 + 0.05
    sample = local[(abs(local[:, :2]) < half).all(1) & (abs(local[:, 2]) < 0.30)]
    shape = np.array([6, 4])
    cell = np.floor((sample[:, :2] / half + 1) / 2 * shape).astype(int)
    groups, inv, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    supported = counts >= 3
    groups = groups[supported]
    quadrants = (groups[:, 0] >= 3).astype(int) + 2 * (groups[:, 1] >= 2)
    if len(groups) < 16 or len(np.unique(quadrants)) != 4:
        raise ValueError("Insufficient measured road across the asset footprint")
    reps = np.stack([np.median(sample[inv == i], axis=0) for i in np.flatnonzero(supported)])
    z = reps[:, 2]
    design = np.column_stack([reps[:, :2], np.ones(len(reps))])
    slope_x, slope_y, offset = np.linalg.lstsq(design, z, rcond=None)[0]
    residual = z - (reps[:, 0] * slope_x + reps[:, 1] * slope_y + offset)
    spread = float(np.ptp(np.percentile(z, [10, 90])))
    tail = float(np.percentile(abs(residual), 90))
    # A coherent step or a global plane passing above a gutter must not be averaged into a seat.
    if abs(offset) > 0.04 or spread > 0.08 or np.hypot(slope_x, slope_y) > 0.12 or tail > 0.035:
        raise ValueError(f"Footprint crosses a step or an inconsistent road plane (offset {offset:.3f} m, "
                         f"height spread {spread:.3f} m, residual P90 {tail:.3f} m)")
    normal = unit(basis @ [-slope_x, -slope_y, 1])
    forward = unit(basis[:, 0] - normal * (basis[:, 0] @ normal))
    new_basis = np.column_stack([forward, np.cross(normal, forward), normal])
    return dict(anchor, center_world=(center + offset * basis[:, 2]).tolist(), basis_columns=new_basis.tolist(),
                footprint_support=dict(size_m=size.tolist(), cells=len(groups), points=len(sample),
                                       center_adjustment_m=float(offset), cell_height_spread_m=spread,
                                       cell_residual_p90_m=tail))


def footprint_size(folder):
    """Return the centre-symmetric road-contact envelope recorded in the asset cut and height fields."""
    with np.load(Path(folder) / "surface.npz", allow_pickle=False) as field:
        extent = np.array([float(field["width"]), float(field["length"])])
        mask = (field["contact_mask"].astype(bool) if "contact_mask" in field else
                (field["defect_mask"] > 0.5) | (field["height"] > 0.003))
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("No contact footprint in the asset field")
    pos = np.column_stack([x, y]) / (np.array(mask.shape[::-1]) - 1) - 0.5
    return 2 * np.max(abs(pos), axis=0) * extent


def pixel_rays(calibration, *, return_valid=False):
    """Unit optical-frame directions at pixel centres. Supports COLMAP PINHOLE and OPENCV_FISHEYE."""
    model, w, h, fx, fy, cx, cy, *dist = calibration
    if w < 1 or h < 1 or not np.isfinite([fx, fy, cx, cy, *dist]).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid calibration")
    yy, xx = np.indices((h, w), dtype=float)
    x, y = (xx + 0.5 - cx) / fx, (yy + 0.5 - cy) / fy
    if model == "PINHOLE" and not dist:
        out = np.stack([x, y, np.ones_like(x)], -1)
        rays = np.float32(out / np.linalg.norm(out, axis=-1, keepdims=True))
        return (rays, np.ones((h, w), bool)) if return_valid else rays
    if model != "OPENCV_FISHEYE" or len(dist) != 4:
        raise ValueError("Supply calibrated per-pixel unit rays for other camera models")
    radius = np.hypot(x, y)
    angles = np.linspace(0, np.pi, 16385)
    coeff = np.r_[1.0, dist]
    distorted = angles * np.polynomial.polynomial.polyval(angles**2, coeff)
    turn = np.flatnonzero(np.diff(distorted) <= 0)
    end = int(turn[0] + 1) if len(turn) else len(angles)
    valid = radius <= distorted[end - 1]
    if end < 2 or (not return_valid and not valid.all()):
        raise ValueError("Fisheye polynomial has no unique inverse over this image")
    theta = np.interp(radius, distorted[:end], angles[:end])
    scale = np.divide(np.sin(theta), radius, out=np.ones_like(theta), where=radius > 0)
    rays = np.float32(np.stack([x * scale, y * scale, np.cos(theta)], -1))
    return (rays, valid) if return_valid else rays


class RoadMesh:
    def __init__(self, folder, *, filter_pavement=False, measured_rim=False):
        self.filtered_textures = {}
        folder = Path(folder)
        self.digest = sha256(folder / "asset.glb")
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.surfaces = {}
        bounds = []
        with np.load(folder / "surface.npz", allow_pickle=False) as field:
            self.cut = field["defect_mask"].astype(bool)
            self.extent = np.array([float(field["width"]), float(field["length"])])
            self.measured_rim = measured_rim and "view_count" in field
            self.standing = "sky_blocked" in field
            if self.standing:
                self.sky_blocked = field["sky_blocked"]
        self.materials = {}
        self.baked_sky = {}
        self.dielectrics = {}
        for part, xyz, tri, uv, image, *pbr in glb_triangles(folder / "asset.glb", appearance=True, packed_material=self.standing):
            gid = self.scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(tri))
            texture = np.asarray(Image.open(BytesIO(image)).convert("RGB"))
            self.surfaces[gid] = part, tri, uv, texture
            if filter_pavement:
                from pavement_texture import pyramid
                self.filtered_textures[gid] = xyz, pyramid(texture)
            if pbr:
                self.materials[gid] = pbr[0], np.asarray(Image.open(BytesIO(pbr[1])).convert("RGB"))
            if len(pbr) > 2:
                sky, spec = pbr[2]
                if sky is not None: self.baked_sky[gid] = sky
                if spec is not None: self.dielectrics[gid] = spec
            bounds.extend([xyz.min(axis=0), xyz.max(axis=0)])
        self.bounds = np.array([np.min(bounds, axis=0), np.max(bounds, axis=0)])
        self.measured_rim &= {surface[0] for surface in self.surfaces.values()} == {1, 2}

    def sky_gain(self, points, normals):
        """Diffuse skylight reaching the exposed frame, cut edges and well, plus a soft bounce floor."""
        # Uniform solid angle quadrature. Keep directions fixed in road coordinates across exposures.
        i = np.arange(24)
        z = (i + 0.5) / len(i)
        azimuth = i * (np.pi * (3 - np.sqrt(5)))
        sky = np.float32(np.c_[np.cos(azimuth) * np.sqrt(1 - z*z), np.sin(azimuth) * np.sqrt(1 - z*z), z])
        origins = points + normals * 0.0004
        light = np.zeros(len(points), 'f4')
        for direction in sky:
            cosine = normals @ direction
            facing = cosine > 0
            if facing.any():
                rays = np.column_stack([origins[facing], np.broadcast_to(direction, (int(facing.sum()), 3))])
                blocked = self.scene.test_occlusions(o3d.core.Tensor(np.float32(rays)), nthreads=2).numpy()
                light[facing] += cosine[facing] * ~blocked
        return 0.26 + 0.74 * np.clip(light / sky[:, 2].sum(), 0, 1)

    def render(self, rgb, camera_to_world, rays_camera, anchor, host_range, protected, *,
               occluder_range=None, road_tolerance_m=0.10, with_shadow=False, receiver_mask=None, pixel_width=1.0):
        """Compare uncertain host pavement and optional positively identified foreground in radial metres."""
        h, w = rgb.shape[:2]
        pose, basis = np.asarray(camera_to_world), np.asarray(anchor["basis_columns"])
        if (
            rgb.shape != (h, w, 3)
            or rgb.dtype != np.uint8
            or rays_camera.shape != (h, w, 3)
            or host_range.shape != (h, w)
            or protected.shape != (h, w)
            or np.any(host_range <= 0)
            or not np.isfinite(road_tolerance_m)
            or road_tolerance_m < 0
            or (occluder_range is not None and
                (occluder_range.shape != (h, w) or np.any(occluder_range <= 0)))
            or (receiver_mask is not None and (receiver_mask.shape != (h,w) or receiver_mask.dtype != bool))
            or pose.shape != (4, 4)
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.isfinite(pose).all()
        ):
            raise ValueError("Invalid host image, camera pose, unit rays, radial range or foreground mask")
        valid_ray = np.isclose(np.linalg.norm(rays_camera, axis=2), 1, atol=1e-4)
        if not valid_ray.all():
            if np.any(~valid_ray & ~np.asarray(protected, bool)):
                raise ValueError("Unmasked camera ray must be finite and unit length")
            rays_camera = np.where(valid_ray[..., None], rays_camera, np.array([0, 0, 1], 'f4'))
        for frame in (basis, pose[:3, :3]):
            if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-5) or np.linalg.det(frame) < 0.99999:
                raise ValueError("Expected right-handed rigid transforms")
        origin = (pose[:3, 3] - anchor["center_world"]) @ basis
        directions = rays_camera @ (pose[:3, :3].T @ basis)
        rays = np.concatenate([np.broadcast_to(origin, directions.shape), directions], -1).astype("f4")
        hits = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
        distance = hits["t_hit"].numpy()
        geom, faces = hits["geometry_ids"].numpy(), hits["primitive_ids"].numpy()
        weights = hits["primitive_uvs"].numpy()
        part = np.zeros((h, w), "u1")
        color = np.zeros_like(rgb)
        if self.standing:
            normals = np.zeros((*part.shape, 3), 'f4')
            ambient_visibility = np.ones(part.shape, 'f4')
            f0 = np.zeros((*part.shape,3), "f4") if self.dielectrics else None
            rough = np.ones(part.shape, "f4") if self.dielectrics else None
            sky = np.ones(part.shape, 'f4') if self.baked_sky else None
        if self.filtered_textures:
            from pavement_texture import filtered, projected_axes, ray_derivatives
            derivatives = ray_derivatives(directions)
        for gid, (code, tri, uv, texture) in self.surfaces.items():
            mask = geom == gid
            if not mask.any():
                continue
            part[mask] = code
            pair = weights[mask]
            bary = np.column_stack([1 - pair.sum(1), pair])
            texcoord = (uv[tri[faces[mask]]] * bary[:, :, None]).sum(1)
            if gid in self.filtered_textures:
                xyz, levels = self.filtered_textures[gid]
                face = tri[faces[mask]]
                dx, dy = projected_axes(xyz[face], uv[face], directions[mask], distance[mask],
                                        [d[mask] for d in derivatives], texture.shape[1::-1], pixel_width)
                color[mask] = filtered(levels, texcoord, dx, dy)
            else:
                coords = (texcoord * texture.shape[1::-1] - 0.5)[:, ::-1].T
                color[mask] = np.stack(
                    [
                        map_coordinates(channel, coords, order=1, mode="grid-wrap" if self.standing else "nearest")
                        for channel in texture.transpose(2, 0, 1)
                    ],
                    axis=1,
                )
            if self.standing:
                if gid in self.baked_sky:
                    sky[mask] = (self.baked_sky[gid][tri[faces[mask]]] * bary).sum(1)
                verts, arm = self.materials[gid]
                n = (verts[tri[faces[mask]]] * bary[:, :, None]).sum(1)
                normals[mask] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
                rc = (texcoord * arm.shape[1::-1] - .5)[:, ::-1].T
                if gid in self.dielectrics:
                    spec, factor = self.dielectrics[gid]
                    f0[mask] = spec
                    rough[mask] = factor * map_coordinates(arm[...,1],rc,order=1,mode="grid-wrap")/255.
                ambient_visibility[mask] = map_coordinates(arm[..., 0], rc, order=1, mode="grid-wrap") / 255
        # Only a ray entering the cut at pavement height may see buried sidewalls or floor.
        plane = np.divide(
            -origin[2], directions[..., 2], out=np.full((h, w), np.inf), where=directions[..., 2] < -1e-6
        )
        finite = np.isfinite(distance)
        ray_z = origin[2] + directions[..., 2] * np.where(finite, distance, 0)
        crossing = origin[:2] + directions[..., :2] * np.where(np.isfinite(plane), plane, 0)[..., None]
        uv = 0.5 + crossing / self.extent * [1, -1]
        gr = np.rint(np.clip(uv, 0, 1) * (np.array(self.cut.shape[::-1]) - 1)).astype(int)
        surveyed = (uv >= 0).all(-1) & (uv <= 1).all(-1)
        # A continuous measured grid includes the surrounding asphalt and the actual uneven lip.
        # Its first mesh hit establishes visibility. Shaft models instead require a cut at grade.
        aperture = surveyed if self.measured_rim else surveyed & self.cut[gr[..., 1], gr[..., 0]]
        target = (part > 1) & ((ray_z >= -0.0015) | ((plane > 0) & np.isfinite(plane) & aperture))
        if receiver_mask is not None and not self.standing:
            # The host must actually show replaceable road at the entrance to a buried hit.
            target &= (ray_z > 0.0015) | receiver_mask
        # The original pavement will be removed. Only measured returns in front of both
        # the impact and the entry plane may hide the replacement surface.
        near = np.minimum(plane, distance)
        blocked = np.asarray(protected, bool) | (host_range < near - road_tolerance_m)
        if occluder_range is not None:
            # A return positively identified as an object is not replaceable pavement.
            # Compare it with the actual hit, even when it lies inside the opening.
            blocked |= occluder_range < distance - 1e-5
        visible = target & ~blocked
        if self.standing:
            from prop_lighting import compose
            return compose(self, rgb, part, color, normals, ambient_visibility, visible, distance, plane, crossing,
                           host_range, protected, occluder_range, road_tolerance_m, anchor, with_shadow, receiver_mask,
                           origin + directions[visible] * distance[visible,None], None if sky is None else sky[visible],
                           None if f0 is None else (f0[visible],rough[visible],-directions[visible]))
        context = (part == 1) & ~blocked
        out = rgb.copy()
        if visible.any() and context.sum() >= 8:
            src = np.median(color[context], axis=0).clip(20)
            host = np.median(rgb[context], axis=0).clip(10)
            gain = np.clip(host / src, 0.48, 1.85)
            smooth = cv2.GaussianBlur(rgb, (0, 0), 2.5)
            local_light = np.clip(
                (smooth.astype(float) @ [0.2126, 0.7152, 0.0722]) / (host @ [0.2126, 0.7152, 0.0722]),
                0.50,
                1.55,
            )
            ao = np.ones((h, w), 'f4')
            inside = visible & np.isin(part, [5, 6, 7, 8, 10, 11, 20])
            if inside.any():
                directions_hit = directions[inside]
                points = origin + directions_hit * distance[inside, None]
                normals = hits["primitive_normals"].numpy()[inside].copy()
                normals[(normals * directions_hit).sum(1) > 0] *= -1
                ao[inside] = self.sky_gain(points, normals)
            values = color[visible] * gain * (local_light[visible] * ao[visible])[:, None]
            out[visible] = np.uint8(np.clip(np.rint(values), 0, 255))
        else:
            visible[:] = False
        result = out, np.where(visible, part, 0).astype("u1"), np.where(visible, distance, np.nan)
        return (*result, np.zeros((h, w), "f4")) if with_shadow else result


def resolve_samples(samples):
    """Resolve subpixel color coverage, and return the nearest visible surface's part and radial range."""
    after = np.uint8(np.rint(np.mean([v[0] for v in samples], axis=0)))
    ranges = np.stack([np.where(np.isfinite(v[2]), v[2], np.inf) for v in samples])
    nearest = np.argmin(ranges, axis=0)[None]
    distance = np.take_along_axis(ranges, nearest, axis=0)[0]
    part = np.take_along_axis(np.stack([v[1] for v in samples]), nearest, axis=0)[0]
    distance[part == 0] = np.nan
    if len(samples[0]) == 4:
        return after, part, distance, np.mean([s[3] for s in samples], axis=0)
    return after, part, distance
