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


def pixel_rays(calibration):
    """Unit optical-frame directions at pixel centres. Supports COLMAP PINHOLE and OPENCV_FISHEYE."""
    model, w, h, fx, fy, cx, cy, *dist = calibration
    if w < 1 or h < 1 or not np.isfinite([fx, fy, cx, cy, *dist]).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid calibration")
    yy, xx = np.indices((h, w), dtype=float)
    x, y = (xx + 0.5 - cx) / fx, (yy + 0.5 - cy) / fy
    if model == "PINHOLE" and not dist:
        out = np.stack([x, y, np.ones_like(x)], -1)
        return np.float32(out / np.linalg.norm(out, axis=-1, keepdims=True))
    if model != "OPENCV_FISHEYE" or len(dist) != 4:
        raise ValueError("Supply calibrated per-pixel unit rays for other camera models")
    radius = np.hypot(x, y)
    angles = np.linspace(0, np.pi, 16385)
    coeff = np.r_[1.0, dist]
    distorted = angles * np.polynomial.polynomial.polyval(angles**2, coeff)
    turn = np.flatnonzero(np.diff(distorted) <= 0)
    end = int(turn[0] + 1) if len(turn) else len(angles)
    if end < 2 or radius.max() > distorted[end - 1]:
        raise ValueError("Fisheye polynomial has no unique inverse over this image")
    theta = np.interp(radius, distorted[:end], angles[:end])
    scale = np.divide(np.sin(theta), radius, out=np.ones_like(theta), where=radius > 0)
    return np.float32(np.stack([x * scale, y * scale, np.cos(theta)], -1))


class RoadMesh:
    def __init__(self, folder):
        folder = Path(folder)
        self.digest = sha256(folder / "asset.glb")
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.surfaces = {}
        with np.load(folder / "surface.npz", allow_pickle=False) as field:
            self.cut = field["defect_mask"].astype(bool)
            self.extent = np.array([float(field["width"]), float(field["length"])])
        for part, xyz, tri, uv, image in glb_triangles(folder / "asset.glb", appearance=True):
            gid = self.scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(tri))
            texture = np.asarray(Image.open(BytesIO(image)).convert("RGB"))
            self.surfaces[gid] = part, tri, uv, texture

    def render(self, rgb, camera_to_world, rays_camera, anchor, host_range, protected):
        """host_range is radial metres. NaNs denote unobserved range. protected is a hard foreground mask."""
        h, w = rgb.shape[:2]
        pose, basis = np.asarray(camera_to_world), np.asarray(anchor["basis_columns"])
        if (
            rgb.shape != (h, w, 3)
            or rgb.dtype != np.uint8
            or rays_camera.shape != (h, w, 3)
            or host_range.shape != (h, w)
            or protected.shape != (h, w)
            or np.any(host_range <= 0)
            or pose.shape != (4, 4)
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.isfinite(pose).all()
            or not np.allclose(np.linalg.norm(rays_camera, axis=2), 1, atol=1e-4)
        ):
            raise ValueError("Invalid host image, camera pose, unit rays, radial range or foreground mask")
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
        for gid, (code, tri, uv, texture) in self.surfaces.items():
            mask = geom == gid
            if not mask.any():
                continue
            part[mask] = code
            pair = weights[mask]
            bary = np.column_stack([1 - pair.sum(1), pair])
            texcoord = (uv[tri[faces[mask]]] * bary[:, :, None]).sum(1)
            coords = (texcoord * texture.shape[1::-1] - 0.5)[:, ::-1].T
            color[mask] = np.stack(
                [
                    map_coordinates(channel, coords, order=1, mode="nearest")
                    for channel in texture.transpose(2, 0, 1)
                ],
                axis=1,
            )
        # Only a ray entering the cut at pavement height may see buried sidewalls or floor.
        plane = np.divide(
            -origin[2], directions[..., 2], out=np.full((h, w), np.inf), where=directions[..., 2] < -1e-6
        )
        finite = np.isfinite(distance)
        ray_z = origin[2] + directions[..., 2] * np.where(finite, distance, 0)
        crossing = origin[:2] + directions[..., :2] * np.where(np.isfinite(plane), plane, 0)[..., None]
        uv = 0.5 + crossing / self.extent * [1, -1]
        gr = np.rint(np.clip(uv, 0, 1) * (np.array(self.cut.shape[::-1]) - 1)).astype(int)
        aperture = (uv >= 0).all(-1) & (uv <= 1).all(-1) & self.cut[gr[..., 1], gr[..., 0]]
        target = (part > 1) & ((ray_z >= -0.0015) | ((plane > 0) & np.isfinite(plane) & aperture))
        # The original pavement will be removed. Only measured returns in front of both
        # the impact and the entry plane may hide the replacement surface.
        near = np.minimum(plane, distance)
        blocked = np.asarray(protected, bool) | (host_range < near - 0.10)
        visible = target & ~blocked
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
            # The photographed pit floor already contains texture shading. Reduce light
            # only on newly exposed shaft lining, measured relative to the rim datum.
            ao = np.where(np.isin(part, [6, 7, 10]), np.clip(0.71 + 0.66 * ray_z, 0.24, 0.80), 1.0)
            values = color[visible] * gain * (local_light[visible] * ao[visible])[:, None]
            out[visible] = np.uint8(np.clip(np.rint(values), 0, 255))
        else:
            visible[:] = False
        return out, np.where(visible, part, 0).astype("u1"), np.where(visible, distance, np.nan)


def resolve_samples(samples):
    """Resolve subpixel color coverage, and return the nearest visible surface's part and radial range."""
    after = np.uint8(np.rint(np.mean([v[0] for v in samples], axis=0)))
    ranges = np.stack([np.where(np.isfinite(v[2]), v[2], np.inf) for v in samples])
    nearest = np.argmin(ranges, axis=0)[None]
    distance = np.take_along_axis(ranges, nearest, axis=0)[0]
    part = np.take_along_axis(np.stack([v[1] for v in samples]), nearest, axis=0)[0]
    distance[part == 0] = np.nan
    return after, part, distance
