"""Light freestanding objects and their host-ground shadow receivers in the same metric frame."""

import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, map_coordinates


def linear(rgb):
    s = np.asarray(rgb, dtype="f4") / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def srgb(values):
    v = np.clip(values, 0, 1)
    return np.uint8(np.rint(np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055) * 255))


def fit_receivers(points, anchor, radius=3.6, pitch=0.12):
    """Map nearby road samples to independently supported pavement. Preserve measured steps."""
    side = int(np.ceil(radius / pitch))
    n = side * 2 + 1
    local = (points - anchor["center_world"]) @ np.asarray(anchor["basis_columns"])
    local = local[(abs(local[:, :2]) < radius).all(1) & (abs(local[:, 2]) < 0.32)]
    ij = np.rint(local[:, :2] / pitch).astype(int) + side
    flat = ij[:, 1] * n + ij[:, 0]
    total = np.bincount(flat, minlength=n * n).reshape(n, n)
    near = np.bincount(flat[abs(local[:, 2]) < 0.047], minlength=n * n).reshape(n, n)
    good = (total >= 2) & (near >= total * 0.7)
    bad = (total >= 2) & (near < total * 0.5)
    if not good.any():
        raise ValueError("No measured receiver ground near the object")
    dg = distance_transform_edt(~good) * pitch
    db = distance_transform_edt(~bad) * pitch if bad.any() else np.full_like(dg, np.inf)
    confidence = np.clip((0.23 - dg) / 0.12, 0, 1) * np.clip((db - dg) / 0.1, 0, 1)
    return dict(
        pitch_m=pitch,
        size=n,
        weights_u8=np.uint8(confidence * 255 + 0.5).tolist(),
        measured_cells=int(good.sum()),
        conflicting_cells=int(bad.sum()),
    )


def geometry_light(anchor):
    spec = anchor.get("lighting", {})
    color = np.asarray(spec.get("road_srgb", [100.0, 100.0, 100.0]), float)
    fraction = float(spec.get("direct_fraction", 0))
    if (
        color.shape != (3,)
        or not np.isfinite(color).all()
        or np.any((color < 0) | (color > 255))
        or not 0 <= fraction <= 0.9
    ):
        raise ValueError("Expected RGB reference in [0,255] and direct fraction in [0,.9]")
    sun = None
    if fraction:
        direction = np.asarray(spec["sun_world"], float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError("Expected a nonzero finite world-space sun direction")
        sun = direction @ np.asarray(anchor["basis_columns"])
        sun = sun / np.linalg.norm(sun)
        if not np.isfinite(sun).all() or sun[2] < 0.22:
            raise ValueError("The sun must lie above the fitted plane with a supported elevation")
    return color, fraction, sun


def view_bounds(mesh, anchor):
    bounds = mesh.bounds.copy()
    if mesh.standing:
        _, fraction, sun = geometry_light(anchor)
        bounds[0, :2] = np.minimum(bounds[0, :2], -mesh.extent / 2)
        bounds[1, :2] = np.maximum(bounds[1, :2], mesh.extent / 2)
        if fraction:
            shift = -sun[:2] / sun[2] * mesh.bounds[1, 2]
            bounds[0, :2] = np.minimum(bounds[0, :2], mesh.bounds[0, :2] + shift - 0.04)
            bounds[1, :2] = np.maximum(bounds[1, :2], mesh.bounds[1, :2] + shift + 0.04)
    bounds[0, 2] = min(-0.0015, bounds[1, 2])
    return bounds


def irradiance(normals, visibility, anchor, direct_visibility=1.0):
    color, direct, sun = geometry_light(anchor)
    ref = linear(color)
    level = np.clip((ref @ [0.2126, 0.7152, 0.0722]) / linear(100), 0.22, 2.8)
    balance = np.clip((ref + 0.009) / (ref.mean() + 0.009), 0.80, 1.20)
    diffuse = (0.39 + 0.61 * np.clip((1 + normals[:, 2]) * 0.5, 0, 1)) * (0.50 + 0.50 * visibility)
    if direct:
        diffuse = (1 - direct) * diffuse + direct * np.clip(normals @ sun, 0, 1) / sun[2] * direct_visibility
    return level * balance * diffuse[:, None]


def blocked_sun(mesh, xyz, sun):
    # A small source disk softens the geometric terminator over finite solar/pose uncertainty.
    east = np.cross(sun, [0.0, 1.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(sun, east)
    blocked = np.zeros(len(xyz), "f4")
    for dx, dy in ((0, 0), (0.012, 0), (-0.012, 0), (0, 0.012), (0, -0.012)):
        direction = np.float32(sun + dx * east + dy * north)
        rays = np.column_stack([xyz, np.broadcast_to(direction, xyz.shape)]).astype("f4")
        blocked += mesh.scene.test_occlusions(o3d.core.Tensor(rays), nthreads=2).numpy()
    return blocked / 5


def ground_confidence(anchor, xy):
    ground = anchor["ground_receivers"]
    raster = np.asarray(ground["weights_u8"], "f4") / 255.0
    coords = xy / ground["pitch_m"] + (ground["size"] - 1) / 2
    return map_coordinates(raster, coords[:, ::-1].T, order=1, mode="constant")


def ground_loss(mesh, xy, anchor):
    """Fraction of linear road radiance lost at known receivers on the fitted plane."""
    _, direct, sun = geometry_light(anchor)
    uv = xy / mesh.extent * [1, -1] + 0.5
    ij = uv * (np.array(mesh.sky_blocked.shape[::-1]) - 1)
    loss = (1 - direct) * 0.78 * map_coordinates(mesh.sky_blocked, ij[:, ::-1].T, order=1, mode="constant")
    if direct:
        sweep = -sun[:2] / sun[2] * mesh.bounds[1, 2]
        lower = mesh.bounds[0, :2] + np.minimum(sweep, 0) - 0.04
        upper = mesh.bounds[1, :2] + np.maximum(sweep, 0) + 0.04
        active = (xy >= lower).all(1) & (xy <= upper).all(1)
        if active.any():
            points = np.column_stack([xy[active], np.full(active.sum(), 0.002)])
            loss[active] += direct * blocked_sun(mesh, points, sun)
    return loss


def compose(
    mesh,
    rgb,
    part,
    color,
    normals,
    ambient,
    visible,
    distance,
    plane,
    crossing,
    host,
    protected,
    occluder,
    tolerance,
    anchor,
    with_shadow,
    receiver_mask,
    visible_xyz,
):
    _, direct, sun = geometry_light(anchor)
    out = rgb.copy()
    shadow = np.zeros(part.shape, "f4")
    if with_shadow:
        eligible = ~np.asarray(protected, bool) & ~visible & np.isfinite(plane) & (plane > 0)
        eligible &= ~(host < plane - tolerance)
        if occluder is not None:
            eligible &= ~(occluder < plane - 1e-5)
        confidence = np.zeros(part.shape, "f4")
        if receiver_mask is not None:
            confidence[:] = receiver_mask
        elif "ground_receivers" in anchor:
            confidence[eligible] = ground_confidence(anchor, crossing[eligible])
        else:
            match = np.isfinite(host) & eligible
            confidence[match] = abs(host[match] - plane[match]) <= tolerance
        eligible &= confidence > 0
        shadow[eligible] = ground_loss(mesh, crossing[eligible], anchor) * confidence[eligible]
        apply = shadow >= 0.004
        shadow[~apply] = 0
        out[apply] = srgb(linear(rgb[apply]) * (1 - shadow[apply, None]))
    if visible.any():
        n = normals[visible]
        direct_vis = 1 - blocked_sun(mesh, visible_xyz + n * 0.001, sun) if direct else 1.0
        light = irradiance(n, ambient[visible], anchor, direct_vis)
        out[visible] = srgb(linear(color[visible]) * light)
    result = out, np.where(visible, part, 0).astype("u1"), np.where(visible, distance, np.nan)
    return (*result, shadow) if with_shadow else result
