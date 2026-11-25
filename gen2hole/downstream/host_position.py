"""Register unfiltered Cartesian hit positions from another renderer to individual camera samples."""

import numpy as np

from host_depth import optical_samples, reference_samples, sampled, sun_samples
from host_exposure import exposure


def register_positions(view, tolerance=0.001, calibration=None):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max-ray-error-m must be finite and nonnegative")
    if any(k in view for k in ("host_range", "host_z", "occluder_range")):
        raise ValueError(
            "Only one source of depth is allowed in a surface-position exposure"
        )
    rays, protected, origins = optical_samples(view, calibration)
    pose, rays, origins = reference_samples(view, rays, origins, with_origins=True)
    shape = rays.shape[:-1]
    xyz = np.asarray(view["surface_position"])
    if xyz.shape == rays.shape[1:] and len(rays) == 1:
        xyz = xyz[None]
    if xyz.shape != rays.shape or xyz.dtype.kind != "f":
        raise ValueError("Provide one floating XYZ hit per optical sample")
    known = sampled(view["position_valid"], shape, "position_valid")
    if known.dtype != bool or np.any(known & ~np.isfinite(xyz).all(-1)):
        raise ValueError("position_valid must identify finite Cartesian hits")
    transform = np.asarray(view.get("world_from_position", np.eye(4)), dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
    ):
        raise ValueError("Expected a finite affine world_from_position")
    basis = transform[:3, :3]
    scale = np.linalg.norm(basis[:, 0])
    if not np.isfinite(scale) or scale <= 0 or not np.allclose(
        (basis / scale).T @ (basis / scale), np.eye(3), atol=1e-6, rtol=0
    ):
        raise ValueError("Position axes must be orthogonal and share one metric scale")
    use = known & ~protected
    rgb = np.asarray(view["rgb"])
    if rgb.shape != (*shape[1:], 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected a registered uint8 sRGB image")
    exposure(view, shape)
    # Subtract the reference-camera translation before adding local positions.
    # Keeping large global offsets in float64 preserves small camera-relative distances.
    delta = xyz[use].astype(np.float64) @ basis.T + (transform[:3, 3] - pose[:3, 3])
    delta = delta @ pose[:3, :3]
    if origins is not None:
        delta -= origins[use]
    unit = rays[use].astype(np.float64)
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    radial = np.sum(delta * unit, axis=-1)
    error = np.linalg.norm(delta - radial[:, None] * unit, axis=-1)
    if not np.isfinite(radial).all() or not np.isfinite(error).all():
        raise ValueError("Position transform produced a nonfinite metric distance")
    if (radial <= 0).any():
        raise ValueError(
            "Position hit lies at or behind the origin along its optical ray"
        )
    if (error > tolerance).any():
        raise ValueError(
            f"Position does not register to its ray (max lateral residual {error.max():.6g} m)"
        )
    # Neighbours still supply ray differentials at a protected boundary.
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    directions = np.empty_like(rays, dtype=np.result_type(rays, np.float32))
    directions[:] = [0, 0, 1]
    np.divide(rays, norm, out=directions, where=np.isfinite(norm) & (norm > 0))
    directions[use] = unit
    distance = np.full(shape, np.inf)
    distance[use] = radial
    out = dict(
        rgb=rgb,
        camera_to_world=pose,
        rays_camera=directions,
        protected=~use,
        host_range=distance,
    )
    if origins is not None:
        out["ray_origins_camera"] = origins
    if "sun_visibility" in view:
        vis = sun_samples(view, shape)
        out["sun_visibility"] = vis[0] if view["sun_visibility"].ndim == 2 else vis
    if "road_albedo_srgb" in view:
        albedo = np.asarray(view["road_albedo_srgb"])
        if albedo.shape != rgb.shape or albedo.dtype != np.uint8:
            raise ValueError("Expected road_albedo_srgb registered to the RGB image")
        out["road_albedo_srgb"] = albedo
    if "reflector_labels" in view:
        labels = sampled(view["reflector_labels"], shape, "reflector_labels")
        if labels.dtype.kind not in "iu" or (labels < 0).any():
            raise ValueError("Expected nonnegative integer reflector labels")
        out["reflector_labels"] = np.where(use, labels, 0)
    for name in (
        "sample_rgb",
        "sample_weights",
        "sensor_name",
        "receiver_band_m",
        "reflection_rays_world",
        "reflection_srgb",
        "reflection_radius_deg",
        "reflector_vertices_world",
        "reflector_triangles",
        "reflector_srgb",
    ):
        if name in view:
            out[name] = view[name]
    road = None
    for name in ("receiver_mask", "occluder_mask"):
        if name in view:
            mask = sampled(view[name], shape, name)
            if mask.dtype != bool or (mask & ~known).any():
                raise ValueError(f"{name} must label registered hits")
            if name == "receiver_mask":
                road = mask
                out[name] = mask & use
            else:
                if road is not None and (mask & road).any():
                    raise ValueError("Road and occluder labels overlap")
                out["occluder_range"] = np.where(mask, distance, np.inf)
    return out
