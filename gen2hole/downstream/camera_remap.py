"""Resample a metric host packet at nearest raw pixel centres after registering the exposure."""

import numpy as np

from host_depth import geometry, sampled, sun_samples


def remap_packet(view, source_xy, calibration=None):
    """Map each output pixel to one complete source pixel, retaining all its optical subrays."""
    xy = np.asarray(source_xy)
    if (xy.ndim != 3 or xy.shape[-1] != 2 or not all(xy.shape[:2])
            or xy.dtype.kind not in "fiu"):
        raise ValueError("source_xy must be a nonempty Hout×Wout×2 real array of source pixel coordinates")
    rays, distance, protected, pose, origins = geometry(view, calibration, with_origins=True)
    samples, h, w = distance.shape
    rgb = np.asarray(view["rgb"])
    if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected aligned uint8 RGB")
    # Classify before casting. This also allows NaNs and huge sentinels from lens solvers.
    valid = np.isfinite(xy).all(-1) & ((xy >= -0.5) & (xy < [w - 0.5, h - 0.5])).all(-1)
    centres = np.where(valid[..., None], xy, 0.0).astype(np.float64)
    indices = np.floor(centres + 0.5).astype(np.intp)
    # At the last half-pixel, floating addition can round up to the extent.
    x, y = np.moveaxis(np.clip(indices, 0, [w - 1, h - 1]), -1, 0)

    def pick(field, border):
        data = field[:, y, x].copy()
        data[:, ~valid] = border
        return data[0] if samples == 1 else data

    rgb_out = rgb[y, x].copy()
    rgb_out[~valid] = 0
    out = dict(rgb=rgb_out, camera_to_world=pose, rays_camera=pick(rays, [0., 0., 1.]),
               host_range=pick(distance.astype(np.result_type(distance, np.float32), copy=False), np.inf),
               protected=pick(protected, True))
    if "sample_rgb" in view:
        from host_exposure import exposure
        colors, _ = exposure(view, distance.shape)
        out["sample_rgb"] = pick(colors, 0).reshape(samples, *valid.shape, 3)
        if "sample_weights" in view:
            wgt = view["sample_weights"]
            out["sample_weights"] = wgt if wgt.ndim == 1 else pick(wgt, 1).reshape(samples, *valid.shape)
    if origins is not None:
        out["ray_origins_camera"] = pick(origins, 0.0)
    for name, border in (("receiver_mask", False), ("occluder_range", np.inf)):
        if name in view:
            value = sampled(view[name], distance.shape, name)
            if name == "receiver_mask" and value.dtype != bool:
                raise ValueError("receiver_mask must be boolean")
            if name == "occluder_range":
                value = value.astype(np.result_type(value, np.float32), copy=False)
            out[name] = pick(value, border)
    if "sun_visibility" in view:
        visibility = pick(sun_samples(view, distance.shape), 0)
        if samples > 1 and view["sun_visibility"].ndim == 2:
            visibility = visibility[0]
        out["sun_visibility"] = visibility
    if "reflector_labels" in view:
        out["reflector_labels"] = pick(sampled(view["reflector_labels"], distance.shape, "reflector_labels"), 0)
    if "road_albedo_srgb" in view:
        albedo = view["road_albedo_srgb"]
        if albedo.shape != rgb.shape or albedo.dtype != np.uint8:
            raise ValueError("road_albedo_srgb must match the uint8 RGB image")
        out["road_albedo_srgb"] = albedo[y, x].copy()
        out["road_albedo_srgb"][~valid] = 0
    for name in ("sensor_name", "receiver_band_m", "reflection_rays_world", "reflection_srgb", "reflection_radius_deg",
                 "reflector_vertices_world", "reflector_triangles", "reflector_srgb"):
        if name in view:
            out[name] = view[name]
    return out
