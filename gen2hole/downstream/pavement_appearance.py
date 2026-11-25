"""Register exposure on intact pavement around a cut and reuse it at the same metric location."""

from pathlib import Path

import numpy as np
from prop_lighting import linear, srgb


def illumination_field(points, residual):
    """Interpolate locally supported shade when neighbouring road cells predict one another."""
    delta = points[:, None] - points[None, :]
    dist2 = (delta * delta).sum(-1)
    constant = np.mean(np.minimum(abs(residual), 0.8))
    best = None
    for sigma in (0.10, 0.18, 0.28):
        kernels = np.exp(-dist2 / (2 * sigma**2))
        np.fill_diagonal(kernels, 0.0)
        # Withhold this cell's residual when selecting the spatial bandwidth.
        estimate = (kernels @ residual) / np.maximum(kernels.sum(1), 1.0)
        error = np.mean(np.minimum(abs(residual - estimate), 0.8))
        if (
            error < 0.83 * constant
            and constant - error > 0.025
            and (best is None or error < best[0])
        ):
            best = (error, sigma)
    if best is None:
        return {}
    error, sigma = best
    # Limit isolated paint or dark material before storing the final illumination field.
    order = np.argsort(dist2, axis=1)[:, 1:5]
    middle = np.median(residual[order], axis=1)
    clean = np.clip(residual, middle - 0.30, middle + 0.30)
    axis = np.linspace(-0.5, 0.5, 25)
    x, y = np.meshgrid(axis, axis)
    grid = np.stack([x, y], axis=-1)
    k = np.exp(-((grid[:, :, None] - points) ** 2).sum(-1) / (2 * sigma**2))
    field = np.clip((k @ clean) / np.maximum(k.sum(-1), 1.0), -0.85, 0.85)
    return {
        "shade_grid": np.round(field, 5).tolist(),
        "shade_sigma": float(sigma),
        "constant_cell_mae": float(constant),
        "leave_cell_out_mae": float(error),
    }


def fit_apron(xy, source, target, extent):
    """Fit RGB exposure and one shared luminance slope, giving equal weight to spatial cells."""
    xy = np.asarray(xy, dtype=float) / np.asarray(extent)
    source, target = np.asarray(source), np.asarray(target)
    valid = (
        np.isfinite(xy).all(1) & (source.min(axis=1) >= 12) & (target.min(axis=1) >= 8)
    )
    valid &= (source.max(axis=1) <= 249) & (target.max(axis=1) <= 249)
    xy = xy[valid]
    ratios = np.log(linear(target[valid]) / linear(source[valid]))
    cell = np.floor((xy + 0.5) * 10).astype(int)
    _, index = np.unique(cell, axis=0, return_inverse=True)
    points, colors = [], []
    for i, n in enumerate(np.bincount(index)):
        if n >= 4:
            m = index == i
            points.append(np.median(xy[m], axis=0))
            colors.append(np.median(ratios[m], axis=0))
    if len(points) < 6:
        raise ValueError(
            "Insufficient visible intact pavement; register a wider or less occluded view"
        )
    points, colors = np.array(points), np.array(colors)
    slope = np.zeros(2)
    intercept = np.median(colors, axis=0)
    # Poorly spread samples can determine exposure but cannot support an extrapolated gradient.
    if np.linalg.svd(points - points.mean(0), compute_uv=False)[-1] > 0.09 * np.sqrt(
        len(points)
    ):
        for _ in range(6):
            residual = colors - intercept - (points @ slope)[:, None]
            errors = np.max(np.abs(residual), axis=1)
            width = max(0.055, 1.5 * np.median(errors))
            weights = np.minimum(1.0, width / np.maximum(errors, 1e-8)) ** 2
            centre = np.average(points, axis=0, weights=weights)
            delta = points - centre
            grey = np.median(colors - intercept, axis=1)
            slope = np.linalg.lstsq(
                delta * np.sqrt(weights[:, None]), grey * np.sqrt(weights), rcond=None
            )[0]
            intercept = np.median(colors - (points @ slope)[:, None], axis=0)
    shade = illumination_field(
        points, np.median(colors - intercept, axis=1) - points @ slope
    )
    return dict(
        **shade,
        log_gain_rgb=intercept.tolist(),
        log_gain_slope=slope.tolist(),
        extent_m=list(map(float, extent)),
        cells=len(points),
        pixels=int(valid.sum()),
        method="intact-apron-log-linear-v1",
    )


def apply_apron(source, xy, model, visibility=1.0):
    gain, slope, extent = (
        np.asarray(model[k], dtype=float)
        for k in ("log_gain_rgb", "log_gain_slope", "extent_m")
    )
    if (
        model.get("method") != "intact-apron-log-linear-v1"
        or gain.shape != (3,)
        or slope.shape != (2,)
        or extent.shape != (2,)
        or not np.isfinite(np.r_[gain, slope, extent]).all()
        or extent.min() <= 0
    ):
        raise ValueError("Invalid registered pavement appearance")
    position = np.clip(xy / extent, -0.5, 0.5)
    light = np.clip(position @ slope, -0.7, 0.7)
    if "shade_grid" in model:
        from scipy.ndimage import map_coordinates

        grid = np.asarray(model["shade_grid"])
        if grid.shape != (25, 25) or not np.isfinite(grid).all():
            raise ValueError("Invalid surveyed illumination field")
        light += map_coordinates(
            grid, ((position + 0.5) * 24)[:, ::-1].T, order=1, mode="nearest"
        )
    exposure = np.exp(np.clip(gain + light[:, None], -2.3, 2.3))
    return srgb(linear(source) * exposure * np.asarray(visibility).reshape(-1, 1))


def camera_anchor(anchor, camera):
    refs = anchor.get("pavement_appearance_by_camera")
    if refs is None:
        return anchor
    if camera not in refs:
        raise ValueError(
            "Select a registered camera for this exposure or recalibrate its appearance"
        )
    return apply_models(anchor, refs[camera])


def apply_models(anchor, models):
    if set(models) != {s["id"] for s in anchor["instances"]}:
        raise ValueError("Appearance instance IDs differ from the surveyed layout")
    return dict(
        anchor,
        instances=[
            dict(s, anchor=dict(s["anchor"], pavement_appearance=models[s["id"]]))
            for s in anchor["instances"]
        ],
    )


def register_views(layout, anchor, paths, calibration=None, sun_survey=None):
    """Derive each instance's local road transfer from unobstructed reference exposures."""
    from host_depth import geometry, sampled, sun_samples

    layout.bind(anchor)
    observations = {s["id"]: [] for s in anchor["instances"]}
    grade = {key: [] for key in observations}
    albedo_mode = None
    for path in paths:
        with np.load(Path(path), allow_pickle=False) as packet:
            rays, host, protected, pose, origins = geometry(
                packet, calibration, with_origins=True
            )
            direct = None
            if sun_survey is not None and "sun_visibility" in packet:
                direct = sun_samples(packet, host.shape)
            rgb = packet["rgb"]
            albedo = packet.get("road_albedo_srgb")
            if albedo is not None and (
                albedo.shape != rgb.shape or albedo.dtype != np.uint8
            ):
                raise ValueError(
                    "Expected aligned, unlit H×W×3 uint8 sRGB road_albedo_srgb"
                )
            has_albedo = albedo is not None
            if albedo_mode is not None and albedo_mode != has_albedo:
                raise ValueError(
                    "Provide the diffuse albedo in every calibration packet or in none"
                )
            albedo_mode = has_albedo
            occluder = (
                sampled(packet["occluder_range"], host.shape, "occluder_range")
                if "occluder_range" in packet
                else None
            )
            receiver = (
                sampled(packet["receiver_mask"], host.shape, "receiver_mask")
                if "receiver_mask" in packet
                else None
            )
            from host_exposure import exposure

            colors, _ = exposure(packet, host.shape)
            # A resolved photograph supplies one observation; a sampled exposure supplies one per ray.
            images = [rgb] if colors is None else colors
            if (
                colors is None
                and direct is not None
                and packet["sun_visibility"].ndim == 3
            ):
                images = np.broadcast_to(rgb, (*host.shape, 3))
            for i, image in enumerate(images):
                for s in anchor["instances"]:
                    mesh = layout.meshes[s["asset"]]
                    result = mesh.render(
                        image,
                        pose,
                        rays[i],
                        s["anchor"],
                        host[i],
                        protected[i],
                        ray_origins_camera=None if origins is None else origins[i],
                        apron_samples=True,
                        grade_samples=sun_survey is not None,
                        receiver_mask=None if receiver is None else receiver[i],
                        occluder_range=None if occluder is None else occluder[i],
                        pixel_width=len(rays) ** -0.5,
                    )
                    xy, source, context = result[:3]
                    if sun_survey is not None:
                        points, mask = result[3:]
                        grade[s["id"]].append(
                            (
                                points,
                                image[mask],
                                None if direct is None else direct[i][mask],
                            )
                        )
                    if len(xy):
                        base = albedo[context] if has_albedo else source
                        observations[s["id"]].append((xy, base, image[context]))
    result = {}
    for s in anchor["instances"]:
        batches = observations[s["id"]]
        if not batches:
            raise ValueError("No visible intact pavement around " + s["id"])
        result[s["id"]] = fit_apron(
            *(np.concatenate(v) for v in zip(*batches)),
            layout.meshes[s["asset"]].extent,
        )
        if sun_survey is not None:
            from pavement_sun import fit_sun

            mesh = layout.meshes[s["asset"]]
            result[s["id"]]["sunlight"] = fit_sun(
                sun_survey,
                s["anchor"],
                mesh.extent,
                np.concatenate([v[1] for v in batches]),
                grade[s["id"]],
            )
        if albedo_mode:
            result[s["id"]]["reference_material"] = "host_diffuse_albedo"
    return result
