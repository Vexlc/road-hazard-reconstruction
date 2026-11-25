"""Convert six co-located 90 degree raster cameras to a calibrated nearest-texel exposure."""

import numpy as np

from camera_readout import check_rigid
from host_depth import sampled
from host_exposure import integrate, weights


def directions(faces):
    frames = np.asarray(faces, dtype=np.float64)
    if frames.shape != (6, 3, 3) or not np.isfinite(frames).all():
        raise ValueError(
            "Supply six face_to_cube rotations with right/down/forward columns"
        )
    relative = frames[0].T @ frames
    axes = np.rint(relative)
    if (
        not np.allclose(frames.swapaxes(-2, -1) @ frames, np.eye(3), atol=1e-6, rtol=0)
        or np.any(np.linalg.det(frames) < 0.999999)
        or not np.allclose(relative, axes, atol=1e-6, rtol=0)
        or len(np.unique(axes[:, :, 2], axis=0)) != 6
    ):
        raise ValueError(
            "Face cameras must form the six axis-aligned directions of one cube"
        )
    return frames


def convert(cube, lens):
    rgb = np.asarray(cube["rgb"])
    if (
        rgb.ndim != 4
        or rgb.shape[0] != 6
        or rgb.shape[1] != rgb.shape[2]
        or not rgb.shape[1]
        or rgb.shape[3] != 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError("Use six square uint8 sRGB faces, ordered like face_to_cube")
    size = rgb.shape[1]
    depth = np.asarray(cube["device_depth"])
    known = np.asarray(cube["depth_valid"])
    if (
        depth.shape != rgb.shape[:3]
        or depth.dtype.kind != "f"
        or known.shape != depth.shape
        or known.dtype != bool
        or not np.isfinite(depth[known]).all()
        or np.any((depth[known] < 0) | (depth[known] > 1))
    ):
        raise ValueError(
            "Supply normalized window depth in [0,1] and an explicit boolean depth_valid"
        )
    near, far, reversed_z = (
        np.asarray(cube[k]) for k in ("near_m", "far_m", "reversed_z")
    )
    if (
        near.shape
        or far.shape
        or reversed_z.shape
        or reversed_z.dtype != bool
        or not np.isfinite(near)
        or not (0 < near < far)
    ):
        raise ValueError(
            "Supply 0 < near_m < far_m in metres and scalar boolean reversed_z"
        )
    frames = directions(cube["face_to_cube"])
    reference, target = (
        np.asarray(cube["camera_to_world"]),
        np.asarray(lens["camera_to_world"]),
    )
    if reference.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("Expected one world pose each for the cube and target lens")
    check_rigid(reference)
    check_rigid(target)
    if (
        not np.allclose(reference[:3, 3], target[:3, 3], atol=1e-7, rtol=0)
        or "camera_to_world_rows" in lens
        or "row_time_ns" in lens
        or ("ray_origins_camera" in lens and np.any(lens["ray_origins_camera"] != 0))
    ):
        raise ValueError("These six faces represent one time and one optical centre")
    rays = np.asarray(lens["rays_camera"])
    if rays.ndim == 3:
        rays = rays[None]
    if (
        rays.ndim != 4
        or rays.shape[-1] != 3
        or min(rays.shape[:-1]) < 1
        or rays.dtype.kind != "f"
    ):
        raise ValueError("Expected H×W×3 or S×H×W×3 floating target rays")
    protected = sampled(lens["protected"], rays.shape[:-1], "protected")
    valid = np.isclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-4)
    if protected.dtype != bool or np.any(~protected & ~valid):
        raise ValueError("Use unit directions on the unmasked lens")
    lens_to_cube = target[:3, :3].T @ reference[:3, :3]
    rays_cube = np.where(valid[..., None], rays, [0.0, 0.0, 1.0]) @ lens_to_cube
    face = np.argmax(rays_cube @ frames[:, :, 2].T, axis=-1)
    address = lens.get("cube_texel")
    if address is not None:
        address = np.asarray(address)
        if address.shape == rays.shape[1:] and len(rays) == 1:
            address = address[None]
        if address.shape != rays.shape or address.dtype.kind not in "iu":
            raise ValueError(
                "cube_texel must give integer (face, column, row) per target ray"
            )
        if np.any((address < 0) | (address >= np.array([6, size, size]))):
            raise ValueError("Cube texel address outside framebuffer")
        face = address[..., 0]
    # Work by face to avoid one 3×3 double matrix for every output subpixel.
    col = np.empty(face.shape, np.intp)
    row = np.empty_like(col)
    actual = np.empty(rays.shape, np.float32)
    scale = np.empty(face.shape, np.float64)
    for index, basis in enumerate(frames):
        mask = face == index
        local = rays_cube[mask] @ basis
        if address is not None and np.any((local[:, 2] <= 0) & ~protected[mask]):
            raise ValueError("Cube texel faces away from the target bearing")
        uv = np.divide(
            local[:, :2],
            local[:, 2:],
            out=np.zeros_like(local[:, :2]),
            where=local[:, 2:] > 0,
        )
        ij = np.clip(np.floor((uv + 1) * (size / 2)), 0, size - 1).astype(np.intp)
        if address is not None:
            ij = address[mask][:, 1:]
            # A host sampler may round a boundary differently. Accept only a
            # nearest source texel (plus 1/128 texel for hardware precision).
            offset = np.max(np.abs((uv + 1) * (size / 2) - (ij + 0.5)), axis=1)
            if np.any((offset > 0.5 + 1 / 128) & ~protected[mask]):
                raise ValueError(
                    "Recorded texel is inconsistent with the target bearing"
                )
        col[mask], row[mask] = ij.T
        bearing = np.column_stack([(ij + 0.5) * (2 / size) - 1, np.ones(len(ij))])
        norm = np.linalg.norm(bearing, axis=1)
        scale[mask] = norm
        actual[mask] = (bearing / norm[:, None]) @ basis.T @ lens_to_cube.T
    select = (face, row, col)
    ok = known[select] & valid
    d = depth[select][ok].astype(np.float64)
    nweight, fweight = (d, 1 - d) if reversed_z else (1 - d, d)
    inv_z = nweight / near + fweight / far
    if np.any(inv_z <= 0):
        raise ValueError(
            "An infinite-far clear value cannot represent an observed surface"
        )
    distance = np.full(face.shape, np.inf, dtype=np.float64)
    distance[ok] = scale[ok] / inv_z
    colors = rgb[select].copy()
    colors[~valid] = 0
    packet = dict(
        camera_to_world=target,
        rays_camera=actual,
        protected=protected | ~valid,
        host_range=distance,
        sample_rgb=colors,
    )
    if "sample_weights" in lens:
        packet["sample_weights"] = np.asarray(lens["sample_weights"])
    packet["rgb"] = integrate(colors, weights(packet, face.shape))
    for field in ("receiver_mask", "occluder_mask"):
        if field not in cube:
            continue
        mask = np.asarray(cube[field])
        if mask.shape != known.shape or mask.dtype != bool or (mask & ~known).any():
            raise ValueError(f"{field} must label observed cube texels")
        if field == "receiver_mask":
            packet[field] = mask[select] & ok
        else:
            if "receiver_mask" in cube and np.any(mask & cube["receiver_mask"]):
                raise ValueError("An observed texel cannot be both road and foreground")
            packet["occluder_range"] = np.where(mask[select], distance, np.inf)
    if "sensor_name" in lens:
        packet["sensor_name"] = lens["sensor_name"]
    return packet
