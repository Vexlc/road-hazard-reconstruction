"""Metric scanline poses for central and noncentral rolling-readout cameras."""

import numpy as np
from scipy.spatial.transform import Rotation


def check_rigid(poses):
    if (
        poses.shape[-2:] != (4, 4)
        or not np.issubdtype(poses.dtype, np.floating)
        or not np.isfinite(poses).all()
    ):
        raise ValueError("Expected finite floating-point 4×4 poses")
    rotation = poses[..., :3, :3]
    if (
        not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or np.any(np.linalg.det(rotation) < 0.99999)
    ):
        raise ValueError("Expected right-handed rigid camera poses in metres")


def interpolate_rows(time_ns, poses, row_ns):
    """Linear camera centres and shortest-arc rotations between bracketing sensor poses."""
    time_ns, poses, row_ns = map(np.asarray, (time_ns, poses, row_ns))
    if time_ns.dtype != np.int64 or row_ns.dtype != np.int64 or time_ns.ndim != 1 or len(time_ns) < 2:
        raise ValueError("Expected int64 sample and exposure timestamps in nanoseconds")
    if poses.shape != (len(time_ns), 4, 4) or row_ns.ndim not in (1, 2) or not row_ns.size:
        raise ValueError("Expected N camera poses and H or S×H row timestamps")
    check_rigid(poses)
    if np.any(time_ns[1:] <= time_ns[:-1]) or row_ns.min() < time_ns[0] or row_ns.max() > time_ns[-1]:
        raise ValueError("Every row exposure must be bracketed by a strictly increasing camera trajectory")
    if int(time_ns[-1]) - int(time_ns[0]) > np.iinfo(np.int64).max:
        raise ValueError("Trajectory time span exceeds int64 nanoseconds")
    query = row_ns.ravel()
    lo = (np.searchsorted(time_ns, query, side="right") - 1).clip(0, len(time_ns) - 2)
    # Subtract in integer nanoseconds before converting the small interval to float.
    u = (query - time_ns[lo]) / (time_ns[lo + 1] - time_ns[lo])
    rot = Rotation.from_matrix(poses[:, :3, :3])
    delta = (rot[:-1].inv() * rot[1:]).as_rotvec()
    result = np.broadcast_to(np.eye(4), (len(query), 4, 4)).copy()
    result[:, :3, :3] = (rot[lo] * Rotation.from_rotvec(delta[lo] * u[:, None])).as_matrix()
    result[:, :3, 3] = poses[lo, :3, 3] + u[:, None] * (poses[lo + 1, :3, 3] - poses[lo, :3, 3])
    result[u == 0] = poses[lo[u == 0]]
    result[u == 1] = poses[lo[u == 1] + 1]
    return result.reshape(*row_ns.shape, 4, 4)


def rolling_rays(row_poses, reference, rays, origins):
    """Express each scanline's camera rays in a common reference camera for ray casting."""
    samples, height, width = rays.shape[:3]
    rows = np.asarray(row_poses)
    if rows.shape == (height, 4, 4):
        rows = rows[None]
    if rows.shape not in ((1, height, 4, 4), (samples, height, 4, 4)):
        raise ValueError("camera_to_world_rows requires one 4×4 world pose per row, shared or per subray")
    check_rigid(rows)
    rotation = rows[..., :3, :3]
    relative = reference[:3, :3].T @ rotation
    offset = (rows[..., :3, 3] - reference[:3, 3]) @ reference[:3, :3]
    direction = np.einsum("shij,shwj->shwi", relative, rays)
    starts = np.broadcast_to(offset[:, :, None], (samples, height, width, 3))
    if origins is not None:
        starts = starts + np.einsum("shij,shwj->shwi", relative, origins)
    return direction, starts
