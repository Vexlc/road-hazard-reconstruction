"""Read metric depth from a host camera and unproject labelled road samples in its world frame."""

import json
import numpy as np

from mesh_transfer import pixel_rays


def sampled(array, shape, name):
    value = np.asarray(array)
    if value.shape == shape[1:]:
        return np.broadcast_to(value, shape)
    if value.shape != shape:
        raise ValueError(f"{name} must have H×W or S×H×W samples aligned to the rays")
    return value


def sample_origins(view, rays, protected):
    """Ray starts in metres in the camera reference frame, shared or per subpixel."""
    if "ray_origins_camera" not in view:
        return None
    origins = np.asarray(view["ray_origins_camera"])
    if origins.shape == rays.shape[1:]:
        origins = np.broadcast_to(origins, rays.shape)
    if origins.shape != rays.shape or not np.issubdtype(origins.dtype, np.number) or np.iscomplexobj(origins):
        raise ValueError("ray_origins_camera must be a metric H×W×3 or S×H×W×3 array")
    valid = np.isfinite(origins).all(-1)
    if np.any(~valid & ~protected):
        raise ValueError("Unmasked ray origins must be finite")
    return origins if valid.all() else np.where(valid[..., None], origins, 0.)


def optical_samples(view, calibration=None):
    if ("rays_camera" in view) == (calibration is not None):
        raise ValueError("Provide registered rays or one lens calibration")
    if calibration is None:
        rays = np.asarray(view["rays_camera"])
        valid = True
    else:
        rays, valid = pixel_rays(json.loads(calibration.read_text()), return_valid=True)
    if rays.ndim == 3:
        rays = rays[None]
    if rays.ndim != 4 or rays.shape[-1] != 3 or not rays.shape[0]:
        raise ValueError("Expected S×H×W×3 ray directions")
    shape = rays.shape[:-1]
    protected = sampled(view["protected"], shape, "protected")
    if np.issubdtype(protected.dtype, np.integer):
        protected = protected.astype(bool)
    if protected.dtype != bool:
        raise ValueError("protected must be boolean")
    protected = protected | ~np.asarray(valid)
    if "sample_weights" in view:
        from host_exposure import weights
        protected |= weights(view, shape) == 0
    unit = np.isclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-4)
    if np.any(~unit & ~protected):
        raise ValueError("Unmasked rays must be unit vectors")
    return rays, protected, sample_origins(view, rays, protected)


def reference_samples(view, rays, origins, *, with_origins):
    pose = np.asarray(view["camera_to_world"])
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Expected a finite homogeneous camera_to_world")
    rot = pose[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or np.linalg.det(rot) < 0.99999:
        raise ValueError("Expected metric right-handed rotation")
    if "row_time_ns" in view and "camera_to_world_rows" not in view:
        raise ValueError("Register row_time_ns with the sensor trajectory before rendering a rolling exposure")
    if "camera_to_world_rows" in view:
        from camera_readout import rolling_rays

        if not with_origins:
            raise ValueError("Rolling readout requires origins as well as directions")
        rays, origins = rolling_rays(view["camera_to_world_rows"], pose, rays, origins)
    return pose, rays, origins


def geometry(view, calibration=None, *, with_origins=False):
    rays, protected, origins = optical_samples(view, calibration)
    shape = rays.shape[:-1]
    if ("host_z" in view) == ("host_range" in view):
        raise ValueError("Supply exactly one of host_range or host_z")
    if "host_z" in view:
        z = sampled(view["host_z"], shape, "host_z")
        if origins is not None:
            z = z - origins[..., 2]
        rz = rays[..., 2]
        known = np.isfinite(z) & ~protected
        # A signed axial plane degenerates at a 90-degree bearing; that pixel needs radial data.
        if np.any(known & ((z * rz <= 0) | (abs(rz) < 1e-6))):
            raise ValueError("Axial displacement from the ray origin needs the sign of ray.z; use radial range at the optical horizon")
        distance = np.divide(
            z, rz, out=np.full(shape, np.inf, dtype=np.result_type(z, np.float32)), where=known
        )
    else:
        distance = sampled(view["host_range"], shape, "host_range")
        if np.any((distance <= 0) & ~protected):
            raise ValueError("Expected positive radial range or unknown")
    # Decode signed Z before expressing the row's rays in the reference camera.
    pose, rays, origins = reference_samples(view, rays, origins, with_origins=with_origins)
    result = rays, distance, protected, pose
    return (*result, origins) if with_origins else result


def road_samples(view, calibration=None, stride=1):
    if stride < 1 or "receiver_mask" not in view:
        raise ValueError("Provide a positive stride and a registered road receiver_mask")
    rays, dist, protected, pose, origins = geometry(view, calibration, with_origins=True)
    selected = sampled(view["receiver_mask"], dist.shape, "receiver_mask")
    if selected.dtype != bool:
        raise ValueError("receiver_mask must be boolean")
    valid = selected & ~protected & np.isfinite(dist) & (dist > 0)
    if "occluder_range" in view:
        near = sampled(view["occluder_range"], dist.shape, "occluder_range")
        valid &= ~(near < dist - 0.00001)
    subs = (slice(None), slice(None, None, stride), slice(None, None, stride))
    valid = valid[subs]
    points = rays[subs][valid] * dist[subs][valid, None]
    if origins is not None:
        points = points + origins[subs][valid]
    return points @ pose[:3, :3].T + pose[:3, 3]
