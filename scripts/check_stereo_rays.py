"""Exercise degeneracies and per-ray origins with a metric plane of known depth."""

import numpy as np

from host_stereo import triangulate


def check_ray_contract():
    y, x = np.indices((5, 14))
    f, z = 30.0, 3.0
    # Non-central rays originate at different heights in each scanline on both cameras.
    directions = np.stack([(x - 6.5) / f, (y - 2) / f, np.ones_like(x)], -1)
    lengths = np.linalg.norm(directions, axis=-1)
    directions /= lengths[..., None]
    origins = np.zeros_like(directions)
    origins[..., 1] = y * 0.009
    pose = np.eye(4)
    pose[:3, 3] = [4300000.0, 6100000.0, 160.0]
    rows = np.repeat(pose[None], len(y), axis=0)
    rows[:, 0, 3] += np.arange(len(y)) * 0.003
    left = dict(
        rays_camera=np.stack([directions] * 2),
        protected=np.zeros(x.shape, bool),
        camera_to_world=pose,
        camera_to_world_rows=rows,
        ray_origins_camera=origins,
        receiver_mask=np.ones(x.shape, bool),
        rgb=np.zeros((*x.shape, 3), "u1"),
    )
    right_pose = pose.copy()
    right_pose[0, 3] += 0.15
    right_rows = rows.copy()
    right_rows[:, 0, 3] += 0.15
    right = dict(
        left,
        rays_camera=directions,
        camera_to_world=right_pose,
        camera_to_world_rows=right_rows,
    )
    delta = f * 0.15 / z
    xy = np.stack([x - delta, y], -1)
    reverse = np.stack([x + delta, y], -1)
    matches = dict(
        right_xy=np.stack([xy] * 2),
        left_xy=reverse,
        match_valid=np.ones((2, *x.shape), bool),
    )
    packet = triangulate(left, right, matches)
    valid = ~packet["protected"]
    assert np.all(valid[..., 2:]) and not np.any(valid[..., :2])
    # Bilinear interpolation of unit vectors approximates the sub-pixel pinhole direction.
    np.testing.assert_allclose(
        packet["host_range"][valid],
        np.broadcast_to(lengths * z, valid.shape)[valid],
        atol=0.006,
        rtol=0,
    )
    # A second camera with a vertically varying pupil offset has row-dependent disparity.
    pupil = origins.copy()
    pupil[..., 0] = (0.5 + y) * z / f - 0.15
    variable = dict(
        matches,
        right_xy=np.stack([np.stack([x - y - 0.5, y], -1)] * 2),
        left_xy=np.stack([x + y + 0.5, y], -1),
    )
    noncentral = triangulate(left, dict(right, ray_origins_camera=pupil), variable)
    good = ~noncentral["protected"]
    assert good.sum() > 80
    np.testing.assert_allclose(
        noncentral["host_range"][good],
        np.broadcast_to(lengths * z, good.shape)[good],
        atol=0.013,
        rtol=0,
    )
    broken = matches["right_xy"].copy()
    broken[:, 1, 4] = np.nan
    broken[:, 1, 5] = [
        6.5,
        1,
    ]  # right of the left sample, intersecting behind the cameras
    broken[:, 1, 6] = [6, 1]  # parallel rays
    broken[:, 1, 7, 1] += 1  # vertical epipolar error
    right_mask = right["protected"].copy()
    right_mask[1, 7] = True
    broken[:, 1, 10] = [
        8,
        1,
    ]  # integer match must ignore masked or nonfinite zero-weight neighbours
    back = reverse.copy()
    back[1, 8] = [10, 1]
    back[1, 9] = np.nan
    probed = triangulate(
        left,
        dict(right, protected=right_mask),
        dict(matches, right_xy=broken, left_xy=back),
    )
    for col in (4, 5, 6, 7, 8, 9):
        assert (
            probed["protected"][:, 1, col].all()
            and not probed["receiver_mask"][:, 1, col].any()
        )
    assert not probed["protected"][:, 1, 10].any()
    zero = triangulate(
        left, dict(right, camera_to_world=pose, camera_to_world_rows=rows), matches
    )
    assert zero["protected"].all() and np.isinf(zero["host_range"]).all()
    for invalid in (
        dict(matches, match_valid=matches["match_valid"].astype("u1")),
        dict(matches, right_xy=xy),
        dict(matches, left_xy=reverse[:1]),
    ):
        try:
            triangulate(left, right, invalid)
        except ValueError:
            continue
        raise AssertionError("Accepted an invalid correspondence array")
