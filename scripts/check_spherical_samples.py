"""Independent registration checks at spherical crop boundaries and on confirmed sky rays."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mei_camera import calibrated_samples
from host_position import register_positions
from host_stereo import triangulate


def verify():
    with TemporaryDirectory() as tmp:
        lens = Path(tmp) / "sphere.json"
        lens.write_text(json.dumps(["EQUIRECTANGULAR", 192, 96]))
        offsets = np.array([[-0.33, 0.25], [0.41, -0.21]])
        raw = dict(sample_offsets_px=offsets)
        rays, _ = calibrated_samples(lens, raw)
        target = dict(raw, sensor_window_xywh=[187, 35, 13, 8])
        wrapped, _ = calibrated_samples(lens, target)
        np.testing.assert_array_equal(
            wrapped, rays[:, 35:43][:, :, np.arange(187, 200) % 192]
        )
        # Optical axes are right, down, forward; signed latitude decreases down the raster.
        for i, (dx, dy) in enumerate(offsets):
            u = (
                (np.arctan2(wrapped[i, ..., 0], wrapped[i, ..., 2]) / (2 * np.pi) + 0.5)
                * 192
                - 0.5
                - dx
            )
            v = (np.arcsin(wrapped[i, ..., 1]) / np.pi + 0.5) * 96 - 0.5 - dy
            np.testing.assert_allclose(
                (u - np.arange(187, 200) + 96) % 192 - 96, 0, atol=3e-5
            )
            np.testing.assert_allclose(v - np.arange(35, 43)[:, None], 0, atol=2e-5)
        y, x = np.indices((8, 13))
        pose = np.eye(4)
        known = x < 4
        clear = (x >= 7) & (x <= 10)
        protected = x == 9
        view = dict(
            target,
            camera_to_world=pose,
            rgb=np.zeros((8, 13, 3), "u1"),
            protected=protected,
            empty_space_mask=clear,
            position_valid=known,
            surface_position=wrapped.astype(float) * 2.3,
        )
        result = register_positions(view, calibration=lens)
        expected = np.broadcast_to(
            ~(known | clear) | protected, result["protected"].shape
        )
        np.testing.assert_array_equal(result["protected"], expected)
        assert np.isinf(result["host_range"][:, clear]).all()
        source = {
            k: v
            for k, v in view.items()
            if k not in ("position_valid", "surface_position")
        }
        matches = dict(
            match_valid=np.zeros_like(wrapped[..., 0], bool),
            right_xy=np.zeros((*wrapped.shape[:-1], 2)),
        )
        other = dict(source, camera_to_world=pose.copy())
        other["camera_to_world"][0, 3] = 0.15
        packet = triangulate(source, other, matches, lens, lens)
        assert np.array_equal(packet["protected"][0], ~clear | protected)
        measured = np.full(x.shape, np.nan)
        measured[:, :3] = 4.1
        measured[:, 5] = 0.84
        supplemented = triangulate(
            dict(source, supplemental_range_m=measured), other, matches, lens, lens
        )
        for c, d in [(1, 4.1), (5, 0.84)]:
            np.testing.assert_allclose(
                supplemented["host_range"][:, :, c], d, rtol=0, atol=1e-6
            )
            assert supplemented["supplemental_range_used"][:, :, c].all()
        assert (
            supplemented["protected"][:, :, 4].all()
            and supplemented["protected"][:, :, 9].all()
        )
        # Supplemental distances are measured from each moving pupil, even in a large world frame.
        shifted = np.eye(4)
        shifted[:3, 3] = [4100000, 5100000, 70]
        rows = np.repeat(shifted[None], len(y), axis=0)
        rows[:, 0, 3] += np.arange(len(y)) * 0.003
        centers = np.zeros_like(wrapped)
        centers[..., 2] = 0.06
        moving = dict(
            source,
            camera_to_world=shifted,
            camera_to_world_rows=rows,
            ray_origins_camera=centers,
            supplemental_range_m=measured,
        )
        moved = triangulate(moving, other, matches, lens, lens)
        np.testing.assert_allclose(
            moved["host_range"][:, :, 5], 0.84, rtol=0, atol=2e-7
        )
        np.testing.assert_allclose(
            moved["ray_origins_camera"][:, :, 5, 2], 0.06, atol=1e-7
        )
        for value in (0.0, -1.0, np.inf):
            illegal = measured.copy()
            illegal[0, 0] = value
            try:
                triangulate(
                    dict(source, supplemental_range_m=illegal),
                    other,
                    matches,
                    lens,
                    lens,
                )
            except ValueError as error:
                if "Supplemental" not in str(error):
                    raise
            else:
                raise AssertionError("Invalid supplemental measurement accepted")
        bad = dict(
            matches, match_valid=np.broadcast_to(clear, matches["match_valid"].shape)
        )
        try:
            triangulate(source, other, bad, lens, lens)
        except ValueError as error:
            if "empty" not in str(error):
                raise
        else:
            raise AssertionError("Conflicting stereo hit and confirmed empty space")
        for window in ([192, 0, 2, 2], [191, 0, 193, 1], [0, 94, 5, 3], [-1, 0, 1, 1]):
            try:
                calibrated_samples(lens, dict(raw, sensor_window_xywh=window))
            except ValueError:
                continue
            raise AssertionError("Invalid panorama window")
