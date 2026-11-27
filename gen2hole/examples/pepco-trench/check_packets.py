"""Verify shared beam indices and bounded/visual ranges against reraycasting the delivered GLBs."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from build import Surface, S
from casting import visible_face


def check(models, lidar, out):
    info = json.loads((lidar / "report.json").read_text())
    states = S["states"]
    mesh = [Surface(models / s / "model/series/assets/raised/A591/asset.glb") for s in states]
    calibration = json.loads((models / "registration.json").read_text())["calibration"]
    change = np.zeros((3, 3), int)
    sensor_rows = {}
    total = 0
    max_error = 0.0
    for frame in info["frames"]:
        if "excluded" in frame:
            continue
        path = lidar / f"{frame['frame']:02}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame["packet_sha256"]:
            raise ValueError(path)
        with np.load(path) as p:
            assert list(p["states"]) == states and np.all(np.diff(p["source_row"]) > 0)
            n = len(p["source_row"])
            total += n
            for sensor, count in zip(*np.unique(p["sensor"], return_counts=True)):
                key = str(sensor)
                sensor_rows[key] = sensor_rows.get(key, 0) + int(count)
            parts = p["inserted_part"]
            assert np.array_equal(np.isnan(p["plate_constrained_range_m"]), p["unresolved"])
            change += (parts[:, None] != parts[None, :]).sum(2)
            for i, m in enumerate(mesh):
                part = parts[i]
                use = part != 0
                dist, hit, _ = m.cast(p["origin_local_m"], p["unit_ray_local"])
                assert np.array_equal(hit[use], part[use])
                shift = np.array(S["offset_xy_m"]) if states[i] == "offset" else np.zeros(2)

                def covered(d, l):
                    xy = (
                        p["origin_local_m"][:, :2]
                        + np.where(np.isfinite(d), d, 0)[:, None] * p["unit_ray_local"][:, :2]
                    )
                    return use & (l == 9) & visible_face(calibration, xy - shift)

                source = covered(dist, hit)
                assert np.array_equal(p["unresolved"][i], use & ~source)
                limits = []
                known = source.copy()
                for inches in S["reported_approximate_raise_inches"]:
                    eye = p["origin_local_m"] - [0, 0, 0.0254 * inches - S["model_raise_m"]]
                    t, l, _ = m.cast(eye, p["unit_ray_local"])
                    known &= covered(t, l)
                    limits.append(t)
                saved = p["rise_bracket_range_m"][i]
                assert np.array_equal(known, np.isfinite(saved).all(1))
                if known.any():
                    np.testing.assert_allclose(
                        saved[known], np.sort(np.stack(limits, 1)[known], axis=1), atol=1e-4, rtol=0
                    )
                if use.any():
                    max_error = max(
                        max_error, float(abs(dist[use] - p["visual_range_m"][i, use]).max())
                    )
                assert np.array_equal(p["visual_range_m"][i, ~use], p["host_range_m"][~use])
                resolved = ~p["unresolved"][i]
                assert np.array_equal(
                    p["plate_constrained_range_m"][i, resolved], p["visual_range_m"][i, resolved]
                )
    assert (
        np.array_equal(change, info["changed_part_rows"])
        and len(sensor_rows) == 2
        and max_error < 0.0001
    )
    out.write_text(
        json.dumps(
            dict(
                candidate_rows=total,
                per_sensor=sensor_rows,
                max_recast_error_m=max_error,
                changed_first_part_pairs=change.tolist(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("models", "lidar", "out"):
        p.add_argument(n, type=Path)
    check(**vars(p.parse_args()))
