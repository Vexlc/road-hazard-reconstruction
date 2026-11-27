"""Verify state-dependent aperture visibility, source holdouts and the native Gaussian outputs."""

import argparse, json
from pathlib import Path
import numpy as np
from build import Surface, KEY, S, OPEN, H, THICK, R, grid, xyz, write_json
from camera import solve
from casting import project
from inspect_assets import inspect
from road_geometry import glb_triangles


def check(models, out):
    out.mkdir(parents=True, exist_ok=True)
    reg = json.loads((models / "registration.json").read_text())
    obs = np.load(models / "observations.npz")
    cal = reg["calibration"]
    M = np.array(cal["homography"])
    uv_checks = []
    for label, vertices, faces, uv, _ in glb_triangles(
        models / "observed-pair.glb", appearance=True
    ):
        assert label == 9
        world = vertices[:, :2] + cal["near_xz_m"]
        mapped = uv * [1936, 2592] - 0.5
        pixel_error = np.linalg.norm(mapped - project(M, world), axis=1)
        mid_error = np.linalg.norm(mapped[faces].mean(1) - project(M, world[faces].mean(1)), axis=1)
        assert max(pixel_error) < 0.01 and max(mid_error) < 0.1 and np.allclose(vertices[:, 2], H)
        uv_checks.append(
            dict(
                vertices=len(vertices),
                triangles=len(faces),
                max_vertex_px=float(max(pixel_error)),
                max_centroid_px=float(max(mid_error)),
            )
        )
    assert len(uv_checks) == 2
    measured = []
    for i, name in enumerate(("near", "far"), 1):
        # Rays projected back to the top plane, held out of the joint camera solve.
        pix = obs["hold" + str(i)]
        q = np.c_[pix, np.ones(len(pix))] @ np.linalg.inv(M).T
        local = q[:, :2] / q[:, 2:] - cal[name + "_xz_m"]
        radius = R if i == 1 else cal["far_diameter_m"] / 2
        measured.append(float(np.median(abs(np.linalg.norm(local, axis=1) - radius)) * 1000))
    rings = reg["rings"]
    trials = [solve(f, obs, rings) for f in [1600, 2500]]
    state_checks = {}
    for state in S["states"]:

        def evidence(folder, surf, meta, spec, run):
            xy = grid([2 * OPEN, 2 * OPEN], 0.003).reshape(-1, 2)
            xy = xy[np.linalg.norm(xy, axis=1) < OPEN - 0.003]
            eye = xyz(xy, 1)
            t, part, _ = surf.cast(eye, np.broadcast_to([0.0, 0.0, -1.0], eye.shape))
            if not np.isfinite(t).all():
                raise ValueError("Interior ray escaped optical envelope")
            uncovered = np.isin(part, [6, 7])
            ground = Surface(models / "loose-cover.glb")
            center = np.array(meta["cover_translation_xy_m"])
            # Radial sample on the moveable casting to catch face topology or scale regressions.
            rays = np.c_[
                R * 0.85 * np.cos(np.arange(48) * 2 * np.pi / 48),
                R * 0.85 * np.sin(np.arange(48) * 2 * np.pi / 48),
            ]
            top = xyz(rays, 1)
            down = np.broadcast_to([0.0, 0.0, -1.0], top.shape)
            stand, l, _ = ground.cast(top, down)
            assert (l == 9).all() and np.max(abs(stand - 1)) < 2e-6
            if state != "open":
                hit, p, _ = surf.cast(top + np.r_[center, 0], down)
                assert (p == 9).all() and np.max(
                    abs(1 - hit - H - (THICK if state == "offset" else 0))
                ) < 2e-6
            with np.load(folder / "surface.npz") as field:
                nr = (field["first_part"] == 9).sum()
                observed = int(field["source_face"].sum())
            return dict(
                aperture_uncovered_fraction=float(uncovered.mean()),
                lid_source_texel_fraction=observed / int(nr) if nr else None,
                median_inner_first_z_m=float(np.median(1 - t)),
            )

        row = inspect(models / state, out / state, evidence, ids={KEY})
        state_checks[state] = row[KEY]["evidence"]
    f = [state_checks[s]["aperture_uncovered_fraction"] for s in S["states"]]
    assert f[0] == 0 and 0.15 < f[1] < 0.85 and f[2] == 1.0
    write_json(
        out / "checks.json",
        dict(
            states=state_checks,
            observed_pair_uv_checks=uv_checks,
            heldout_ring_radial_median_mm=measured,
            nominal_scale_trials=[
                dict(diameter_m=d, center_spacing_m=cal["centre_separation_m"] * d / (2 * R))
                for d in [0.65, 0.72, 0.8]
            ],
            focal_prior_trials=[
                dict(prior_px=f, center_spacing_m=c["centre_separation_m"], p90_px=c["hold_p90_px"])
                for f, c in zip([1600, 2500], trials)
            ],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("models", "out"):
        p.add_argument(n, type=Path)
    check(**vars(p.parse_args()))
