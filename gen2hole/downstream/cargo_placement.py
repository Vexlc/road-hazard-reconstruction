"""Find a supported, vehicle-clear seat within an explicitly allowed horizontal displacement."""

import json
import heapq
from pathlib import Path

import numpy as np

from cargo_support import check_members
from mesh_transfer import fit_footprint, fit_road, footprint_size, unit
from prop_registration import driving_clearance


def offset_grid(max_shift, pitch):
    """Visit nearest grid sites first, preferring longitudinal shifts at equal radius."""
    pending = [(0, 0, 0, 0, 0)]
    seen = {(0, 0)}
    while pending:
        _, _, _, x, y = heapq.heappop(pending)
        yield x, y
        for u, v in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (u, v) in seen or np.hypot(u, v) * pitch > max_shift + 1e-9:
                continue
            seen.add((u, v))
            heapq.heappush(pending, (u * u + v * v, abs(v), abs(u), u, v))


def find_seat(
    folder,
    points,
    seed,
    heading,
    up_world,
    radius,
    max_shift,
    bounds,
    trajectory,
    *,
    initial=None,
    half_width_m=1.03,
    half_length_m=2.55,
    margin_m=0.12,
):
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError("Search radius must be positive finite metres")
    if not json.loads((Path(folder) / "metadata.json").read_text()).get(
        "member_ground_check"
    ):
        raise ValueError("Placement search requires a surveyed freestanding assembly")
    if trajectory is None:
        raise ValueError("Provide the registered centre trajectory for clearance")
    seed = np.asarray(seed, float)
    up = unit(up_world)
    axis = unit(np.asarray(heading) - np.dot(heading, up) * up)
    basis = np.column_stack([axis, np.cross(up, axis)])
    size = footprint_size(folder)
    pitch = min(0.1, max_shift)
    failures = []
    for x, y in offset_grid(max_shift, pitch):
        offset = pitch * np.array([x, y])
        target = seed + basis @ offset
        try:
            base = (
                initial
                if initial is not None and (x, y) == (0, 0)
                else fit_road(points, target, axis, up_world=up, radius_m=radius)
            )
            seat = fit_footprint(base, points, size)
            seat["vehicle_clearance"] = driving_clearance(
                seat,
                bounds,
                trajectory,
                half_width_m=half_width_m,
                half_length_m=half_length_m,
                margin_m=margin_m,
            )
            seat["member_support"] = check_members(folder, seat, points)
            # fit_footprint can tilt and translate the result; check the actual horizontal movement.
            actual = (np.array(seat["center_world"]) - seed) @ basis
            if np.linalg.norm(actual) > max_shift + 1e-6:
                raise ValueError("Fitted anchor exceeds the permitted translation")
        except ValueError as error:
            failures.append(dict(offset_m=offset.tolist(), reason=str(error)))
            continue
        seat["placement_search"] = dict(
            requested_seed_world=seed.tolist(),
            max_shift_m=max_shift,
            grid_pitch_m=pitch,
            seed_offset_m=offset.tolist(),
            fitted_offset_m=actual.tolist(),
            rejected=failures,
        )
        return seat
    raise ValueError(
        f"No supported and vehicle-clear placement within {max_shift:g} m "
        f"({len(failures)} candidates; original: {failures[0]['reason']})"
    )
