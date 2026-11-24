"""Check closest reflection, noncentral ray origins and large map coordinates on a real photographed pit."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from host_depth import geometry
from metric_reflection import MetricReflectors
from pavement_appearance import camera_anchor
from pavement_layout import PavementLayout
from pavement_water import interface
from road_geometry import save_json


def check(models, exposure, replay):
    layout = PavementLayout(ROOT / "gen2hole/examples/roadside-reflections/pair.json", models)
    with np.load(exposure / "urban-front.npz") as raw:
        rays, _, _, pose, origins = geometry(raw, with_origins=True)
        anchor = camera_anchor(
            json.loads((exposure / "appearance.json").read_text()), str(raw["sensor_name"].item())
        )
    layout.bind(anchor)
    seat = next(i["anchor"] for i in anchor["instances"] if i["id"] == "oval")
    water = layout.water_states["oval"]
    bed = layout.pond_beds["oval"]
    wet = np.array(Image.open(replay / "part.png")) == 4
    cam = rays[0][wet]
    start = (np.zeros_like(cam) if origins is None else origins[0][wet]) @ pose[:3, :3].T + pose[:3, 3]
    world = cam @ pose[:3, :3].T
    basis = np.asarray(seat["basis_columns"])
    up = np.asarray(water["up_local"])
    local = world @ basis
    initial = (start - seat["center_world"]) @ basis
    ranges = (water["free_surface_m"] - initial @ up) / (local @ up)
    point = initial + local * ranges[:, None]
    mirror, _, fraction = interface(local, up)
    bank, _ = bed.trace(point, mirror, seat["pavement_appearance"])
    candidates = np.flatnonzero(np.isfinite(bank) & (bank > 0.005) & (fraction > 0.03))
    if not len(candidates):
        raise ValueError("Exposure must include a measurable bank reflection")
    i = candidates[np.argmax(fraction[candidates])]
    direction = mirror[i] @ basis.T
    hit = point[i] @ basis.T + seat["center_world"]
    tangent = np.cross(direction, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    across = np.cross(direction, tangent)
    span = 0.025 * np.array([-tangent - across, -tangent + across, tangent, tangent - across])
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype="i4")
    original = {}
    for calibrated in (True, False):
        local_seat = dict(seat)
        if not calibrated:
            local_seat.pop("pavement_appearance", None)
        for factor in (0.5, 1.5):
            colors = []
            for offset in (np.zeros(3), np.array([281500.0, 4905723.0, 160.0])):
                shifted = dict(local_seat, center_world=np.asarray(seat["center_world"]) + offset)
                sensor = pose.copy()
                # Move the nominal lens while preserving the registered physical optical origin.
                origin_cam = np.array([[0.12, -0.04, 0.03]])
                sensor[:3, 3] = start[i] + offset - origin_cam[0] @ sensor[:3, :3].T
                vertices = hit + offset + bank[i] * factor * direction + span
                result = []
                for rgb in ([230, 18, 12], [9, 41, 245]):
                    surface = MetricReflectors(vertices, tri, np.tile(np.uint8(rgb), (len(vertices), 1)))
                    result.append(
                        bed.shade(
                            np.uint8([[90, 80, 80]]),
                            sensor,
                            cam[i : i + 1],
                            origin_cam,
                            ranges[i : i + 1],
                            shifted,
                            None,
                            surface,
                        )[0]
                    )
                colors.append(np.array(result))
            if not np.array_equal(*colors):
                raise ValueError("Rebasing camera and mesh changed the reflection")
            diff = int(np.abs(colors[0][0].astype(int) - colors[0][1]).max())
            if (factor < 1 and diff < 5) or (factor > 1 and diff):
                raise ValueError("Foreground reflector and measured bank are incorrectly ordered")
            original[f"{'registered' if calibrated else 'unregistered'}-{factor:g}"] = diff
    return dict(bank_distance_m=float(bank[i]), fresnel=float(fraction[i]), rgb_channel_change=original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "exposure", "replay"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = check(args.models, args.exposure, args.replay)
    save_json(args.replay / "bank-check.json", result)
    print(result)
