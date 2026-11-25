"""Exercise vehicle clearance and a bounded search on a tilted road in a large world frame."""

import json
from importlib import import_module
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from mesh_transfer import RoadMesh

cli = import_module("check-cone-cycles").cli


def verify(models, dest, pairs):
    dest.mkdir(exist_ok=True, parents=True)
    axis = np.arange(-3, 3.001, 0.035)
    x, y = np.meshgrid(axis, axis)
    local = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    basis = Rotation.from_euler("zyx", [29, 8, -3], degrees=True).as_matrix()
    origin = np.array([3550210.0, 4825090.0, 65.0])
    world = local @ basis.T + origin
    trajectory = np.linspace([-3, 0, 0], [3, 0, 0], 41) @ basis.T + origin
    np.save(dest / "road.npy", world)
    np.save(dest / "trajectory.npy", trajectory)
    report = {}
    for pair in pairs:
        model = models / pair
        # The closest side has only 75 mm of clearance at the seed. Require 120 mm.
        seed = origin + basis @ np.array(
            [0, 0.575 - RoadMesh(model).bounds[0, 1], 0.04]
        )
        anchor = dest / (pair + ".json")
        args = (
            dest / "road.npy",
            "--asset",
            model,
            "--trajectory",
            dest / "trajectory.npy",
            "--seed",
            *seed,
            "--up",
            *basis[:, 2],
            "--heading",
            *basis[:, 0],
            "--radius",
            1.35,
            "--vehicle-width",
            1.0,
            "--vehicle-length",
            2.4,
            "--clearance",
            0.12,
        )
        rejected = dest / (pair + "-rejected.json")
        command = [
            sys.executable,
            str(
                import_module("check-cone-cycles").ROOT
                / "scripts/transfer-mesh-view.py"
            ),
            "fit",
            str(args[0]),
            str(rejected),
            *map(str, args[1:]),
            "--placement-search-m",
            ".025",
        ]
        failed = subprocess.run(command, capture_output=True, text=True)
        if (
            failed.returncode == 0
            or "No supported and vehicle-clear placement" not in failed.stderr
            or rejected.exists()
        ):
            raise ValueError(
                "Expected an explicit clearance rejection with no candidate anchor"
            )
        cli("fit", args[0], anchor, *args[1:], "--placement-search-m", 0.25)
        a = json.loads(anchor.read_text())
        pos = (np.array(a["center_world"]) - seed) @ basis
        measured = a["vehicle_clearance"]["minimum_clearance_m"]
        if not np.allclose(pos[:2], [0, 0.1], atol=1e-5) or not 0.17 < measured < 0.18:
            raise ValueError("Incorrect displacement or vehicle envelope")
        report[pair] = dict(
            shift_m=pos[:2].tolist(),
            minimum_clearance_m=measured,
            rejected_candidates=len(a["placement_search"]["rejected"]),
        )
    return report
