"""Export the textured, removable cover at each reconstructed pose in its original coordinate system."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import trimesh

from export_pit_components import MAPS, export_checked


def export(bundle):
    report = {}
    for path in sorted((bundle / "assets").glob("*/*/asset.glb")):
        meta = json.loads((path.parent / "metadata.json").read_text())
        scene = trimesh.load_scene(path, process=False)
        parts = {}
        for node in scene.graph.nodes_geometry:
            if not node.startswith("Lid "):
                continue
            matrix, geometry = scene.graph[node]
            if not np.allclose(matrix, np.eye(4)):
                raise ValueError(f"{node} uses a non-identity node transform")
            mesh = scene.geometry[geometry]
            for channel in MAPS:
                image = getattr(mesh.visual.material, channel)
                if image is not None:
                    image.format = "PNG"
            parts[node] = mesh, mesh.faces
        if not parts:
            continue
        output = path.with_name("lid.glb")
        credit = {k: meta[k] for k in ("credit", "source_url", "license_url") if k in meta}
        error = export_checked(output, parts, np.zeros(3), metadata=credit)
        report[meta["id"]] = dict(
            file=str(output.relative_to(bundle)),
            parts=sorted(parts),
            triangles=sum(len(faces) for _, faces in parts.values()),
            maximum_vertex_error_m=error,
            parent_sha256=sha256(path.read_bytes()).hexdigest(),
            sha256=sha256(output.read_bytes()).hexdigest(),
        )
    if not report:
        raise ValueError(f"No detachable lid surfaces in {bundle}")
    (bundle / "assets/lids.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    export(parser.parse_args().bundle)
