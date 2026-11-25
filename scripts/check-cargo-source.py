"""Compare the original glTF hierarchy and rebuilt metric vertices using Blender's own importer."""
import argparse
import json
from pathlib import Path

import bpy
import numpy as np
from scipy.spatial import cKDTree


def vertices(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    parts = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        points = np.empty(len(obj.data.vertices) * 3, dtype="f4")
        obj.data.vertices.foreach_get("co", points)
        transform = np.array(obj.matrix_world)
        parts.append(points.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3])
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("recipe", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    recipe = json.loads(args.recipe.read_text())
    # glTF (x,y,z) maps to Blender's (x,-z,y).
    frame = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    result = {}
    for key, spec in recipe["assets"].items():
        source = spec["source"]
        original = vertices(args.recipe.parent / "original" / source / (source + "_1k.gltf"))
        folder = args.models / key
        meta = json.loads((folder / "metadata.json").read_text())
        rotation = np.array(meta["support_pose"]["rotation_from_source"])
        expected = (original @ frame @ rotation.T - meta["source_origin_y_up_m"]) @ frame.T
        rebuilt = vertices(folder / "asset.glb")
        errors = [float(cKDTree(b).query(a)[0].max()) for a, b in [(expected, rebuilt), (rebuilt, expected)]]
        if max(errors) > .00002:
            raise ValueError(f"{key}: original node hierarchy differs by {max(errors):.6f} metres")
        result[key] = dict(max_original_to_rebuilt_m=errors[0], max_rebuilt_to_original_m=errors[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
