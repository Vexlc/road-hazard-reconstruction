"""Read the displacement actually wired to Material Output in a Poly Haven Blender material."""

import json
from pathlib import Path
import sys
import bpy


def calibration(path):
    with bpy.data.libraries.load(str(path), link=False) as (src, loaded):
        loaded.materials = src.materials
    (mat,) = loaded.materials
    (output,) = [n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    (link,) = output.inputs['Displacement'].links
    node = link.from_node
    (image_link,) = node.inputs['Height'].links
    tex = image_link.from_node
    if (
        node.type != 'DISPLACEMENT'
        or tex.type != 'TEX_IMAGE'
        or tex.image.colorspace_settings.name != 'Non-Color'
    ):
        raise ValueError('Expected a directly connected, linear displacement image')
    if any(node.inputs[k].is_linked for k in ('Scale', 'Midlevel')):
        raise ValueError('Expected scalar displacement parameters')
    return dict(
        image=Path(tex.image.filepath).name,
        scale_m=node.inputs['Scale'].default_value,
        midlevel=node.inputs['Midlevel'].default_value,
    )


if __name__ == '__main__':
    print(json.dumps(calibration(Path(sys.argv[1]))))
