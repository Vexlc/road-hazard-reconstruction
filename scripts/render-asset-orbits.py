"""Render fixed-camera turntables from the actual GLBs for the local atlas."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import bpy
from mathutils import Vector
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gen2hole/recipes" / name) for name in ("common","series")]
from render_mesh import prepare_view

OUT = ROOT / "tools/asset_atlas/orbits"
FRAMES = 24


def render(bundle, name):
    rows = json.loads((bundle / "assets/asset_manifest.json").read_text())
    row = next(r for r in rows if r["id"] == name)
    folder = (bundle / "assets" / row["glb"]).parent
    width, height = 540, 360
    scene, points = prepare_view(folder,bundle/"mesh_cache","inspection",width,height)
    # A fixed target and orthographic scale show perspective changes without zoom or recentering.
    centre = (points.min(0)+points.max(0))/2
    delta = points-centre
    radius = float(np.linalg.norm(delta[:,:2],axis=1).max())
    elevation = math.radians(39)
    vertical = radius*math.sin(elevation) + float(np.abs(delta[:,2]).max())*math.cos(elevation)
    scene.camera.data.ortho_scale = 2.18*max(radius,vertical*width/height)
    scene.cycles.samples = 12
    scene.render.image_settings.quality = 91
    with TemporaryDirectory(prefix="road-orbit-") as temp:
        frames = []
        for i in range(FRAMES):
            angle = 2*math.pi*i/FRAMES - math.pi/2
            away = Vector((math.cos(angle)*math.cos(elevation),
                           math.sin(angle)*math.cos(elevation), math.sin(elevation)))
            scene.camera.location = Vector(centre) + 3.4*away
            scene.camera.rotation_euler = (-away).to_track_quat("-Z","Y").to_euler()
            frame = Path(temp)/"view.jpg"
            scene.render.filepath = str(frame)
            bpy.ops.render.render(write_still=True)
            with Image.open(frame) as img:
                frames.append(img.convert("RGB"))
        # Publish only a complete loop. Keep the immutable first frame as the initial angle.
        target = OUT/(name+".webp")
        with TemporaryDirectory(dir=OUT, prefix=".orbit-") as publication:
            draft = Path(publication)/target.name
            frames[0].save(draft,save_all=True,append_images=frames[1:],duration=140,loop=0,quality=84,method=4)
            draft.replace(target)
        print(name, target, flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,help="series directory containing the asset manifest and mesh cache")
    parser.add_argument("--ids",nargs="+",help="IDs in this bundle")
    args=parser.parse_args()
    if bool(args.bundle)!=bool(args.ids):
        parser.error("Specify both --bundle and --ids")
    if args.bundle:
        records=json.loads((args.bundle/'assets/asset_manifest.json').read_text())
        if not set(args.ids)<={r['id'] for r in records}:
            parser.error("Some requested IDs are absent from the bundle")
        jobs=[(args.bundle,key) for key in dict.fromkeys(args.ids)]
    else:
        jobs=[(ROOT/("runs/rgb-box-P90/series" if key=="P90" else "gen2hole/runs/demo/series"),key)
              for key in ("A13","M13","P13","P90")]
    OUT.mkdir(parents=True,exist_ok=True)
    for bundle,name in jobs:
        render(bundle,name)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
