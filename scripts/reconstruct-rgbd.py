"""Build a measured pothole asset from a registered RGB, depth image and binary mask."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "outputs/road_hazard_expansion/reproduction"
sys.path.insert(0, str(PIPELINE))
from batch_rgbd import project_rgbd, reconstruct_checked


def write(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def build(args):
    with Image.open(args.rgb) as img:
        color = np.array(img.convert("RGB"))
    with Image.open(args.mask) as img:
        mask = np.array(img.convert("L")) > 0
    if args.depth.suffix.lower() == ".npy":
        raw = np.load(args.depth,allow_pickle=False)
    else:
        with Image.open(args.depth) as img:
            raw = np.array(img)
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise ValueError("Depth must be a single numeric channel of optical-axis distances")
    h,w = raw.shape
    intrinsics = (args.fx,args.fy,(w-1)*.5 if args.cx is None else args.cx,(h-1)*.5 if args.cy is None else args.cy)
    frames = project_rgbd(color,raw,mask,intrinsics,args.depth_scale)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",args.id):
        raise ValueError("Use one asset ID with letters, numbers, periods, underscores or hyphens")
    out = args.out.absolute()
    if out.exists() or out.is_symlink():
        raise ValueError(f"Choose a new output directory: {out}")
    out.parent.mkdir(parents=True,exist_ok=True)
    inputs = {k:dict(filename=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for k,p in (("rgb",args.rgb),("depth",args.depth),("mask",args.mask))}
    with TemporaryDirectory(dir=out.parent,prefix=".rgbd-") as tmp:
        stage = Path(tmp)/"library"
        relative = Path("rgbd")/args.id
        folder = stage/relative
        samples = Path(tmp)/"samples"
        detail, quality = reconstruct_checked(frames,folder,samples/(args.id+".npz"),inputs,args.id,args.depth_scale)
        detail.update(construction="Registered RGB and depth, pinhole projection, local road plane and measured height field",
                      limits=["Only the visible surface is observed; gaps are locally interpolated.",
                              "Geometry depends on the supplied calibration and depth scale."],
                      intrinsics_px=dict(zip(("fx","fy","cx","cy"),intrinsics)))
        write(folder/"capture_metadata.json",detail)
        width,length = detail["pothole_extent_xy_m_p0_4_to_p99_6"]
        meta = dict(id=args.id,category="rgbd",family=args.id,material="captured_rgb",units="metres",
                    width_m=width,length_m=length,depth_p95_m=detail["depth_mm_p50_p95_p99_in_mask"][1]/1000,
                    geometry_source="registered_rgbd_with_supplied_intrinsics",mesh_name="pothole.glb",
                    depth_input_metres_per_unit=args.depth_scale,intrinsics_px=detail["intrinsics_px"],
                    quality_checks=quality,measured_support_in_mask=detail["stereo_near_sample_coverage_in_mask"])
        write(folder/"metadata.json",meta)
        files = dict(glb="pothole.glb",surface="surface.npz",metadata="metadata.json")
        if not args.mesh_only:
            paths = [str(ROOT/"work/asset_pipeline"),str(ROOT/"work/road_hazard_pipeline")]
            env = dict(os.environ,PYTHONPATH=os.pathsep.join(paths),OMP_NUM_THREADS="4")
            subprocess.run([args.gaussian_python,PIPELINE/"export_gaussians.py","--out",stage,"--asset",args.id,
                            "--meshes",samples,"--samples",samples],env=env,check=True)
            meta = json.loads((folder/"metadata.json").read_text())
            files.update(ply="asset.ply",preview="preview.jpg")
        write(stage/"asset_manifest.json",[dict(meta,**{k:str(relative/v) for k,v in files.items()})])
        stage.rename(out)
    print(out/"asset_manifest.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb","depth","mask","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--id",required=True)
    p.add_argument("--fx",required=True,type=float)
    p.add_argument("--fy",required=True,type=float)
    p.add_argument("--cx",type=float,help="Principal point x in pixels; default is image centre")
    p.add_argument("--cy",type=float,help="Principal point y in pixels; default is image centre")
    p.add_argument("--depth-scale",type=float,required=True,help="Metres per stored depth unit")
    p.add_argument("--mesh-only",action="store_true")
    p.add_argument("--gaussian-python",type=Path,default=ROOT/"work/envs/streetcrafter/bin/python")
    try:
        build(p.parse_args())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        p.exit(1,f"rgbd: {error}\n")
