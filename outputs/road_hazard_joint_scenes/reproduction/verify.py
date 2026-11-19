"""Check every editable scene and confirm every source asset is present in it at true scale."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

TRAINING = {"optimizer","xyz_gradient_accum","denom","max_radii2D","spatial_lr_scale","active_sh_degree"}

def exact(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):
        keys=set(a)-TRAINING
        return keys==set(b)-TRAINING and all(exact(a[k],b[k]) for k in keys)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def count_frames(path,expected):
    written,_=imageio_ffmpeg.count_frames_and_secs(str(path))
    assert written==expected,(str(path),written,expected)

def run(project,work):
    root=project/"outputs/road_hazard_joint_scenes"
    library=project/"outputs/road_hazard_expansion"
    plan=json.loads((work/"plan.json").read_text())
    manifest={r["id"]:r for r in json.loads((library/"asset_manifest.json").read_text())}
    baseline = {}
    for site, label in (("055", "B"), ("056", "A")):
        checkpoint = project / "work/runs" / f"pandaset{site}_warmup" / "trained_model/iteration_7000.pth"
        if not checkpoint.is_file():
            checkpoint = project / "outputs" / f"scene_{label}_{site}" / "checkpoint_7000.pth"
        baseline[site] = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    all_assets=set()
    checks=[]
    for i,batch in enumerate(plan["scenes"],1):
        dir_=root/("scene_"+batch["id"])
        result=json.loads((dir_/"scene.json").read_text())
        assert result["id"]==batch["id"] and result["assets"]==batch["assets"]
        assert [r["asset_id"] for r in result["placements"]]==batch["assets"]
        path=dir_/"trained_model/iteration_7000.pth"
        assert path.stat().st_size==result["checkpoint_bytes"]
        source=baseline[batch["site"]]
        data=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        components=set(source)-{"background","iter"}
        assert set(data)-{"background","iter"}==components
        assert all(exact(source[k],data[k]) for k in components)
        points=data["background"]["xyz"].numpy()
        assert len(points)==result["composited_background_points"]
        start=len(source["background"]["xyz"])-result["removed_background_points"]
        coords_error=[]
        min_center_contrast=1_000_000
        for a in result["placements"]:
            asset=a["asset_id"]
            row=manifest[asset]
            assert asset not in all_assets and a["asset_points"]==row["gaussians"] and a["size_multiplier"]==1.
            local=PlyData.read(library/row["ply"],mmap="r")["vertex"].data
            xyz=np.stack([local[k] for k in ("x","y","z")],axis=1)
            target=xyz@np.asarray(a["basis_columns"]).T + np.asarray(a["center_world"])
            length=len(local)
            error=float(np.max(np.abs(points[start:start+length]-target)))
            assert error<.000026,(asset,error)
            coords_error.append(error)
            start += length+a["apron_points"]
            with Image.open(dir_/a["preview_file"]) as p:
                assert p.size==(880,506)
                p.load()
            assert a["contrast_pixels_near_center"]>25, (batch["id"],asset,a["contrast_pixels_near_center"])
            min_center_contrast=min(min_center_contrast,a["contrast_pixels_near_center"])
            all_assets.add(asset)
        assert start==len(points)
        if batch["showcase"]:
            count_frames(dir_/"drive.mp4",50)
            count_frames(dir_/"inspect.mp4",len(batch["assets"])*40)
            assert (dir_/"scene_frame038.ply").stat().st_size>20_000_000
        checks.append({"scene":batch["id"],"site":batch["site"],"assets":len(batch["assets"]),
                       "max_transform_error_m":max(coords_error),"min_changed_near_pixels":min_center_contrast,
                       "native_checkpoint_reloaded":True,"all_non_background_components_bit_exact":True,
                       "checkpoint_bytes":path.stat().st_size})
        if i%25==0:
            print("Checked",i,"scenes,",len(all_assets),"assets",flush=True)
    assert all_assets==manifest.keys()
    total={"verified_scenes":len(checks),"verified_assets":len(all_assets),
           "showcase_scenes":sum(x["showcase"] for x in plan["scenes"]),"scenes":checks}
    (root/"joint_validation.json").write_text(json.dumps(total,indent=2)+"\n")
    print("PASS",total["verified_scenes"],total["verified_assets"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path);p.add_argument("work",type=Path)
    a=p.parse_args()
    run(a.project,a.work)
