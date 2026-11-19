"""Reload one saved native scene and reproduce all five inspection views."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

import render_scenes as r

@torch.no_grad()
def main():
    scene=r.create_scene();renderer=r.StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    r.rb.resize_cameras(cameras)
    dest=Path(r.cfg.model_path)
    details=json.loads((dest/'scene.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((r.rb.LIBRARY/'asset_manifest.json').read_text(encoding='utf-8'))}
    comparisons=[]
    for d in details['placements']:
        a=SimpleNamespace(center=np.array(d['center_world']),basis=np.array(d['basis_columns']),
                          reference=cameras[d['source_reference_frame']])
        picture=r.rb.numpy_image(renderer.render(r.near_camera(a,models[d['asset_id']],0),scene.gaussians))
        saved=np.asarray(Image.open(dest/f'{d["category"]}_after.jpg')).astype('f4')
        error=float(np.mean(np.abs(picture.astype('f4')-saved)))
        assert error<2.5,(d['asset_id'],error)
        comparisons.append({'asset_id':d['asset_id'],'mean_abs_pixel_difference_from_saved_jpeg':error})
    (dest/'reload_validation.json').write_text(json.dumps({'native_checkpoint_reload':True,'views':comparisons},indent=2)+'\n',encoding='utf-8')
    print('RELOAD VERIFIED',dest.name,comparisons,flush=True)

if __name__=='__main__':main()
