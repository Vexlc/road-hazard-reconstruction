"""Bring the inspection camera closer without changing the measured hole geometry."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
import render_scenes as r

CACHE=r.WORK/'composite4/stereo'

def camera(anchor,meta,angle):
    size=max(meta['width_m'],meta['length_m'])
    distance=max(.075,size*1.25);height=max(.095,size*1.35)
    target=anchor.center-anchor.basis[:,2]*min(.012,meta['depth_p95_m']*.25)
    eye=anchor.center-anchor.basis[:,1]*distance*np.cos(angle)+anchor.basis[:,0]*distance*np.sin(angle)+anchor.basis[:,2]*height
    return r.rb.camera_at(anchor.reference,eye,target,(1280,720),1280*.91)

@torch.no_grad()
def main():
    torch.set_num_threads(4);site=Path(r.cfg.source_path).name
    models={m['id']:m for m in json.loads((r.rb.LIBRARY/'asset_manifest.json').read_text(encoding='utf-8'))}
    scene=r.create_scene();g=scene.gaussians;renderer=r.StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame']);r.rb.resize_cameras(cameras)
    source={k:getattr(g.background,a) for k,a in r.rb.ATTRS.items()}
    for i in range(1,11):
        folder=r.OUT/'迁入场景'/f'{site}_{i:02d}';dest=CACHE/f'{site}_{i:02d}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        record=json.loads((folder/'scene.json').read_text(encoding='utf-8'));d=next(d for d in record['placements'] if d['category']=='stereo')
        m=models[d['asset_id']]
        anchor=SimpleNamespace(center=np.array(d['center_world']),basis=np.array(d['basis_columns']),reference=cameras[d['source_reference_frame']])
        angles=np.linspace(-.45,.45,96)
        r.rb.set_background(g.background,source)
        with r.write_video(dest/'before.mp4') as video:
            for angle in angles:video.append_data(r.rb.numpy_image(renderer.render(camera(anchor,m,float(angle)),g)))
        checkpoint=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cuda',weights_only=False)
        inserted={k:checkpoint['background'][k] for k in r.rb.ATTRS};r.rb.set_background(g.background,inserted)
        with r.write_video(dest/'after.mp4') as video:
            for j,angle in enumerate(angles):
                picture=r.rb.numpy_image(renderer.render(camera(anchor,m,float(angle)),g));video.append_data(picture)
                if j==48:Image.fromarray(picture).save(dest/'after.jpg',quality=95)
        r.rb.set_background(g.background,source);del checkpoint,inserted
        (dest/'complete.json').write_text(json.dumps({'asset_id':d['asset_id'],'scene':site,'frames':96,'fps':24,
             'geometry_scale':1.,'camera_distance_m':max(.075,max(m['width_m'],m['length_m'])*1.25),
             'camera_height_m':max(.095,max(m['width_m'],m['length_m'])*1.35),
             'matched_before_after_cameras':True},indent=2),encoding='utf-8')
        print('STEREO DETAIL',site,i,flush=True)

if __name__=='__main__':main()
