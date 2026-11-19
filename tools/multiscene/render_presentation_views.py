"""Render matched global and local source views for the five-panel presentation."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

import render_scenes as r

CACHE=r.WORK/'composite3/views'

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    site=Path(r.cfg.source_path).name
    plan=json.loads((r.WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((r.rb.LIBRARY/'asset_manifest.json').read_text(encoding='utf-8'))}
    scene=r.create_scene();renderer=r.StreetGaussianRenderer();g=scene.gaussians
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    r.rb.resize_cameras(cameras)
    source={k:getattr(g.background,a) for k,a in r.rb.ATTRS.items()}
    for batch in [b for b in plan['scenes'] if b['site']==site]:
        folder=r.OUT/'迁入场景'/batch['id']
        record=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
        assert record['insertion_revision']==2
        dest=CACHE/batch['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        r.rb.set_background(g.background,source)
        views=[]
        for d in record['placements']:
            m=models[d['asset_id']];cat=m['category']
            anchor=SimpleNamespace(center=np.array(d['center_world']),basis=np.array(d['basis_columns']),
                                   reference=cameras[d['source_reference_frame']])
            global_cam=cameras[d['source_reference_frame']]
            original=r.rb.numpy_image(renderer.render(global_cam,g))
            Image.fromarray(original).save(dest/f'{cat}_global_before.jpg',quality=96)
            with r.write_video(dest/f'{cat}_local_before.mp4') as video:
                for angle in np.linspace(-.38,.65 if cat=='raised' else .38,r.FRAMES):
                    video.append_data(r.rb.numpy_image(renderer.render(r.near_camera(anchor,m,float(angle)),g)))
            point=torch.as_tensor(d['center_world'],dtype=torch.float32,device='cuda')
            w=global_cam.get_extrinsic(world2cam=True)
            q=w[:3,:3]@point+w[:3,3]
            uv=global_cam.K@q;uv=uv[:2]/uv[2]
            views.append({'asset_id':d['asset_id'],'category':cat,'frame':d['source_reference_frame'],
                          'global_center_pixels':uv.cpu().tolist(),'global_resolution':[1280,720],
                          'local_camera_angles':[-.38,.65 if cat=='raised' else .38],
                          'local_camera_frames':72,'local_camera_fps':24})
        checkpoint=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cuda',weights_only=False)
        inserted={k:checkpoint['background'][k] for k in r.rb.ATTRS}
        r.rb.set_background(g.background,inserted)
        for d in record['placements']:
            picture=r.rb.numpy_image(renderer.render(cameras[d['source_reference_frame']],g))
            Image.fromarray(picture).save(dest/f'{d["category"]}_global_after.jpg',quality=96)
        r.rb.set_background(g.background,source)
        del checkpoint,inserted
        (dest/'complete.json').write_text(json.dumps({'scene':site,'batch':batch['id'],'views':views,
             'global_matching':'same original calibrated camera and original time before/after',
             'local_matching':'same camera sweep as the existing verified local after video'},indent=2),encoding='utf-8')
        print('MATCHED VIEWS',batch['id'],flush=True)

if __name__=='__main__':main()
