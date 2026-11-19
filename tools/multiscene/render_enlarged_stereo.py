"""Create a separate demonstration with stereo asset geometry enlarged tenfold."""
import copy
import json
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
import render_scenes as r

SCALE=10.
ORIGINAL_OUT=r.OUT
NEW_OUT=r.OUT/'双目坑洞_几何放大10倍'
r.OUT=NEW_OUT;r.rb.OUTPUT=NEW_OUT/'迁入场景'
original_read=r.rb.read_asset
original_vertices=r.rb.load_vertices

def scaled_read(meta):
    vertices,surface=original_read(meta)
    if meta['category']=='stereo':
        vertices=vertices.copy();vertices[:,:3]*=SCALE;vertices[:,16:19]+=np.log(SCALE)
        surface=dict(surface)
        for key in ['width','length','height']:surface[key]=surface[key]*SCALE
    return vertices,surface

def scaled_vertices(path,fields):
    value=original_vertices(path,fields)
    if Path(path).parent.name.startswith('stereo_pothole_') and fields==['x','y','z']:value*=SCALE
    return value

r.rb.read_asset=scaled_read
r.rb.load_vertices=scaled_vertices

@torch.no_grad()
def main():
    torch.set_num_threads(4);site=Path(r.cfg.source_path).name
    with FileLock(str(r.WORK/f'enlarged10_{site}.lock')):
        plan=json.loads((r.WORK/'plan.json').read_text(encoding='utf-8'))
        models={m['id']:m for m in json.loads((r.rb.LIBRARY/'asset_manifest.json').read_text(encoding='utf-8'))}
        for m in models.values():
            if m['category']=='stereo':
                for key in ['width_m','length_m','depth_p95_m','metric_border_m']:
                    if key in m:m['original_'+key]=m[key];m[key]*=SCALE
        selected={name:models[name] for name in plan['selected_assets']['stereo']}
        NEW_OUT.mkdir(parents=True,exist_ok=True)
        sizes_file=NEW_OUT/'放大资产尺寸.json'
        if not sizes_file.exists():sizes_file.write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
        scene=r.create_scene();g=scene.gaussians;renderer=r.StreetGaussianRenderer()
        cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame']);r.rb.resize_cameras(cameras)
        bg=g.background;source={k:getattr(bg,v) for k,v in r.rb.ATTRS.items()}
        source_hash=r.hash_state(g.save_state_dict(is_final=True))
        baseline=[r.rb.numpy_image(renderer.render(cam,g)) for cam in cameras]
        teachers=[]
        for frame in range(0,80,3):
            cam=cameras[frame];result=renderer.render(cam,g)
            rgb=F.interpolate(cam.original_image[None].to('cuda'),size=(720,1280),mode='bilinear',align_corners=False)
            teachers.append((cam.get_extrinsic(world2cam=True).clone(),cam.K.clone(),rgb,result['depth'][None].clone()))
        records=plan['slots_by_site'][site];centers=np.array([d['center_world'][:2] for d in records])
        world=source['xyz'].detach().cpu().numpy();spread=source['scaling'].exp().max(1).values.cpu().numpy()
        anchors=[r.rb.Anchor(rec,centers,cameras,world,spread,teachers) for rec in records]
        for i,(a,record) in enumerate(zip(anchors,records)):
            a.reference=cameras[record['source_reference_frame']]
            if plan['category_order'][i] in ['stereo','missing','raised']:
                local=(world-a.center)@a.basis
                mask=(np.max(abs(local[:,:2]),axis=1)<4.25)&(local[:,2]>-1.7)&(local[:,2]<.50)
                a.parent_ids=np.flatnonzero(mask);a.parent_local=local[mask];a.parent_radius=np.minimum(3.*spread[mask]+.04,.91)
        del teachers,result
        requested=set(os.environ.get('SCALED_ONLY_BATCHES','').split(','))-{''}
        for batch in plan['scenes']:
            if batch['site']!=site or (requested and batch['id'] not in requested):continue
            dest=r.rb.OUTPUT/batch['id'];summary_file=dest/'scene.json'
            if summary_file.exists() and json.loads(summary_file.read_text(encoding='utf-8')).get('stereo_geometry_scale')==SCALE:continue
            name=batch['assets'][0];meta=models[name];anchor=anchors[0]
            dest.mkdir(parents=True,exist_ok=True);r.rb.set_background(bg,source)
            with r.write_video(dest/'stereo_local_before.mp4') as video:
                for angle in np.linspace(-.38,.38,r.FRAMES):
                    video.append_data(r.rb.numpy_image(renderer.render(r.near_camera(anchor,meta,float(angle)),g)))
            start=time.time();r.render_batch(scene,renderer,cameras,source,models,anchors,batch,baseline,source_hash)
            summary=json.loads(summary_file.read_text(encoding='utf-8'))
            summary['stereo_geometry_scale']=SCALE
            summary['scale_note']='Stereo-derived geometry uniformly enlarged 10x for demonstration; not original measured dimensions.'
            for d in summary['placements']:
                if d['category']=='stereo':
                    d['size_multiplier']=SCALE
                    d['original_width_m']=meta['original_width_m'];d['original_length_m']=meta['original_length_m'];d['original_depth_m']=meta['original_depth_p95_m']
            summary_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            print('ENLARGED',batch['id'],round(time.time()-start,1),'opening_m',meta['width_m'],meta['length_m'],flush=True)

if __name__=='__main__':main()
