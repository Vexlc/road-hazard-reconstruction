"""Place the same fifty assets into three freshly reconstructed road scenes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from filelock import FileLock

PROJECT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).parent
OUT=PROJECT/'outputs/道路病害_多场景迁入'
sys.path.insert(0,str(PROJECT/'outputs/road_hazard_joint_scenes/reproduction'))
import render_batches as rb
from create_scene import create_scene
from export_reconstruction import export_splat
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.utils.loss_utils import ssim
from street_gaussian.utils.lpipsPyTorch import lpips

rb.WIDTH,rb.HEIGHT=1280,720
rb.WORK=WORK
rb.OUTPUT=OUT/'迁入场景'
FRAMES=72
FPS=24

def write_video(path,fps=FPS):
    return imageio.get_writer(str(path),fps=fps,codec='libx264',quality=8,macro_block_size=2,
                             output_params=['-threads','3','-movflags','+faststart'])

def hash_state(state):
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x): h.update(k.encode());walk(x[k])
        elif isinstance(x,(list,tuple)):
            for y in x:walk(y)
        else: h.update(repr(x).encode())
    walk({k:v for k,v in state.items() if k not in ('background','iter')})
    return h.hexdigest()

def near_camera(a,m,angle=0):
    e=max(m['width_m'],m['length_m'])
    if m['category'] in ('missing','raised'):
        return rb.near_camera(a,m,960,540,angle)
    # The stereo captures are only centimetres wide; inspect them at their true size.
    distance=max(.24,e*1.48);rise=max(.29,e*1.32)
    target=a.center+a.basis[:,2]*(-.006)
    position=a.center+a.basis[:,1]*(-distance*np.cos(angle))+a.basis[:,0]*(distance*np.sin(angle))+a.basis[:,2]*rise
    return rb.camera_at(a.reference,position,target,(960,540),960*.91)

@torch.no_grad()
def reconstruction_export(scene,renderer,cameras,site):
    folder=OUT/'重建场景'/site;folder.mkdir(parents=True,exist_ok=True)
    frames_dir=folder/'frames';frames_dir.mkdir(exist_ok=True)
    drive=[];metrics=[]
    with write_video(folder/'重建行车.mp4',10) as video,write_video(folder/'原图与重建对照.mp4',10) as compare:
        for i,cam in enumerate(cameras):
            result=renderer.render(cam,scene.gaussians)
            rgb=result['rgb'].clamp(0,1)
            original=F.interpolate(cam.original_image[None].to('cuda'),size=(720,1280),mode='bilinear',align_corners=False)[0]
            picture=rb.numpy_image(result);truth=np.uint8(original.permute(1,2,0).clamp(0,1).cpu().numpy()*255+.5)
            video.append_data(picture);compare.append_data(np.concatenate([truth,picture],axis=1))
            drive.append(picture)
            Image.fromarray(picture).save(frames_dir/f'{i:03d}.jpg',quality=93)
            if i in (20,40,60):Image.fromarray(np.concatenate([truth,picture],1)).save(folder/f'原图与重建_{i:03d}.jpg',quality=95)
            if cam.meta['is_val']:
                # Metrics use native training resolution, avoiding upsampled comparisons.
                gt=cam.original_image.to('cuda')
                native=copy.copy(cam)
                native.world_view_transform=cam.world_view_transform.clone()
                native.image_height,native.image_width=gt.shape[-2:]
                intr=cam.K.clone();intr[0]*=native.image_width/cam.image_width;intr[1]*=native.image_height/cam.image_height
                native.set_intrinsic(intr)
                small=renderer.render(native,scene.gaussians)['rgb'].clamp(0,1)
                metrics.append({'frame':i,'psnr':float(-10*torch.log10((small-gt).square().mean())),
                                'ssim':float(ssim(small,gt)),'lpips':float(lpips(small*2-1,gt*2-1))})
    summary={'scene':site,'iterations':7000,'stage':'initial dynamic 3DGS; no diffusion distillation',
             'training_resolution':[960,540],'export_resolution':[1280,720],
             'train_views':len(scene.getTrainCameras()),'held_out_views':len(metrics),
             'moving_objects':len(scene.dataset.metadata['obj_meta']),
             'metric_note':'Within-sequence interpolation; full sequence LiDAR/color initializes geometry.',
             'mean':{k:float(np.mean([r[k] for r in metrics])) for k in ['psnr','ssim','lpips']},'per_frame':metrics}
    (folder/'重建指标.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return drive,summary

@torch.no_grad()
def render_batch(scene,renderer,cameras,source,models,anchors,batch,baseline,source_hash):
    dest=rb.OUTPUT/batch['id'];dest.mkdir(parents=True,exist_ok=True)
    g=scene.gaussians;bg=g.background
    rb.set_background(bg,source)
    keep=torch.ones(len(source['xyz']),device='cuda',dtype=torch.bool)
    additions=[];details=[];before=[]
    for slot,name in enumerate(batch['assets']):
        m=models[name];a=anchors[slot]
        asset,apron,cut,exposure=a.fit_asset(m)
        keep[torch.as_tensor(cut,device='cuda',dtype=torch.long)]=False
        camera=near_camera(a,m,0)
        before.append(rb.numpy_image(renderer.render(camera,g)))
        additions.extend([asset,apron])
        details.append({'asset_id':name,'category':m['category'],'size_multiplier':1.0,'exposure_gain':exposure,
                        'asset_points':len(asset['xyz']),'apron_points':len(apron['xyz']),'cut_points':len(cut),
                        'road_color_coverage':a.coverage,'width_m':m['width_m'],'length_m':m['length_m'],
                        'depth_m':m.get('depth_p95_m',m.get('depth_design_m')),**a.record})
    kept=int(keep.sum())
    merged={k:torch.nn.Parameter(torch.cat([source[k][keep]]+[s[k] for s in additions]),requires_grad=False) for k in rb.ATTRS}
    offset=kept
    for i,d in enumerate(details):
        n=d['asset_points'];d['merged_asset_offset']=offset
        local=rb.load_vertices(rb.LIBRARY/models[d['asset_id']]['ply'],['x','y','z'])
        world=local@anchors[i].basis.T+anchors[i].center
        error=np.max(np.abs(merged['xyz'][offset:offset+n].cpu().numpy()-world))
        d['world_position_max_error_m']=float(error)
        assert error<2e-5,(d['asset_id'],error)
        offset+=n+d['apron_points']
    rb.set_background(bg,merged)
    for i,name in enumerate(batch['assets']):
        m=models[name];a=anchors[i];cat=m['category']
        after=rb.numpy_image(renderer.render(near_camera(a,m,0),g))
        diff=np.abs(after.astype('f4')-before[i]).mean(2)
        d=details[i];d['changed_pixels']=int((diff>8).sum())
        Image.fromarray(before[i]).save(dest/f'{cat}_before.jpg',quality=94)
        Image.fromarray(after).save(dest/f'{cat}_after.jpg',quality=94)
        with write_video(dest/f'{cat}_close.mp4') as video:
            for angle in np.linspace(-.38,.65 if cat=='raised' else .38,FRAMES):
                video.append_data(rb.numpy_image(renderer.render(near_camera(a,m,float(angle)),g)))
    with write_video(dest/'行车迁入前后.mp4',10) as compare,write_video(dest/'行车迁入后.mp4',10) as drive:
        drive_start=0 if batch['site']=='110' else 20
        for frame in range(drive_start,drive_start+40):
            picture=rb.numpy_image(renderer.render(cameras[frame],g))
            drive.append_data(picture);compare.append_data(np.concatenate([baseline[frame],picture],axis=0))
            if frame in (drive_start+10,drive_start+25):Image.fromarray(picture).save(dest/f'drive_{frame}.jpg',quality=94)
    state=g.save_state_dict(is_final=True)
    assert hash_state(state)==source_hash,'Original actor/sky parameters changed'
    state['iter']=7001
    model=dest/'trained_model';model.mkdir(exist_ok=True)
    torch.save(state,model/'iteration_7000.pth')
    config=WORK/f'pandaset_{batch["site"]}.yaml'
    (dest/'render_config.yaml').write_text(yaml.safe_dump({'parent_cfg':str(config),'model_path':str(dest),
         'mode':'trajectory','loaded_iter':7000,'resolution':1280},allow_unicode=False),encoding='utf-8')
    if batch['id'].endswith('_01'):
        export_splat(g,cameras[40],dest/'scene_frame040.ply')
    summary={**batch,'insertion_revision':2,'placements':details,'background_points_retained':kept,'merged_background_points':len(merged['xyz']),
             'source_actor_sky_sha256':source_hash,'actors_sky_unchanged':True,
             'source_config':str(config),'checkpoint':str(model/'iteration_7000.pth'),
             'appearance':'native Gaussian geometry; host road color baked from calibrated RGB and rendered depth; no final 2D compositing'}
    (dest/'scene.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rb.set_background(bg,source)

@torch.no_grad()
def render_main():
    site=Path(cfg.source_path).name
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((rb.LIBRARY/'asset_manifest.json').read_text(encoding='utf-8'))}
    scene=create_scene();renderer=StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    rb.resize_cameras(cameras)
    baseline,metrics=reconstruction_export(scene,renderer,cameras,site)
    bg=scene.gaussians.background
    source={k:getattr(bg,v) for k,v in rb.ATTRS.items()}
    source_hash=hash_state(scene.gaussians.save_state_dict(is_final=True))
    teachers=[]
    for f in range(0,80,3):
        cam=cameras[f];result=renderer.render(cam,scene.gaussians)
        rgb=F.interpolate(cam.original_image[None].to('cuda'),size=(720,1280),mode='bilinear',align_corners=False)
        teachers.append((cam.get_extrinsic(world2cam=True).clone(),cam.K.clone(),rgb,result['depth'][None].clone()))
    records=plan['slots_by_site'][site]
    centers=np.array([r['center_world'][:2] for r in records])
    world=source['xyz'].cpu().numpy();spread=source['scaling'].exp().max(1).values.cpu().numpy()
    anchors=[rb.Anchor(r,centers,cameras,world,spread,teachers) for r in records]
    for i,(a,r) in enumerate(zip(anchors,records)):
        a.reference=cameras[r['source_reference_frame']]
        if plan['category_order'][i] in ('missing','raised'):
            # Monocular background Gaussians can drift below the measured road.
            # Remove these old surfaces throughout the new shaft, not only at its lip.
            local=(world-a.center)@a.basis
            mask=(np.max(np.abs(local[:,:2]),axis=1)<4.25)&(local[:,2]>-1.70)&(local[:,2]<.50)
            a.parent_ids=np.flatnonzero(mask)
            a.parent_local=local[mask]
            a.parent_radius=np.minimum(3.*spread[mask]+.04,.91)
    del teachers,result
    print('ANCHORS READY',site,[round(a.coverage,3) for a in anchors],flush=True)
    requested=set(os.environ.get('ROAD_ONLY_BATCHES','').split(','))-{''}
    for batch in plan['scenes']:
        if batch['site']!=site or (requested and batch['id'] not in requested):continue
        previous=rb.OUTPUT/batch['id']/'scene.json'
        if previous.exists() and json.loads(previous.read_text(encoding='utf-8')).get('insertion_revision')==2:continue
        t=time.time();render_batch(scene,renderer,cameras,source,models,anchors,batch,baseline,source_hash)
        print('DONE',batch['id'],round(time.time()-t,2),flush=True)
    print('SCENE COMPLETE',site,metrics['mean'],flush=True)

def main():
    site=Path(cfg.source_path).name
    with FileLock(str(WORK/f'render_{site}.lock')):
        plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
        requested=set(os.environ.get('ROAD_ONLY_BATCHES','').split(','))-{''}
        batches=[b for b in plan['scenes'] if b['site']==site and (not requested or b['id'] in requested)]
        def current(b):
            p=rb.OUTPUT/b['id']/'scene.json'
            return p.exists() and json.loads(p.read_text(encoding='utf-8')).get('insertion_revision')==2
        if all(current(b) for b in batches):
            print('ALREADY RENDERED',site,flush=True);return
        render_main()

if __name__=='__main__':main()
