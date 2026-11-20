"""Render real Gaussian orbits and synchronized dynamic five-panel review videos."""
import hashlib
import inspect
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization
import yaml

from run import atomic_json, sha
from media_publish import publication, cached_orbit, save_orbit

W,H=1920,1080
IW,IH=528,396
FPS,FRAMES=24,96
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CATEGORIES=['stereo','rgbd','pothole','missing','raised']
TITLES=['双目实拍坑洞','RGB-D 实拍坑洞','新生成坑洞','新生成缺盖窨井','新生成异常井盖']


def tensor(a):return torch.as_tensor(np.ascontiguousarray(a),device='cuda',dtype=torch.float32)
@lru_cache(maxsize=32)
def font(n):return ImageFont.truetype(FONT,n)
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


@torch.no_grad()
def orbit(model, cache):
    key=sha(Path(model['ply']))
    signature=hashlib.sha256((key+inspect.getsource(orbit)+str((IW,IH,FRAMES))).encode()).hexdigest()
    path=cache/(model['id']+'.npy')
    stamp=path.with_suffix('.json')
    cached=cached_orbit(path,stamp,signature,(FRAMES,IH,IW,3))
    if cached is not None:return cached
    a=PlyData.read(model['ply'],mmap='r')['vertex'].data
    def cols(fields):return np.stack([a[k] for k in fields],axis=1)
    xyz=cols(['x','y','z'])
    opacity=1/(1+np.exp(-a['opacity']))
    extent=max(model['width_m'],model['length_m'])
    roi=(opacity>.45)&(np.max(abs(xyz[:,:2]),axis=1)<extent*.90)
    if model['category'] in ('missing','raised'):
        roi=((roi&(xyz[:,2]>-.16)) | ((xyz[:,2]>.025)&(opacity>.45)))
    support=xyz[roi]
    if len(support)<50:raise ValueError('insufficient framing support')
    lo,hi=np.quantile(support,[.002,.998],axis=0)
    center=(lo+hi)*.5
    support=support[::max(1,len(support)//4000)]
    elevation=math.radians(58 if model['category'] in ('stereo','rgbd','pothole') else 51)
    focal=IH*1.35
    bases=[];distance=.01
    for angle in np.arange(FRAMES)/FRAMES*2*np.pi-math.pi/3:
        away=np.array([np.cos(angle)*np.cos(elevation),np.sin(angle)*np.cos(elevation),np.sin(elevation)])
        forward=-away;right=unit(np.cross(forward,[0.,0.,1.]));down=np.cross(forward,right)
        basis=np.stack([right,down,forward],axis=1)
        q=(support-center)@basis
        distance=max(distance,float(np.max(np.maximum(abs(q[:,0])*focal/(IW*.44)-q[:,2],
                                                        abs(q[:,1])*focal/(IH*.43)-q[:,2]))))
        bases.append((basis,away))
    views=[]
    for basis,away in bases:
        pose=np.eye(4);pose[:3,:3]=basis;pose[:3,3]=center+away*distance
        views.append(np.linalg.inv(pose))
    colors=np.concatenate([cols([f'f_dc_{i}' for i in range(3)])[:,None],
                           cols([f'f_rest_{i}' for i in range(9)]).reshape(-1,3,3).transpose(0,2,1)],axis=1)
    state=dict(means=tensor(xyz),quats=tensor(cols([f'rot_{i}' for i in range(4)])),
               scales=tensor(np.exp(cols([f'scale_{i}' for i in range(3)]))),
               opacities=tensor(opacity),colors=tensor(colors))
    k=tensor([[focal,0,IW*.5],[0,focal,IH*.5],[0,0,1]])[None]
    views=tensor(np.array(views));frames=[]
    for first in range(0,FRAMES,4):
        rgb,_,_=rasterization(**state,viewmats=views[first:first+4],Ks=k.expand(4,-1,-1),
             width=IW,height=IH,sh_degree=1,packed=False,
             backgrounds=tensor([[.12,.15,.19]]).expand(4,-1),rasterize_mode='antialiased')
        if not torch.isfinite(rgb).all():raise ValueError('nonfinite orbit')
        frames.append(np.uint8(rgb.clamp(0,1).cpu().numpy()*255+.5))
    frames=np.concatenate(frames)
    save_orbit(path,stamp,frames,{'ply_sha256':key,'source_and_framing_sha256':signature,'frames':FRAMES,
                       'geometry_scale':1.,'camera':'full 360 degree close inspection'})
    return frames


def pair(path):
    reader=imageio.get_reader(str(path))
    try:return [np.asarray(frame) for frame in reader]
    finally:reader.close()


@lru_cache(maxsize=256)
def calibrated_frame(config_path, frame):
    config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    data=Path(config['source_path'])
    pose=np.loadtxt(data/'extrinsics'/f'{frame:03d}_0.txt')
    fx,fy,cx,cy=np.loadtxt(data/'intrinsics/0.txt').ravel()[:4]
    return pose,fx,fy,cx,cy


def projection(job, site, frame, center):
    pose,fx,fy,cx,cy=calibrated_frame(job['configs'][site],frame)
    q=(np.array(center)-pose[:3,3])@pose[:3,:3]
    if q[2]<=0:return None
    u,v=(q[0]*fx/q[2]+cx)*.35,(q[1]*fy/q[2]+cy)*.35
    return (u,v) if 0<=u<672 and 0<=v<378 else None


def main():
    torch.set_num_threads(4)
    job=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf-8'))
    scenes=Path(job['output']);root=scenes.parent
    work=Path(job['work']);cache=work/'orbits';cache.mkdir(exist_ok=True)
    destination=root/'五类动态展示';destination.mkdir(exist_ok=True)
    plan=json.loads(Path(job['plan']).read_text(encoding='utf-8'))
    models={}
    for library in map(Path,job['libraries']):
        for m in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8')):
            models[m['id']]=dict(m,ply=str(library/m['ply']))
    cases=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(scenes.glob('*/scene.json'))]
    sites=sorted({c['site'] for c in cases})
    groups=sorted({c['id'].split('_')[1] for c in cases})
    expected={(site,group) for site in job['configs'] for group in groups}
    actual={(c['site'],c['id'].split('_')[1]) for c in cases}
    if not cases or expected!=actual:raise ValueError('Incomplete rectangular scene batch')
    registry=work/'case_plan.json'
    if registry.exists():
        expected_groups={f"{g['number']:02d}" for g in json.loads(registry.read_text(encoding='utf-8'))['groups']}
        if set(groups)!=expected_groups:raise ValueError('Scene group registry is not fully rendered')
    source_checks={};review_checks={}
    for case in cases:
        folder=scenes/case['id'];complete=folder/'complete.json'
        record=json.loads(complete.read_text(encoding='utf-8'))
        if sorted(p['category'] for p in case['placements'])!=sorted(CATEGORIES):
            raise ValueError('Five categories required in every scene')
        for name in ['scene.json','global_before_after.mp4']+[c+'_local_before_after.mp4' for c in CATEGORIES]:
            if sha(folder/name)!=record['files'][name]:raise ValueError(f'Incomplete/corrupt review input: {folder/name}')
        source_checks[case['id']]=sha(complete)
        review_checks[case['id']]={name:record['files'][name] for name in ['scene.json','global_before_after.mp4']+[c+'_local_before_after.mp4' for c in CATEGORIES]}
    with publication(destination,'播放索引.json') as stage:
        index=[]
        for category,title in zip(CATEGORIES,TITLES):
            output=stage/f'{CATEGORIES.index(category)+1:02d}_{title.replace(" ","")}_环绕与动态前后.mp4'
            segments=[]
            with imageio.get_writer(str(output),fps=FPS,codec='libx264',quality=8,macro_block_size=2,
                                    output_params=['-threads','3','-movflags','+faststart']) as video:
                for group in groups:
                    for site in sites:
                        case=next(c for c in cases if c['id']==f'{site}_{group}')
                        d=next(x for x in case['placements'] if x['category']==category)
                        model=models[d['asset_id']]
                        frames=orbit(model,cache)
                        global_frames=pair(scenes/case['id']/'global_before_after.mp4')
                        local_frames=pair(scenes/case['id']/f'{category}_local_before_after.mp4')
                        assert len(global_frames)==40 and len(local_frames)==24
                        base=Image.new('RGB',(W,H),(18,23,29));draw=ImageDraw.Draw(base)
                        draw.text((24,18),f'{title} · 原尺度 · 三种道路迁入检查',font=font(38),fill='white')
                        draw.text((24,75),f'{model["id"]}',font=font(18),fill='#b5c5d8')
                        draw.text((20,128),'资产 360° 环绕',font=font(26),fill='#7dd8cd')
                        for x,label in [(560,'全局迁入前'),(1240,'全局迁入后')]:
                            draw.text((x,112),label,font=font(28),fill='white')
                        for x,label in [(560,'局部迁入前'),(1240,'局部迁入后')]:
                            draw.text((x,600),label,font=font(28),fill='white')
                        draw.text((20,738),f'开口：{model["width_m"]*100:.1f} × {model["length_m"]*100:.1f} cm',font=font(24),fill='white')
                        depth=model.get('depth_p95_m',model.get('depth_design_m',0))
                        draw.text((20,782),f'深度：{depth*100:.1f} cm',font=font(24),fill='white')
                        draw.text((20,836),'几何缩放：1.0 ×',font=font(24),fill='#7dd8cd')
                        draw.text((20,892),'局部采用近距相机观察',font=font(21),fill='#bac8d6')
                        note={'stereo':'双目重建尺寸', 'rgbd':'名义内参估算，尺寸未经实物校准'}.get(category,'物理设计尺寸，非实物测量')
                        draw.text((20,932),note,font=font(19),fill='#bac8d6')
                        draw.text((560,552),f'道路 {site} · {plan["scene_titles"][site]}',font=font(22),fill='#bac8d6')
                        draw.text((560,1039),'前后使用同一相机、同一时刻；全局播放连续 40 个真实渲染帧',font=font(21),fill='#bac8d6')
                        start=plan['drive_start_by_site'][site]
                        for frame in range(FRAMES):
                            image=base.copy();canvas=ImageDraw.Draw(image)
                            image.paste(Image.fromarray(frames[frame]),(16,280))
                            gi=frame*40//FRAMES;li=frame*24//FRAMES
                            for x,half in [(560,0),(1240,1)]:
                                gf=Image.fromarray(global_frames[gi][:,half*960:(half+1)*960]).resize((672,378),Image.Resampling.LANCZOS)
                                image.paste(gf,(x,160))
                                lf=Image.fromarray(local_frames[li][:,half*960:(half+1)*960]).resize((672,378),Image.Resampling.LANCZOS)
                                image.paste(lf,(x,650))
                                uv=projection(job,site,start+gi,d['center_world'])
                                if uv:
                                    u,v=uv[0]+x,uv[1]+160
                                    canvas.ellipse((u-14,v-14,u+14,v+14),outline='#ffe072',width=2)
                                else:canvas.text((x+12,170),'目标已离开视野',font=font(18),fill='#ffe072')
                            canvas.text((1700,550),f'源帧 {start+gi:02d}',font=font(20),fill='#bac8d6')
                            video.append_data(np.asarray(image))
                            if group==groups[0] and frame==0:
                                image.save(stage/f'{category}_{site}_预览.jpg',quality=93)
                        segments.append({'case':case['id'],'asset_id':model['id'],'site':site,
                                         'start_seconds':(len(segments))*4,'seconds':4,'geometry_scale':1.,
                                         'global_source_frames':[start,start+39]})
            index.append({'category':category,'title':title,'file':output.name,'seconds':len(segments)*4,
                          'segments':segments,'unique_assets':len({s['asset_id'] for s in segments})})
            print('PRESENTED',category,len(segments),'segments',flush=True)
        atomic_json(stage/'播放索引.json',{'videos':index,'source_complete_sha256':source_checks,
                    'source_review_files_sha256':review_checks})


if __name__=='__main__':main()
