"""Replay a placed pit on calibrated front/rear photographs and compare matched road frames."""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gen2hole/downstream/sensors'))
from camera_geometry import camera_hits, choices, foreground, track
from render_pairs import gather, render
from road_geometry import read_json, save_json, sha256
from sunlight import lighting


@lru_cache()
def font(size):
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size,index=2)


def road_shadow(source, depth, part, alpha, protected, pose, K, rect, placement, site, camera):
    """Carry cast shadows from the photographed road plane down to visible points on the pit."""
    result=np.zeros(part.shape,'f4')
    sun_info=lighting()['sites'][site]
    if not sun_info['direct_light_used']:
        return result
    soil=(part==2)&(alpha>.05)&~protected
    road=(part==1)&~protected
    if road.sum()<100 or not soil.any():
        return result
    left,top,right,bottom=rect
    b=np.asarray(placement['basis_columns']); centre=np.array(placement['center_world'])
    sun=np.asarray(sun_info['sun_world'])@b
    if sun[2]<=.1:
        return result
    # The photographed brightness under a pole/cable shadow is sampled at the
    # sunward intersection with the original road, accounting for the bed depth.
    y,x=np.where(soil)
    rays=np.column_stack([(x+left+.5-K[0,2])/K[0,0],(y+top+.5-K[1,2])/K[1,1],np.ones(len(y))])
    local=(pose[:3,3]-centre)@b+(rays@pose[:3,:3].T@b)*depth[y,x,None]
    entry=local-local[:,2,None]*sun/sun[2]
    cam=(entry@b.T+centre-pose[:3,3])@pose[:3,:3]
    uv=cam[:,:2]/cam[:,2:]*[K[0,0],K[1,1]]+K[:2,2]-[.5,.5]
    src=np.asarray(source)
    # Small branches are preserved. Use soft transitions and require the chroma
    # shift measured in shaded asphalt, so black tar seams do not become shadows.
    u=np.clip(np.rint(uv[:,0]).astype(int),0,src.shape[1]-1)
    v=np.clip(np.rint(uv[:,1]).astype(int),0,src.shape[0]-1)
    smooth=cv2.GaussianBlur(src,(3,3),.72)
    sample=smooth[v,u].astype('f4')
    nearby=smooth[top:bottom,left:right][road].astype('f4')
    lo,hi=np.quantile(nearby.mean(1),[.46,.83])
    light=np.median(nearby[(nearby.mean(1)>=lo)&(nearby.mean(1)<=hi)],axis=0)
    ratio=sample/np.maximum(light,1)
    shade=np.array(lighting()['contrast'][camera]['measured_shade_to_sun_srgb'])
    dark=np.clip((.91-ratio.mean(1))/(.91-shade.mean()),0,1)
    blue=np.clip((ratio[:,2]-ratio[:,0]-.008)/(max(.02,shade[2]-shade[0])*.55),0,1)
    coverage=dark*blue*alpha[y,x]
    result[y,x]=coverage
    return result


def draw_label(image,text,xy=(22,20),size=25):
    x,y=xy
    d=ImageDraw.Draw(image)
    bounds=d.textbbox((x+10,y+6),text,font=font(size))
    d.rectangle((x,y,bounds[2]+12,bounds[3]+9),fill='#152224')
    d.text((x+10,y+6),text,font=font(size),fill='white')


def view_plate(before,after,box,site,key):
    # Use identical camera crops in the lower pair. The clean full frame is also saved separately.
    w,h=before.size
    x,y,r,b=box
    cw=min(w,max(r-x+150,int((b-y+70)*2.30)));ch=min(h,int(cw/2.30))
    left=max(0,min(w-cw,(x+r-cw)//2));top=max(0,min(h-ch,(y+b-ch)//2))
    crop=(left,top,left+cw,top+ch)
    board=Image.new('RGB',(1560,920),'#f3f2ed')
    d=ImageDraw.Draw(board)
    d.text((28,14),f'{key}   /   PandaSet {site} 前视',font=font(35),fill='#1b2c2d')
    for i,(im,tag) in enumerate([(before,'原路面'),(after,'加入 '+key)]):
        col=20+770*i
        d.text((col+3,83),tag,font=font(25),fill='#1b2c2d')
        board.paste(im.resize((750,422),Image.Resampling.LANCZOS),(col,132))
        board.paste(im.crop(crop).resize((750,326),Image.Resampling.LANCZOS),(col,580))
    return board


def review(a):
    cv2.setNumThreads(2)
    bundle=a.bundle.resolve();scene=bundle/'scenes'/a.case
    info=read_json(scene/'scene.json');(placement,)=info['placements']
    key,site=placement['asset_id'],info['site']
    if placement['category']!='pothole' or placement['size_multiplier']!=1:
        raise ValueError('Expected one original-scale pit')
    folder=bundle/'assets/pothole'/key
    if placement['asset_ply_sha256']!=sha256(folder/'asset.ply') or not info['checkpoint_reloaded']:
        raise ValueError('Placement and delivered asset mismatch')
    metadata=read_json(folder/'metadata.json')
    model={key:(folder,metadata)}
    provenance={'generated':'生成参考图','photograph':'实拍参考图'}.get(metadata.get('reference_kind'))
    credit=f"{key} · {provenance}  " if provenance else ''
    points,models,extent=gather(scene/'trained_model/iteration_7000.pth',[placement],model)
    finish=None
    if a.surface_lighting:
        sys.path.insert(0,str(ROOT/'gen2hole'))
        from large_pits.daylight import PitLight
        finish=PitLight(folder/'asset.glb',models[0],lighting()['sites'][site])
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    record=dict(site=site,asset=key,case=a.case,size_multiplier=placement['size_multiplier'],
                glb_sha256=sha256(folder/'asset.glb'),ply_sha256=placement['asset_ply_sha256'],
                source_checkpoint_sha256=read_json(scene/'complete.json')['files']['trained_model/iteration_7000.pth'],
                centre_world=placement['center_world'],basis_columns=placement['basis_columns'],
                reload_max_pixel_error=placement['reload_max_pixel_error'],frames={},selected={})
    if provenance:
        record['reference_kind']=metadata['reference_kind']
        record['reference_source']=metadata['source']
    if finish:
        record['surface_lighting']=dict(material='GLB colour and tangent normals, three-tap anisotropic filter',
            sky_horizon_azimuths=16,sky_horizon_radial_samples=26,
            sun=lighting()['sites'][site],solar_disc_rays=len(finish.directions),receiver_bias_m=.001)
    if a.scans and read_json(a.scans/'base'/site/'source.json')['site']!=site:
        raise ValueError('Unexpected LiDAR site')
    # The two excerpts are contiguous within each camera and retain the source cadence.
    fps=10;size=(1440,810)
    report_frames=record['frames'];stills={}
    with imageio.get_writer(str(out/'drive.stage.mp4'),fps=fps,codec='libx264',quality=8,macro_block_size=2,
                            output_params=['-threads','3','-movflags','+faststart']) as movie:
        for name,title in [('front_camera','前视 · 驶近'),('back_camera','后视 · 驶离')]:
            _,poses,K,(w,h)=track(str(a.raw),site,name)
            ranked=choices(str(a.raw),site,placement,extent[0],[name])
            bounds={f:bb for _,_,f,bb in ranked}
            if not bounds:
                raise ValueError('No in-frame road views for '+name)
            peak=ranked[0][2]
            numbers=(range(max(min(bounds),peak-28),peak+1) if name=='front_camera'
                     else range(peak,min(max(bounds),peak+24)+1))
            best=0;shots=[];report_frames[name]=[]
            for fr in numbers:
                if fr not in bounds:
                    raise ValueError('Discontinuous in-frame shot')
                bb=bounds[fr]
                l,t=np.maximum(np.floor(bb[:2]).astype(int)-22,0)
                r,b=np.minimum(np.ceil(bb[2:]).astype(int)+23,[w,h])
                l,t,r,b=map(int,(l,t,r,b))
                rect=(l,t,r,b)
                pose=poses[fr]
                depth,part,which,_=camera_hits(pose,K,rect,models)
                dynamic,static,ego=foreground(str(a.raw),str(a.processed),a.scans,site,name,fr,rect,depth)
                protected=dynamic|static|ego
                path=a.raw/site/'camera'/name/f'{fr:02d}.jpg'
                with Image.open(path) as im:before=im.convert('RGB')
                pixels=np.asarray(before)
                src=pixels[t:b,l:r]
                after,alpha,_,_=render(points,pose,K,rect,src,depth,part,which,protected,models,site,name)
                cast=road_shadow(pixels,depth,part,alpha,protected,pose,K,rect,placement,site,name)
                shade=np.asarray(lighting()['contrast'][name]['measured_shade_to_sun_srgb'])
                if finish:
                    own=finish.apply(after,depth,part,alpha,protected,pose,K,rect,cast,shade)
                else:
                    own=np.zeros_like(cast)
                    shaded=cast>0
                    after[shaded]=np.uint8(np.clip(np.rint(after[shaded]*(1-cast[shaded,None]*(1-shade))),0,255))
                # Retain only the true opening and contact fringe, including one-pixel antialiasing.
                if not np.array_equal(src[protected],after[protected]):
                    raise ValueError('Foreground was overwritten')
                fringe=cv2.dilate((part==2).astype('u1'),np.ones((3,3),'u1'))>0
                outside=(part==0)&~fringe
                outside_edits=int(np.any(src[outside]!=after[outside],axis=1).sum())
                if outside_edits:
                    raise ValueError('Edited pixels outside the mesh and antialias fringe')
                composite=before.copy();composite.paste(Image.fromarray(after),(l,t))
                mask=(part==2)&~protected&(alpha>.48)
                count=int(mask.sum())
                if count:
                    y,x=np.where(mask);box=[l+int(x.min()),t+int(y.min()),l+int(x.max())+1,t+int(y.max())+1]
                else:box=[]
                report_frames[name].append(dict(frame=fr,bbox=box,visible_bed_pixels=count,
                    protected_target_pixels=int(((part==2)&protected).sum()),shadowed_bed_pixels=int((cast>.35).sum()),
                    self_shadow_pixels=int((own>.35).sum()),outside_changed_pixels=outside_edits,
                    protected_changed_pixels=0))
                frame=composite.resize(size,Image.Resampling.LANCZOS);draw_label(frame,credit+title)
                shots.append(frame)
                if count>best:
                    best=count;stills[name]=(before,composite,box,fr,path)
            if not best:
                raise ValueError('No visible asset in '+name)
            before,composite,box,fr,path=stills[name]
            stem=name.split('_')[0]
            before.save(out/(stem+'-before.webp'),lossless=True,method=2)
            composite.save(out/(stem+'-after.webp'),lossless=True,method=2)
            record['selected'][name]=dict(frame=fr,bbox=box,visible_bed_pixels=best,original_sha256=sha256(path))
            for frame in shots:movie.append_data(np.asarray(frame))
            if stem=='front':
                hold=composite.resize(size,Image.Resampling.LANCZOS);draw_label(hold,credit+'前视 · 近处定格')
                for _ in range(13):movie.append_data(np.asarray(hold))
                plate=view_plate(before,composite,box,site,key);plate.save(out/'comparison.jpg',quality=96,subsampling=0)
                compare=Image.new('RGB',size,'#f3f2ed')
                d=ImageDraw.Draw(compare)
                d.text((25,23),credit+'同一帧路面对比',font=font(34),fill='#22302e')
                left,top,right,bottom=box
                ww=min(w,int((right-left)*1.36));hh=min(h,round(ww/1.73))
                xx=int(np.clip((left+right-ww)//2,0,w-ww));yy=int(np.clip((top+bottom-hh)//2,0,h-hh))
                for j,(im,label) in enumerate(((before,'原路面'),(composite,'加入 '+key))):
                    crop=im.crop((xx,yy,xx+ww,yy+hh)).resize((698,404),Image.Resampling.LANCZOS)
                    compare.paste(crop,(15+712*j,209))
                    d.text((18+712*j,145),label,font=font(28),fill='#22302e')
                for _ in range(20):movie.append_data(np.asarray(compare))
            print(name,fr,best,flush=True)
    (out/'drive.stage.mp4').replace(out/'drive.mp4')
    record['playback_fps']=fps
    save_json(out/'placement.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--case',required=True)
    p.add_argument('--scans',type=Path,help='Optional prepared LiDAR cache; otherwise read the raw site scans')
    p.add_argument('--raw',type=Path,default=ROOT/'tools/multiscene/data/raw')
    p.add_argument('--processed',type=Path,default=ROOT/'tools/multiscene/data/processed')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--surface-lighting',action='store_true',help='Resolve the dry-bed material and trace cavity shadows')
    review(p.parse_args())
