"""Fit five metric road locations in each new PandaSet reconstruction."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree

WORK=Path(__file__).parent
OUT=PROJECT/'outputs/道路病害_多场景迁入'
CATS=['stereo','rgbd','pothole','missing','raised']
TITLES={'034':'浅色分块接缝路面','109':'深色均匀路面','110':'灰色裂缝封补路面'}

def unit(x): return x/np.linalg.norm(x)
def project(pose,k,point):
    xyz=(point-pose[:3,3])@pose[:3,:3]
    return np.array([xyz[0]*k[0]/xyz[2]+k[2],xyz[1]*k[1]/xyz[2]+k[3],xyz[2]])

def fit(points,tree,xy,z_prior,right):
    p=points[tree.query_ball_point(xy,1.45)]
    if len(p)<180: return None
    p=p[np.abs(p[:,2]-z_prior)<.70]
    if len(p)<180: return None
    local=p[:,:2]-xy
    a=np.c_[local,np.ones(len(p))]; z=p[:,2]
    mask=np.ones(len(p),dtype=bool)
    for _ in range(12):
        if mask.sum()<150: return None
        coef=np.linalg.lstsq(a[mask],z[mask],rcond=None)[0]
        residual=z-a@coef
        med=np.median(residual[mask])
        sigma=max(.005,np.median(np.abs(residual[mask]-med))*1.4826)
        mask=np.abs(residual-med)<min(.055,2.8*sigma)
    normal=unit(np.r_[-coef[:2],1.])
    side=unit(right-(right@normal)*normal)
    coverage=np.histogram2d(local[mask,0],local[mask,1],bins=5,range=[[-1.1,1.1],[-1.1,1.1]])[0]
    rms=float(np.sqrt(np.mean(residual[mask]**2)))
    if rms>.030 or np.mean(coverage>0)<.72 or np.linalg.norm(coef[:2])>.38: return None
    return {'center_world':[*xy,float(coef[2])],
            'basis_columns':np.stack([side,np.cross(normal,side),normal],1).tolist(),
            'points':int(mask.sum()),'rms_m':rms,'neighborhood_coverage':float(np.mean(coverage>0)),
            'plane_slope':float(np.linalg.norm(coef[:2]))}

def build(site):
    data=WORK/'data/processed'/site
    poses=[np.loadtxt(data/'extrinsics'/f'{i:03d}_0.txt') for i in range(80)]
    centers=np.array([p[:3,3] for p in poses])
    k=np.loadtxt(data/'intrinsics/0.txt').ravel()
    cloud=[]
    for frame in range(0,74,3):
        v=PlyData.read(data/'lidar_forward/background'/f'{frame:03d}.ply',mmap='r')['vertex'].data
        cloud.append(np.stack([v[n] for n in ('x','y','z')],1))
    points=np.concatenate(cloud)
    points=points[np.min(np.linalg.norm(points[:,None,:2]-centers[::8,None,:2].transpose(1,0,2),axis=2),axis=1)<38]
    tree=cKDTree(points[:,:2])
    base_frame=0 if site=='110' else 30
    origin=poses[base_frame][:3,3]
    right=unit(np.r_[poses[base_frame][:2,0],0.])
    forward=unit(np.r_[poses[base_frame][:2,2],0.])
    slots=[]
    distances=[10.,13.5,14.5,18.,10.5] if site=='110' else [7.5,11.5,15.5,19.5,23.5]
    early_laterals=[[.65,.8,.4],[-1.4,-1.6,-1.2],[1.65,1.4,1.2],[-.4,-.6,-.2],[-3.,-3.2,-2.8]]
    for i,distance in enumerate(distances):
        options=[]
        lateral_options=early_laterals[i] if site=='110' else ([.65,-.60,.15,1.05] if i%2==0 else [-.60,.65,.15,-1.05])
        for side in lateral_options:
            guess=origin+forward*distance+right*side
            closest=np.argmin(np.linalg.norm(centers[:,:2]-guess[:2],axis=1))
            z_prior=centers[closest,2]-2.12
            rec=fit(points,tree,guess[:2],z_prior,right)
            if rec is None: continue
            rec.update(slot=f'{i+1:02d}',distance_m=distance,lateral_m=side)
            best=None
            for frame in range(0 if site=='110' else 12,72,3):
                uvz=project(poses[frame],k,np.array(rec['center_world']))
                x,y,z=uvz
                if not (4<z<16 and 100<x<1820 and 450<y<1040): continue
                dynamic=np.asarray(Image.open(data/'dynamic_masks/all'/f'{frame:03d}_0.png'))
                xx,yy=int(x),int(y)
                patch=dynamic[max(0,yy-12):yy+13,max(0,xx-12):xx+13]
                if np.mean(patch>0)>.02: continue
                photo=np.asarray(Image.open(data/'images'/f'{frame:03d}_0.jpg'))
                rgb=np.median(photo[yy-8:yy+9,xx-8:xx+9].reshape(-1,3),axis=0)
                if rgb.min()>180: continue
                score=abs(z-7.5)
                if best is None or score<best[0]: best=(score,frame,rgb)
            if best is None: continue
            rec.update(source_reference_frame=best[1],source_road_rgb=best[2].tolist())
            # Preserve the preferred lateral position unless its local plane is worse.
            options.append((len(options)*.006+rec['rms_m'],rec))
        if not options: raise ValueError(f'{site}: no road fit for slot {i+1}')
        slots.append(min(options,key=lambda x:x[0])[1])
    centers2=np.array([s['center_world'][:2] for s in slots])
    spacing=np.linalg.norm(centers2[:,None]-centers2[None,:],axis=2)+np.eye(len(centers2))*1e6
    assert spacing.min()>3.05
    canvas=Image.new('RGB',(1920,1200),'#101923');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
    draw.text((20,14),f'PandaSet {site} · {TITLES[site]} · LiDAR 道路定位',font=font,fill='white')
    display_frames=[0,5,10,20] if site=='110' else [22,34,46,58]
    for j,frame in enumerate(display_frames):
        pic=Image.open(data/'images'/f'{frame:03d}_0.jpg').resize((960,540))
        d=ImageDraw.Draw(pic)
        for i,s in enumerate(slots):
            x,y,z=project(poses[frame],k,np.array(s['center_world']))
            if z>1 and 0<x<1920 and 0<y<1080:
                x/=2;y/=2;d.ellipse((x-7,y-7,x+7,y+7),fill='#ffbe66',outline='black',width=2)
                d.text((x+8,y-12),str(i+1),font=font,fill='white',stroke_width=2,stroke_fill='black')
        d.text((20,20),f'源图 / frame {frame}',font=font,fill='white',stroke_width=2,stroke_fill='black')
        canvas.paste(pic,((j%2)*960,70+(j//2)*550))
    OUT.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT/f'道路定位_{site}.jpg',quality=94)
    return slots

def main():
    sample=json.loads((WORK/'selected_assets.json').read_text())
    slots={site:build(site) for site in TITLES}
    scenes=[]
    for site in TITLES:
        for i in range(10):
            scenes.append({'id':f'{site}_{i+1:02d}','site':site,
                           'assets':[sample['selected'][c][i] for c in CATS],'showcase':True})
    plan={'seed':sample['seed'],'category_order':CATS,'selected_assets':sample['selected'],
          'scene_titles':TITLES,'drive_start_by_site':{'034':20,'109':20,'110':0},
          'slots_by_site':slots,'scenes':scenes,'unique_assets':50,'placements':150}
    (WORK/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (OUT/'随机抽样与迁入计划.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({s:[round(x['rms_m'],5) for x in records] for s,records in slots.items()}),flush=True)

if __name__=='__main__': main()
