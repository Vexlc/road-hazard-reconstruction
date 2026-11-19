"""Inspect fixed road positions against source lidar and front-camera images."""
import argparse,json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree

def unit(x):
    return x/np.linalg.norm(x)

def project(pose,K,center):
    xyz=(center-pose[:3,3])@pose[:3,:3]
    return [float(xyz[0]*K[0]/xyz[2]+K[2]),float(xyz[1]*K[1]/xyz[2]+K[3]),float(xyz[2])]

def anchor_fit(tree, lidar, guess, z_prior,right):
    positions = lidar[tree.query_ball_point(guess,1.35)]
    x=positions[:,:2]-guess
    z=positions[:,2]
    mask=abs(z-z_prior)<.40
    if mask.sum()<80:
        return None
    a=np.c_[x,np.ones(len(x))]
    for _ in range(8):
        coef=np.linalg.lstsq(a[mask],z[mask],rcond=None)[0]
        error=z-a@coef
        sigma=max(.006,np.median(abs(error[mask]))*1.4826)
        mask=abs(error)<min(.06,sigma*2.9)
    norm=unit(np.r_[-coef[:2],1.])
    side=unit(right-(right@norm)*norm)
    sample=np.histogram2d(x[mask,0],x[mask,1],bins=5,range=[[-1.1,1.1],[-1.1,1.1]])[0]
    return dict(center_world=[*guess,float(coef[2])],basis_columns=np.stack([side,np.cross(norm,side),norm],1).tolist(),
                points=int(mask.sum()),rms_m=float(np.sqrt(np.mean(error[mask]**2))),neighborhood_coverage=float(np.mean(sample>0)))

def inspect(data, prior, out):
    frames=[24,34,40,44,48,52]
    poses={i:np.loadtxt(data/'extrinsics'/f'{i:03d}_0.txt') for i in frames}
    k=np.loadtxt(data/'intrinsics/0.txt').ravel()
    original=json.loads(prior.read_text())
    ref=original['placements'][1]
    c=np.array(poses[40][:3,3])
    b=np.array(ref['basis_columns'])
    clouds=[]
    for t in range(16,66,3):
        v=PlyData.read(data/'lidar_forward/background'/f'{t:03d}.ply')['vertex']
        xyz=np.stack([v[n] for n in ['x','y','z']],axis=1)
        clouds.append(xyz)
    lidar=np.concatenate(clouds)
    select=np.linalg.norm(lidar[:,:2]-c[:2],axis=1)<42
    lidar=lidar[select]
    tree=cKDTree(lidar[:,:2])
    tested={}
    for d in [5.5,8.5,11.5,14.5,17.5,20.5,23.5,26.5,29.5,32.5]:
        for j,s in enumerate([-1.1,.05,1.15]):
            location=c+(b[:,1]*d+b[:,0]*s)
            coeff=np.array(original['road_plane_z_ax_by_c'])
            z=location[:2]@coeff[:2]+coeff[2]
            result=anchor_fit(tree,lidar,location[:2],z,b[:,0])
            if result is not None:
                result.update(distance_m=d,lateral_m=s)
                tested[f'{int(d):02d}{j}']=result
    out.mkdir(parents=True,exist_ok=True)
    (out/f'{data.name}_probe.json').write_text(json.dumps(tested,indent=2))
    canvas=Image.new('RGB',(1500,1410),'#18222b')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',14)
    for i,f in enumerate(frames):
        with Image.open(data/'images'/f'{f:03d}_0.jpg') as src:
            pic=src.resize((750,422))
            draw=ImageDraw.Draw(pic)
            for key,result in tested.items():
                x,y,d=project(poses[f],k,np.asarray(result['center_world']))
                if d<1 or y<0 or y>=src.height or x<0 or x>=src.width:
                    continue
                x=x*750/src.width;y=y*422/src.height
                color=['#8ffaaf','#fbc275','#9edaff'][int(key[-1])]
                draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black')
                draw.text((x+4,y+2),key,font=font,fill=color,stroke_width=2,stroke_fill='#101422')
            draw.text((24,28),f'{data.name} / {f:02d}',font=font,fill='white')
            canvas.paste(pic,(750*(i%2),465*(i//2)))
    canvas.save(out/f'{data.name}_probe.jpg',quality=92)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('project',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    for sid,tag in [('055','B'),('056','A')]:
        inspect(a.project/'work/data/processed'/sid,a.project/'outputs'/f'inserted_{tag}_{sid}'/'placement.json',a.out)
