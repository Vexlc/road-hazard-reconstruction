"""Locate held-out height-reference outliers in the source photographs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import yaml

from run import atomic_json


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'))
    plan=json.loads(Path(job['plan']).read_text(encoding='utf-8'))
    dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    report=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for site,config in job['configs'].items():
        data=Path(yaml.safe_load(Path(config).read_text(encoding='utf-8'))['source_path'])
        folder=Path(job['output'])/f'{site}_01'
        meta=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
        state=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
        n=meta['host_road_guard']['points'];bg=state['background']
        xyz=bg['xyz'][-n:-n//2].detach().numpy().copy()
        q=bg['rotation'][-n:-n//2].detach().numpy()
        xyz+=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:,2]*.002
        samples=xyz[np.linspace(0,len(xyz)-1,min(10000,len(xyz)),dtype=int)]
        low,high=xyz[:,:2].min(0)-.2,xyz[:,:2].max(0)+.2
        centers=np.array([r['center_world'] for r in plan['slots_by_site'][site]])
        normals=np.array([np.array(r['basis_columns'])[:,2] for r in plan['slots_by_site'][site]])
        points=[];frame_ids=[]
        for frame in range(1,80,3):
            a=PlyData.read(data/'lidar_forward/background'/f'{frame:03d}.ply',mmap='r')['vertex'].data
            p=np.stack([a[k] for k in ('x','y','z')],axis=1)
            p=p[np.all((p[:,:2]>low)&(p[:,:2]<high),axis=1)]
            nearest=np.argmin(np.linalg.norm(p[:,None,:2]-centers[None,:,:2],axis=2),axis=1)
            z=centers[nearest,2]-np.sum((p[:,:2]-centers[nearest,:2])*normals[nearest,:2],axis=1)/normals[nearest,2]
            p=p[abs(p[:,2]-z)<.30]
            points.append(p);frame_ids.append(np.full(len(p),frame,dtype='i2'))
        points=np.concatenate(points);frame_ids=np.concatenate(frame_ids)
        distance,near=cKDTree(points[:,:2]).query(samples[:,:2],k=3,workers=2)
        close=(distance[:,0]<.08)&(distance[:,2]<.16)
        ids=np.flatnonzero(close);reference=np.median(points[near[close],2],axis=1)
        errors=abs(samples[close,2]-reference)
        order=np.argsort(errors)[::-1][:12]
        canvas=Image.new('RGB',(1600,960),(20,25,30));draw=ImageDraw.Draw(canvas)
        records=[]
        intr=np.loadtxt(data/'intrinsics/0.txt').ravel()
        for j,index in enumerate(order):
            original_id=ids[index]
            neighbors=near[original_id]
            median_idx=neighbors[np.argsort(points[neighbors,2])[1]]
            frame=int(frame_ids[median_idx]);target=points[median_idx]
            pose=np.loadtxt(data/'extrinsics'/f'{frame:03d}_0.txt')
            q=(target-pose[:3,3])@pose[:3,:3]
            uv=q[:2]/q[2]*intr[:2]+intr[2:4]
            im=Image.open(data/'images'/f'{frame:03d}_0.jpg').convert('RGB')
            x,y=np.rint(uv).astype(int)
            crop=im.crop((x-160,y-120,x+160,y+120)).resize((400,300))
            d=ImageDraw.Draw(crop);d.ellipse((194,144,206,156),outline='red',width=3)
            col,row=j%4,j//4;canvas.paste(crop,(col*400,row*320+20))
            draw.text((col*400+5,row*320),f'{site} f{frame:02d} Δ{errors[index]*100:.1f}cm',font=font,fill='white')
            records.append({'sample_world':samples[original_id].tolist(),'reference_world':target.tolist(),
                            'frame':frame,'reference_pixel':uv.tolist(),'absolute_height_error_m':float(errors[index]),
                            'neighbor_heights_m':points[neighbors,2].tolist(),'xy_distances_m':distance[original_id].tolist()})
        canvas.save(dest/f'{site}_最大高度差_源图定位.jpg',quality=94)
        report.append({'site':site,'outliers':records,'matched_samples':len(errors),
                       'over_5cm':int((errors>.05).sum()),'over_10cm':int((errors>.1).sum())})
        print('HEIGHT DIAGNOSED',site,len(errors),int((errors>.05).sum()),flush=True)
        del state,bg
    atomic_json(dest/'高度差定位.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
