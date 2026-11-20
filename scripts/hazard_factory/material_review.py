"""Native near-view comparison plus a 3D orbit exposing the same lid's underside."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization


def tensor(x):return torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device='cuda')
def unit(x):return x/np.linalg.norm(x)


@torch.no_grad()
def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'));root=Path(job['output']).parent
    new=Path(job['output'])/args.case;old=args.baseline/args.case
    metadata=json.loads((new/'scene.json').read_text(encoding='utf-8'))
    d=next(x for x in metadata['placements'] if x['category']=='raised')
    library=Path(job['libraries'][-1]);model=next(x for x in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8')) if x['id']==d['asset_id'])
    ply=PlyData.read(library/model['ply'],mmap='r')['vertex'].data
    xyz=np.stack([ply[k] for k in ('x','y','z')],axis=1)
    with np.load(Path(job['mesh_cache'])/(model['id']+'.npz')) as mesh:
        scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(mesh['xyz'].astype('f4')),o3d.core.Tensor(mesh['faces'].astype('u4')))
        primitive=scene.compute_closest_points(o3d.core.Tensor(xyz),nthreads=4)['primitive_ids'].numpy()
        labels=mesh['part_names'][mesh['face_part'][primitive]]
        selected=np.char.startswith(labels,'Lid')
    assert selected.sum()>1000
    states=[]
    for folder in (old,new):
        record=json.loads((folder/'scene.json').read_text(encoding='utf-8'));placement=next(x for x in record['placements'] if x['asset_id']==model['id'])
        checkpoint=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)['background']
        first=placement['merged_asset_offset'];last=first+placement['asset_points']
        subset={key:checkpoint[key][first:last].detach().numpy()[selected] for key in ('xyz','rotation','scaling','opacity','feature_dc','feature_rest')}
        states.append(dict(means=tensor(subset['xyz']),quats=tensor(subset['rotation']),scales=tensor(np.exp(subset['scaling'])),
                     opacities=tensor(1/(1+np.exp(-subset['opacity'][:,0]))),colors=tensor(np.concatenate([subset['feature_dc'],subset['feature_rest']],axis=1))))
    assert torch.max(abs(states[0]['means']-states[1]['means']))<2e-5
    basis=np.array(d['basis_columns']);points=states[1]['means'].cpu().numpy();center=(points.min(0)+points.max(0))*.5
    support=points[::max(1,len(points)//4000)];focal=650.;views=[];parameters=[];distance=.01
    for i in range(96):
        angle=i/96*2*np.pi-np.pi/3;elevation=np.deg2rad(35*np.cos(i/96*2*np.pi))
        away=basis@np.array([np.cos(angle)*np.cos(elevation),np.sin(angle)*np.cos(elevation),np.sin(elevation)])
        forward=-away;right=unit(np.cross(forward,basis[:,2]));down=np.cross(forward,right)
        rotation=np.stack([right,down,forward],axis=1);q=(support-center)@rotation
        distance=max(distance,float(np.max(np.maximum(abs(q[:,0])*focal/(960*.43)-q[:,2],abs(q[:,1])*focal/(540*.42)-q[:,2]))))
        parameters.append((rotation,away))
    for rotation,away in parameters:
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=center+away*distance;views.append(np.linalg.inv(pose))
    views=tensor(np.array(views));k=tensor([[focal,0,480],[0,focal,270],[0,0,1]])[None]
    frames=[]
    for state in states:
        output=[]
        for i in range(0,96,4):
            rgb,_,_=rasterization(**state,viewmats=views[i:i+4],Ks=k.expand(4,-1,-1),width=960,height=540,
                     sh_degree=1,packed=False,rasterize_mode='antialiased',backgrounds=tensor([[.12,.15,.19]]).expand(4,-1))
            output.append(np.uint8(rgb.clamp(0,1).cpu().numpy()*255+.5))
        frames.append(np.concatenate(output))
    clips=[]
    for folder in (old,new):
        reader=imageio.get_reader(str(folder/'raised_local_before_after.mp4'))
        clips.append([f[:,960:] for f in reader]);reader.close()
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',30)
    small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    output=root/'材质与接触遮蔽_三维检查对照.mp4'
    with imageio.get_writer(str(output),fps=24,codec='libx264',quality=8,macro_block_size=2,
                           output_params=['-threads','3','-movflags','+faststart']) as video:
        for phase in range(2):
            for i in range(96):
                image=Image.new('RGB',(1920,640),(18,23,29));draw=ImageDraw.Draw(image)
                draw.text((20,12),'旧版：按位置混合路面色',font=font,fill='white')
                draw.text((980,12),'新版：金属部件保留原材质',font=font,fill='#8de0cc')
                for half in (0,1):image.paste(Image.fromarray(clips[half][i*24//96] if phase==0 else frames[half][i]),(half*960,64))
                text=('原生场景近景：同一相机、同一尺度；新版加入几何接触遮蔽。' if phase==0 else
                      '独立井盖部件检查：相机绕到背面；两版几何与尺寸一致，仅迁入外观处理不同。')
                draw.text((20,609),text,font=small,fill='#c3cfdd')
                video.append_data(np.asarray(image))
                if phase==1 and i==48:image.save(root/'井盖底面材质_修复对照.jpg',quality=94)
    print('MATERIAL REVIEW EXPORTED',int(selected.sum()),'lid Gaussians',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--case',default='110_03');main(p.parse_args())
