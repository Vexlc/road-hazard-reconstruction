"""Check actual serialized underside colors and geometric contact-occlusion witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch

from run import atomic_json

C0=.28209479177387814


def original_mix(xyz,model):
    def smooth(x):
        x=np.clip(x,0,1);return x*x*(3-2*x)
    r=np.linalg.norm(xyz[:,:2],axis=1);z=xyz[:,2]
    return smooth((r-max(model['width_m'],model['length_m'])*.69)/.21)*smooth((z+.021)/.016)*smooth((.022-z)/.012)


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'));root=Path(job['output']).parent
    library=Path(job['libraries'][-1]);models={m['id']:m for m in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8'))}
    checks=[]
    for path in sorted(Path(job['output']).glob('*/scene.json')):
        meta=json.loads(path.read_text(encoding='utf-8'))
        after=torch.load(path.parent/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
        old_folder=args.baseline/meta['id']
        before=torch.load(old_folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False) if old_folder.exists() else None
        old_meta=json.loads((old_folder/'scene.json').read_text(encoding='utf-8')) if before is not None else None
        for d in meta['placements']:
            if d['category']!='raised':continue
            model=models[d['asset_id']];a=PlyData.read(library/model['ply'],mmap='r')['vertex'].data
            xyz=np.stack([a[k] for k in ('x','y','z')],axis=1)
            q=np.stack([a[f'rot_{i}'] for i in range(4)],axis=1)
            normal=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:,2]
            select=(normal[:,2]<-.5)&(xyz[:,2]>-.02)&(xyz[:,2]<.022)&(original_mix(xyz,model)>.025)
            if not select.any():continue
            rgb=np.clip(np.stack([a[f'f_dc_{i}'] for i in range(3)],axis=1)[select]*C0+.5,0,1)
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            def error(state,placement):
                target=np.clip(linear*placement['exposure_gain'],0,1)
                target=np.where(target<=.0031308,target*12.92,1.055*target**(1/2.4)-.055)
                start=placement['merged_asset_offset'];n=placement['asset_points']
                actual=state['background']['feature_dc'][start:start+n,0].detach().numpy()[select]*C0+.5
                return float(np.max(abs(actual-target))),int(np.sum(np.max(abs(actual-target),axis=1)>1e-4))
            new_error,new_bad=error(after,d)
            assert new_error<1e-5 and new_bad==0
            entry={'case':meta['id'],'asset_id':model['id'],'downward_near_road_points':int(select.sum()),
                   'new_max_rgb_error':new_error,'new_recolored_metal_points':new_bad}
            if before is not None:
                old_d=next(p for p in old_meta['placements'] if p['asset_id']==d['asset_id'])
                old_error,old_bad=error(before,old_d)
                entry.update(old_max_rgb_error=old_error,old_recolored_metal_points=old_bad)
            checks.append(entry)
        del after,before
    contacts=[]
    for m in models.values():
        if m['category']!='raised' or m['cover_pose'][0]!=0:continue
        cache=Path(job['work'])/'material_contact'/(m['id']+'.npz')
        with np.load(cache) as fields:road=fields['road'];ao=fields['ambient_visibility']
        a=PlyData.read(library/m['ply'],mmap='r')['vertex'].data
        xyz=np.stack([a[k] for k in ('x','y','z')],axis=1)
        _,dx,dy,yaw=m['cover_pose'];pivot=np.array([0,.525*m['length_m'],.015]);translation=np.array([dx,dy,.021 if dx else 0])
        rotation=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
        local=(xyz-pivot-translation)@rotation+pivot
        radius=(abs(local[:,0]/(m['width_m']*.5*1.036))**m['exponent']+abs(local[:,1]/(m['length_m']*.5*1.036))**m['exponent'])**(1/m['exponent'])
        under=road&(radius<.7)&(local[:,2]<-.017)&(original_mix(xyz,m)>.025)
        center=(-pivot)@rotation.T+pivot+translation
        far=road&(np.linalg.norm(xyz[:,:2],axis=1)>.9)&(np.linalg.norm(xyz[:,:2]-center[:2],axis=1)>1.0)
        assert under.sum()>100 and far.sum()>100
        assert ao[under].mean()<.2 and ao[far].mean()>.95
        contacts.append({'asset_id':m['id'],'state':m['state'],'under_lid_road_points':int(under.sum()),
                         'under_lid_mean_ambient_visibility':float(ao[under].mean()),
                         'far_road_points':int(far.sum()),'far_road_mean_ambient_visibility':float(ao[far].mean())})
    assert checks and contacts
    report={'passed':True,'serialized_metal_checks':checks,'geometric_contact_checks':contacts,
            'method':'Downward-facing metal remains exposure-corrected canonical color; independent lid-pose containment checks occlusion beneath horizontal lids against distant road.'}
    atomic_json(root/'材质与接触遮蔽验证.json',report)
    print('MATERIAL VERIFIED',len(checks),'checkpoint material samples;',len(contacts),'intrinsic lid contact tests',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);main(p.parse_args())
