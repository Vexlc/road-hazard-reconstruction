"""Load a complete indexed scan, or export it as NPZ / an ordinary metric LiDAR PLY."""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT=Path(__file__).resolve().parents[1]
PALETTE=np.array([[140,145,149],[161,164,165],[62,174,230],[45,142,210],[77,207,220],
                  [226,171,66],[58,178,196],[65,121,183],[168,158,135],
                  [246,121,66],[234,148,96],[250,165,79],[235,191,83]],'u1')


def load_scan(root,case,frame,mode='after'):
    if not (0<=frame<80) or mode not in ('before','after'):
        raise ValueError('Expected a source frame from 0 through 79 and before or after')
    root=Path(root);site=case.split('_')[0]
    with np.load(root/'base'/site/f'{frame:03d}.npz') as base:
        scan={k:base[k] for k in base.files}
    n=len(scan['xyz'])
    scan['raw_row']=np.arange(n,dtype='i4')
    scan['source_semantic']=scan['semantic'].copy()
    scan['intensity']=scan['intensity'].astype('f4')
    scan['valid']=np.ones(n,bool)
    for k in ['action','hazard_class','part','instance']:scan[k]=np.zeros(n,'u1')
    if mode=='after':
        with np.load(root/'cases'/(case+'.npz')) as delta:
            start,end=delta['frame_offsets'][frame:frame+2]
            sl=slice(start,end);idx=delta['row'][sl]
            action=delta['action'][sl];cl=delta['hazard_class'][sl]
            scan['xyz'][idx]=delta['xyz'][sl]
            scan['intensity'][idx]=np.nan
            for k in ('action','hazard_class','part','instance'):scan[k][idx]=delta[k][sl]
            scan['valid'][idx]=action==1
            scan['semantic'][idx]=np.where(action==1,100+cl,0)
    return scan


def crop_mask(scan,placement,margin):
    """Identical spatial selection can be applied to baseline and replay using baseline row IDs."""
    B=np.array(placement['basis_columns']); c=np.array(placement['center_world'])
    xyz=(scan['xyz']-c)@B
    return (abs(xyz[:,:2])<=margin).all(1)&(abs(xyz[:,2])<=2.5)


def export(path,scan):
    path=Path(path)
    if path.suffix.lower()=='.npz':
        np.savez_compressed(path,**scan);return
    if path.suffix.lower()!='.ply':raise ValueError('Choose a .npz or .ply path')
    good=scan['valid'];n=int(good.sum())
    fields=[(k,'<f8','double') for k in ('x','y','z','time')]
    fields+= [('intensity','<f4','float'),('raw_row','<i4','int')]
    fields+= [(k,'u1','uchar') for k in ('red','green','blue','sensor','semantic','source_semantic','action','hazard_class','instance','part')]
    data=np.empty(n,dtype=[(k,t) for k,t,_ in fields])
    for i,k in enumerate('xyz'):data[k]=scan['xyz'][good,i]
    for k in ('time','intensity','raw_row','sensor','semantic','source_semantic','action','hazard_class','instance','part'):data[k]=scan[k][good]
    color=np.repeat((68+np.sqrt(np.nan_to_num(data['intensity'])/255)*164).astype('u1')[:,None],3,1)
    changed=data['action']>0;color[changed]=PALETTE[data['part'][changed]]
    for i,k in enumerate(['red','green','blue']):data[k]=color[:,i]
    header=['ply','format binary_little_endian 1.0','comment Metric LiDAR returns, xyz in PandaSet world metres',
            'comment Display RGB encodes measured intensity or simulated surface part',f'element vertex {n}']
    header+=['property '+p+' '+k for k,_,p in fields]
    with path.open('wb') as f:
        f.write(('\n'.join(header+['end_header',''])).encode('ascii'))
        data.tofile(f)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--case',required=True)
    p.add_argument('--frame',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',choices=['before','after'],default='after')
    p.add_argument('--asset',help='Optional spatial crop around this instance, identical source rows in both modes')
    p.add_argument('--margin-m',type=float,default=2)
    a=p.parse_args();scan=load_scan(a.root,a.case,a.frame,a.mode)
    if a.asset:
        scene=json.loads((a.root/'cases'/(a.case+'.json')).read_text())
        pl=next(x for x in scene['assets'] if x['asset_id']==a.asset)
        before=scan if a.mode=='before' else load_scan(a.root,a.case,a.frame,'before')
        mask=crop_mask(before,pl,a.margin_m)
        scan={k:v[mask] for k,v in scan.items()}
    export(a.output,scan)


if __name__=='__main__':main()
