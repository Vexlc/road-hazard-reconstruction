"""Check every height-grid payload and cross-check representative GLBs with Blender BVH."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import bpy  # Registers the bundled mathutils module.
from mathutils.bvhtree import BVHTree
import numpy as np
import trimesh


ROOT=Path(__file__).resolve().parents[1]
ATLAS=ROOT/'tools/asset_atlas'


def decode(row):
    raw=(ATLAS/row['relief']).read_text()
    begin='window.roadRelief('
    key,data=json.loads('['+raw[len(begin):-3]+']')
    if key!=row['id'] or not raw.startswith(begin) or not raw.endswith(');\n'):raise ValueError(key)
    n=data['n']
    for field in ('top','under'):
        if field not in data:continue
        coded=np.frombuffer(base64.b64decode(data[field],validate=True),'<u2').reshape(n,n)
        data[field]=np.where(coded>0,data['low']+(coded.astype(float)-1)*data['step'],np.nan)
        if not np.isfinite(data[field]).any():raise ValueError('Empty grid '+key)
    return data


def compare(path,data,seed):
    with path.open('rb') as f:
        digest=hashlib.sha256()
        for chunk in iter(lambda:f.read(4<<20),b''):digest.update(chunk)
        if digest.hexdigest()!=data['glb_sha256']:raise ValueError('Model changed '+str(path))
    scene=trimesh.load(path,force='scene',process=False,skip_materials=True)
    fixed,moving=[],[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        mesh=scene.geometry[name].copy()
        mesh.apply_transform(transform)
        (moving if name.startswith(('Lid ','Hinge ')) else fixed).append(mesh)
    rng=np.random.default_rng(seed)
    errors=[]
    for field,meshes in [('top',fixed+moving),('under',fixed)]:
        if field not in data:continue
        all_mesh=trimesh.util.concatenate(meshes)
        tree=BVHTree.FromPolygons(all_mesh.vertices.tolist(),all_mesh.faces.tolist(),all_triangles=True)
        ys=np.linspace(*data['y'],data['n']);xs=np.linspace(*data['x'],data['n'])
        samples=np.argwhere(np.isfinite(data[field]))
        # Avoid exact shared triangle edges at the axis of radial meshes. A ray
        # on that seam can legitimately be assigned to either adjoining face.
        mid=data['n']//2
        samples=samples[(abs(samples[:,0]-mid)>1)&(abs(samples[:,1]-mid)>1)]
        for yy,xx in samples[rng.choice(len(samples),min(96,len(samples)),replace=False)]:
            point,_,_,_=tree.ray_cast((xs[xx],float(all_mesh.bounds[1,1]+.5),-ys[yy]),(0,-1,0))
            if point is None:raise ValueError(f'Independent BVH missed {path} at {xx},{yy}')
            errors.append(abs(point.y-data[field][yy,xx]))
    if max(errors)>.002:raise ValueError(f'{path}: inconsistent grid {max(errors)} m')
    return dict(independent_rays=len(errors),max_error_m=max(errors))


def check():
    rows=json.loads((ATLAS/'catalog.js').read_text()[len('window.roadAssets='):-2])
    reference={'A115','A116','A117','M115','A118','A119','A120','M118','P110','P111','P112','P113','P114','A107','A108','M107','A104','A105','M104','A93','A95','M91','M13','P13','P102','P90','stereo_pothole_fan_1_03',
               'rgbd_pothole_20250227_135438','model_raised_rectangular_removed_side_chip_seal'}
    summary=dict(records=len(rows),with_removable_cover=0,valid_top_samples=0,pitch_m=[],checked_meshes={})
    for i,row in enumerate(rows):
        d=decode(row)
        top=d['top'];n=d['n']
        if 'under' in d:
            summary['with_removable_cover']+=1
            ok=np.isfinite(d['under'])&np.isfinite(top)
            if (d['under'][ok]>top[ok]+d['step']).any():raise ValueError(row['id'])
        summary['valid_top_samples']+=int(np.isfinite(top).sum())
        summary['pitch_m'].append(max(abs(d['x'][1]-d['x'][0]),abs(d['y'][1]-d['y'][0]))/(n-1))
        if row['id'] in reference:summary['checked_meshes'][row['id']]=compare(ATLAS/row['glb'],d,i+3567)
    if reference!=set(summary['checked_meshes']):raise ValueError('Missing independent samples')
    pitches=summary.pop('pitch_m')
    summary['pitch_m_min_median_max']=np.percentile(pitches,[0,50,100]).tolist()
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path)
    a=p.parse_args();content=json.dumps(check(),ensure_ascii=False,indent=2)+'\n'
    if a.out:a.out.write_text(content,encoding='utf-8')
    else:print(content,end='')
