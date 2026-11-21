"""Read the published GLBs with trimesh and validate first hits in Blender's separate BVH."""
import argparse
import json
from pathlib import Path

import bpy
import numpy as np
import trimesh
from mathutils.bvhtree import BVHTree

PART_NAMES={'Road around the aperture':1,'Road adjoining the fracture':1,'Fracture lip and exposed bed':2,
            'Fracture sidewall in surface coordinates':3,'Water surface':4,'Cast ring and cover seat':5,
            'Inspection shaft':6,'Shaft bottom':7,'Chipped concrete collar':8,'Lid face':9,
            'Lid underside with casting ribs':10,'Lid rolled edge':11}


def bvh(file):
    scene=trimesh.load(file,force='scene',process=False)
    nodes=[];tri=[];ids=[];cursor=0
    for n in scene.graph.nodes_geometry:
        transform,name=scene.graph.get(n);part=12 if n.startswith('Hinge ') else PART_NAMES[n]
        geometry=scene.geometry[name]
        v=trimesh.transform_points(geometry.vertices,transform)
        nodes.append(np.stack([v[:,0],-v[:,2],v[:,1]],axis=1));tri.append(geometry.faces+cursor)
        ids.append(np.full(len(geometry.faces),part,'u1'));cursor+=len(v)
    # Blender constructs its own tree and uses its own ray/triangle implementation.
    return BVHTree.FromPolygons(np.concatenate(nodes).tolist(),np.concatenate(tri).tolist(),all_triangles=True),np.concatenate(ids)


def main(parent,out):
    samples=json.loads((out/'checks/independent_samples.json').read_text())
    meta=json.loads((out/'geometry/inventory.json').read_text())
    rng=np.random.default_rng(20261121);ranges=[];griderr=[];part_dif=[];per={}
    for name in sorted(meta):
        shots=samples.get(name,[])
        tree,ids=bvh(parent/meta[name]['source_mesh'])
        local=[]
        for ray in shots:
            xyz,norm,face,dist=tree.ray_cast(ray['o'],ray['v'])
            if xyz is None:raise ValueError((name,ray,'no independent intersection'))
            diff=abs(dist-ray['t']);ranges.append(diff);local.append(diff)
            if ids[face]!=ray['part']:part_dif.append([name,ray['case'],ray['frame'],ray['part'],int(ids[face]),diff])
        with np.load(out/meta[name]['file']) as g:
            hp,parts=g['top_z_m'],g['top_part'];xx,yy=g['x_m'],g['y_m']
            cells=[]
            for part in np.unique(parts):
                if part==0:continue
                indices=np.argwhere(parts==part)
                cells.extend(indices[rng.integers(0,len(indices),min(7,len(indices)))])
            for y,x in cells:
                _,_,_,dist=tree.ray_cast((float(xx[x]),float(yy[y]),3.),(0.,0.,-1.))
                griderr.append(abs((3.-dist)-float(hp[y,x])))
        per[name]=dict(samples=len(shots),max_range_error_m=max(local,default=None))
        if len(per)%25==0:print('second geometry engine',len(per),flush=True)
        del tree
    result=dict(reader='trimesh GLB scene graph',intersector='Blender mathutils.bvhtree',assets=len(meta),
        real_beam_samples=len(ranges),vertical_grid_samples=len(griderr),range_p99_m=float(np.percentile(ranges,99)) if ranges else None,
        range_max_m=max(ranges,default=None),grid_height_p99_m=float(np.percentile(griderr,99)),grid_height_max_m=max(griderr),
        nearest_part_disagreements=part_dif,per_asset=per)
    (out/'checks/independent_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    assert max(ranges,default=0)<.002 and max(griderr)<.002 and len(part_dif)<=.005*len(ranges),result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.parent,a.out)
