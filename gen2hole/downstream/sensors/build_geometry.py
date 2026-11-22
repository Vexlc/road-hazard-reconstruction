"""Export centimetre-spaced multilevel surface geometry for the reconstructed assets."""
import argparse
from pathlib import Path

import numpy as np

from prepare_scans import write_npz
from road_geometry import Surface, load_inventory, sha256, save_json


def heights(mesh, x, y, start_z):
    coords=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,start_z))).astype('f4')
    direction=np.zeros_like(coords);direction[:,2]=-1
    distance,parts,normals=mesh.cast(coords,direction)
    hit=parts>0
    z=np.full(x.size,np.nan,'f4');z[hit]=start_z-distance[hit]
    normals[~hit]=0
    # Face winding can be two-sided. Preserve the orientation of the top envelope.
    normals[normals[:,2]<0]*=-1
    return z.reshape(x.shape),parts.reshape(x.shape),normals.reshape((*x.shape,3))


def build(parent, out):
    folder=out/'geometry';folder.mkdir(parents=True,exist_ok=True)
    assets,_,_=load_inventory(parent);records={}
    for asset,(path,meta) in sorted(assets.items()):
        f=path/'asset.glb';mesh=Surface(f)
        pitch=.01
        xmin,ymin=np.ceil((mesh.low[:2]+.00001)/pitch).astype(int)
        xmax,ymax=np.floor((mesh.high[:2]-.00001)/pitch).astype(int)
        xs=np.arange(xmin,xmax+1,dtype='f4')*pitch
        ys=np.arange(ymax,ymin-1,-1,dtype='f4')*pitch
        x,y=np.meshgrid(xs,ys)
        start=float(np.ceil(mesh.high[2]+.2))
        top,part,normal=heights(mesh,x,y,start)
        underlying_z,underlying_part=top.copy(),part.copy()
        lid=(part>=9)&(part<=12)
        overlay=(part==4)|lid|np.isin(part,[13,14,15])
        if overlay.any():
            substrate=Surface(f,exclude=(4,9,10,11,12,13,14,15))
            lower,p,_=heights(substrate,x[overlay],y[overlay],start)
            underlying_z[overlay],underlying_part[overlay]=lower,p
            del substrate
        # The water patches in these GLBs have no submerged bed mesh.
        rigid_z=np.where(part==4,underlying_z,top).astype('f4')
        assert not np.any((underlying_z>top+.001)&np.isfinite(top))
        dest=folder/(asset+'.npz')
        write_npz(dest,x_m=xs,y_m=ys,top_z_m=top,top_part=part,top_normal=normal.astype('f2'),
                  substrate_z_m=underlying_z,substrate_part=underlying_part,rigid_z_m=rigid_z)
        records[asset]=dict(file=str(dest.relative_to(out)),source_mesh=str(f.relative_to(parent)),
            mesh_sha256=sha256(f),grid_sha256=sha256(dest),category=meta['category'],label=meta['label'],
            shape=list(x.shape),pitch_m=pitch,mins_m=mesh.low.tolist(),maxs_m=mesh.high.tolist(),
            lowest_top_m=float(np.nanmin(top)),highest_top_m=float(np.nanmax(top)),water_cells=int((part==4).sum()),
            unknown_rigid_cells_under_water=int(((part==4)&~np.isfinite(rigid_z)).sum()),lid_cells=int(lid.sum()),
            cells_above_open_shaft=int((lid&(underlying_part==7)).sum()))
        if len(records)%20==0: print('height grids',len(records),flush=True)
        del mesh
    save_json(folder/'inventory.json',records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build(a.parent,a.out)
