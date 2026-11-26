"""Create registered fields, textured GLB and SH1 PLY for photographed lane hazards."""
import argparse
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from photo_mesh import Mesh,grid,xyz,Surface,write_json,flatten_scene,make_one,LIB
import manholes,photographed_pits,slabs,washout

HERE=Path(__file__).resolve().parent


def build(root,keys,mesh_only=False):
    sources=json.loads((HERE/'photos.json').read_text());series=root/'series';rows=[]
    for key in keys or sources:
        s=sources[key];src=HERE/'references'/(key+'.jpg')
        if hashlib.sha256(src.read_bytes()).hexdigest()!=s['photo_sha256']:raise ValueError(key)
        photo=Image.open(src).convert('RGB')
        # Retain the photo aspect and pixel-centre convention at the recorded resolution.
        photo.thumbnail((2*s['size'][0],2*s['size'][1]),Image.Resampling.LANCZOS)
        rgb=np.asarray(photo);mesh=Mesh()
        if key=='M552':span,details=manholes.pit(mesh,rgb,s)
        elif key=='A553':
            road=np.asarray(Image.open(LIB/'examples/perforated-covers/split-iron/pavement.jpg').convert('RGB'))
            span,details=manholes.pieces(mesh,rgb,s,road)
        elif key=='P556':span,details=washout.make(mesh,rgb,s)
        elif key=='P555':span,details=slabs.make(mesh,rgb,s)
        else:span,details=photographed_pits.make(mesh,rgb,s)
        category={'M':'missing','A':'raised','P':'pothole'}[key[0]]
        dest=series/'assets'/category/key
        (series/'references').mkdir(exist_ok=True,parents=True);shutil.copyfile(src,series/'references'/(key+'.jpg'))
        xy=grid(span,max(span)/512);ext=np.ptp(xy,axis=(0,1))
        meta=dict(id=key,category=category,label=s['label'],family='lane_damage_photos',state='photo_reconstruction',seed=23000+int(key[1:]),
                  source=s['source'],source_author=s['author'],source_license=s['licence'],source_image=f'references/{key}.jpg',
                  source_sha256=s['photo_sha256'],surface_dimensions_m=ext.tolist(),scale_basis='Monocular field image. Dimensions and unseen surfaces are inferred.',**details)
        tmp=dict(height=np.zeros(xy.shape[:2]),width=ext[0],length=ext[1],alpha=np.ones(xy.shape[:2]))
        mesh.save(dest,series/'mesh_cache',meta,tmp);flatten_scene(dest/'asset.glb')
        surf=Surface(dest/'asset.glb');start=xyz(xy,2.0);rays=np.broadcast_to([0.,0.,-1.],start.shape)
        dist,part,_=surf.cast(start,rays)
        height=np.where(np.isfinite(dist),2.-dist,0)
        field=dict(height=height,alpha=(part>0).astype(float),width=ext[0],length=ext[1],defect_mask=(part>1),installation_mask=(part>1),
                   opening_mask=((part==7)|(part==6)) if key=='M552' else (part>1),water_mask=part==4,photographic_rgb=True)
        if key=='P551':
            under=Surface(dest/'asset.glb',exclude={4});ds,_,_=under.cast(start,rays)
            field['substrate_height']=np.where(np.isfinite(ds),2.-ds,0)
        np.savez_compressed(dest/'surface.npz',**{k:np.asarray(v,dtype='f4') for k,v in field.items()})
        row=json.loads((dest/'metadata.json').read_text())
        row['depth_p95_m']=float(np.percentile(np.maximum(-height[part>1],0),95))
        write_json(dest/'metadata.json',row)
        if not mesh_only:
            row=make_one(dest,series/'mesh_cache',point_limit=610000,kernel_factors=(.75,.16))
        for role,name in [('glb','asset.glb'),('metadata','metadata.json'),('surface','surface.npz')]+([] if mesh_only else [('ply','asset.ply')]):
            row[role]=f'{category}/{key}/{name}'
        rows.append(row);print(key,flush=True)
    write_json(series/'assets/asset_manifest.json',rows);write_json(root/'run.json',dict(editions=['series']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('keys',nargs='*');p.add_argument('--mesh-only',action='store_true')
    build(**vars(p.parse_args()))
