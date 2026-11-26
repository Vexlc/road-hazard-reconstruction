"""Reconstruct the pen-scaled Urayasu shaft with separate concrete courses, flange and lid."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from calibration import Camera,fit

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lane-damage'))
from photo_mesh import Mesh,Surface,grid,grid_faces,xyz,face_up,face_polygon,texture_uv,write_json,make_one,flatten_scene,remap


def ring_surface(mesh,name,v,uv,material,outward):
    faces=grid_faces(*v.shape[:2]);v=v.reshape(-1,3)
    a,b,c=v[faces].transpose(1,0,2)
    flip=np.einsum('ij,ij->i',np.cross(b-a,c-a),outward((a+b+c)/3))<0
    faces[flip]=faces[flip,::-1]
    mesh.add(name,v,faces,uv.reshape(-1,2),material)


def build(out):
    s=json.loads((HERE/'urayasu.json').read_text());cal=fit(s)
    img_file=HERE/'references/A563.jpg'
    assert hashlib.sha256(img_file.read_bytes()).hexdigest()==s['sha256']
    obs=np.asarray(Image.open(img_file).convert('RGB').resize((1800,1350),Image.Resampling.LANCZOS))
    # Remove the scale object from the colour map while retaining it in the original evidence.
    mask=np.zeros(obs.shape[:2],'u1')
    a,b=np.int32(s['pen'])*2
    cv2.line(mask,tuple(a),tuple(b),255,48)
    img=cv2.inpaint(obs,mask,10,cv2.INPAINT_TELEA)
    mesh=Mesh();p=cal['profile'];scale=cal['metres_per_unit'];cam=Camera(s,cal['parameters'])
    height=p['cap']['height_above_road_m'];par=np.asarray(cal['parameters'])
    def center(z):
        return (np.asarray(z)[...,None]-height)*par[11:13]
    def project(v):
        return cam.project((v-[0.,0.,height])/scale)
    def uv(v):
        return texture_uv(project(v),s['size'])
    def material(name,rgb,rough,metal=0.):
        return mesh.material(name,rgb,rough,metal,.65,lossless_color=True)
    def cylinder(z0,z1,r0,r1,angle,t):
        rr=r0+(r1-r0)*t;z=z0+(z1-z0)*t
        return xyz(center(z)+np.stack([rr*np.cos(angle),rr*np.sin(angle)],axis=-1),z)
    def section(name,z0,z1,r0,r1,rough,metal):
        # Unwrap the visible cylinder. Complete the hidden arc by folding opposite angles back
        # into the actual visible material interval, retaining the measured axial changes.
        t,theta=np.meshgrid(np.linspace(.04,.96,192),np.linspace(-np.pi,np.pi,1024),indexing='ij')
        axis=cam.eye[:2]*scale-center((z0+z1)/2)
        facing=np.arctan2(axis[1],axis[0]);rel=(theta-facing+np.pi)%(2*np.pi)-np.pi
        cutoff=np.deg2rad(64)
        folded=np.sign(rel)*np.where(abs(rel)<=cutoff,abs(rel),cutoff*(np.pi-abs(rel))/(np.pi-cutoff))
        atlas_xyz=cylinder(z0,z1,r0,r1,facing+folded,t)
        tex=remap(img,(project(atlas_xyz)+.5)*2-.5)
        mat=material(name+' measured front and completed back',tex,rough,metal)
        tt,aa=np.meshgrid(np.linspace(0.,1.,19),np.linspace(-np.pi,np.pi,257),indexing='ij')
        v=cylinder(z0,z1,r0,r1,aa,tt)
        coord=np.stack([(aa+np.pi)/(2*np.pi),1.-tt],axis=-1)
        ring_surface(mesh,name,v,coord,mat,lambda w:xyz(w[:,:2]-center(w[:,2]),0.))
        return mat

    base_r=p['base']['radius_m'];jr=p['joint']['radius_m'];jz=p['joint']['height_above_road_m']
    flange_r=p['flange']['radius_m'];fz=p['flange']['height_above_road_m'];cap_r=p['cap']['radius_m']
    section('Lower concrete shaft course',0.,jz,base_r,jr,.95,0.)
    section('Upper painted concrete course',jz,fz-.008,jr-.0015,flange_r-.005,.93,0.)
    section('Cast seat riser and ledge',fz-.008,fz,flange_r,flange_r,.82,.32)
    section('Lid rolled edge',fz,height-.006,cap_r*.972,cap_r,.74,.35)
    section('Lid aperture walls and perimeter',height-.006,height,cap_r,cap_r*.992,.56,.4)

    # The back part of the flange is occluded by the lid from the source camera:
    # use near-side flange pigment around this shallow ring.
    a=np.linspace(-np.pi,np.pi,257)
    theta=np.stack([a,a]);radii=np.stack([np.ones_like(a)*cap_r*.972,np.ones_like(a)*flange_r])
    ring=xyz(center(fz)+radii[...,None]*np.stack([np.cos(theta),np.sin(theta)],axis=-1),fz)
    tt,theta=np.meshgrid(np.linspace(.06,.94,88),np.linspace(-np.pi,np.pi,1024),indexing='ij')
    facing=np.arctan2(*(cam.eye[:2]*scale-center(fz))[::-1])
    rel=(theta-facing+np.pi)%(2*np.pi)-np.pi
    cutoff=np.deg2rad(58)
    ang=facing+np.sign(rel)*np.where(abs(rel)<=cutoff,abs(rel),cutoff*(np.pi-abs(rel))/(np.pi-cutoff))
    radius=(1-tt)*cap_r*.972+tt*flange_r
    samples=xyz(center(fz)+radius[...,None]*np.stack([np.cos(ang),np.sin(ang)],axis=-1),fz)
    rusty=material('Oxidised flange',remap(img,(project(samples)+.5)*2-.5),.84,.26)
    uv_ring=np.stack([(np.stack([a,a])+np.pi)/(2*np.pi),np.broadcast_to([[1.],[0.]],ring.shape[:2])],axis=-1)
    ring_surface(mesh,'Cast ring and cover seat',ring,uv_ring,rusty,lambda v:np.tile([0,0,1.],(len(v),1)))
    # The cap has a broad worn rim and a patterned iron surface; the photograph supplies
    # colour and material microrelief. Macro height is kept on the fitted cap plane.
    cap=xyz(np.c_[np.cos(a[:-1]),np.sin(a[:-1])]*cap_r*.992,height)
    mat=material('Urayasu cast lid from the field photograph',img,.70,.38)
    face_polygon(mesh,'Lid face',cap,[],mat,uv)

    # Two threaded anchors and two triangular cast ribs visible on the near mounting flange.
    def hit(px):
        q=cam.plane(np.asarray([px]),(fz-height)/scale)[0]*scale+[0.,0.,height]
        return q[:2]
    steel=material('Exposed anchor steel',img[425:474,1182:1193],.74,.35)
    for n,(x,y,h,rad) in enumerate([(234,216,.047,.008),(596,249,.059,.0075)]):
        pos=hit([x,y]);direction=np.c_[np.cos(a),np.sin(a)]
        rings=np.stack([xyz(pos+rad*direction,fz),xyz(pos+rad*.92*direction,fz+h)])
        tex=np.stack([np.c_[np.linspace(0,1,len(a)),np.zeros(len(a))],np.c_[np.linspace(0,1,len(a)),np.ones(len(a))]])
        ring_surface(mesh,f'Steel pipe {n+1:02} anchor',rings,tex,steel,lambda v:xyz(v[:,:2]-pos,0.))
        face_polygon(mesh,f'Steel pipe {n+1:02} tip',rings[1,:-1],[],steel,lambda v:v[:,:2]*10)
    for n,pos in enumerate([[271,264],[463,289]]):
        radial=hit(pos)-center(fz);radial/=np.linalg.norm(radial);across=np.array([-radial[1],radial[0]])*.004
        shape=np.stack([np.r_[center(fz)+radial*cap_r*.976,fz],
                        np.r_[center(fz)+radial*(flange_r-.013),fz+.002],
                        np.r_[center(height-.018)+radial*cap_r*.98,height-.018]])
        v=np.vstack([shape-np.r_[across,0.],shape+np.r_[across,0.]])
        f=np.array([[0,1,2],[3,5,4],[0,3,4],[0,4,1],[1,4,5],[1,5,2],[2,5,3],[2,3,0]])
        x,y,z=v[f].transpose(1,0,2);flip=np.einsum('ij,ij->i',np.cross(y-x,z-x),x-v.mean(0))<0;f[flip]=f[flip,::-1]
        mesh.add(f'Steel pipe {n+3:02} casting rib',v,f,uv(v),mat)

    # A narrow earth contact skirt uses only adjacent ground from this image.
    patch=img[805:911,190:293]
    ground=np.tile(np.vstack([np.hstack([patch,patch[:,::-1]]),np.hstack([patch[::-1],patch[::-1,::-1]])]),(4,4,1))
    road=material('Adjacent disturbed asphalt and silt',ground,.99)
    outline=np.c_[np.cos(a[:-1]),np.sin(a[:-1])]
    toe=center(0.)+base_r*outline;fringe=center(0.)+(base_r+.057)*outline
    face_polygon(mesh,'Road adjoining the fracture',xyz(fringe,0.),[xyz(toe,0.)],road,lambda v:.5+(v[:,:2]-center(0.))/(2*(base_r+.06)))
    span=2*np.max(abs(fringe),axis=0)+.02;xy=grid(span,max(span)/460);ext=np.ptp(xy,axis=(0,1))
    series=out/'series';dest=series/'assets/raised/A563';cache=series/'mesh_cache'
    meta=dict(id='A563',label=s['name'],seed=28563,category='raised',state='photo_reconstruction',family='seismic_uplift',
              source=s['source'],source_author=s['author'],source_license=s['license'],source_sha256=s['sha256'],
              surface_dimensions_m=ext.tolist(),profile_calibration=cal,
              unseen_side='Repeated angular samples from the photographed front. Cone, pen and cast shadow excluded.')
    mesh.save(dest,cache,meta,dict(width=ext[0],length=ext[1],height=np.zeros(xy.shape[:2])))
    flatten_scene(dest/'asset.glb');surface=Surface(dest/'asset.glb')
    start=xyz(xy,2.);d,parts,_=surface.cast(start,np.broadcast_to([0.,0.,-1.],start.shape))
    hit=np.isfinite(d)
    np.savez_compressed(dest/'surface.npz',width=ext[0],length=ext[1],height=np.where(hit,2.-d,0.).astype('f4'),
            alpha=hit,defect_mask=parts>1,installation_mask=parts>1,opening_mask=np.zeros_like(hit),
            water_mask=np.zeros_like(hit),photographic_rgb=True)
    row=make_one(dest,cache,point_limit=465000,kernel_factors=(.76,.15))
    for role,file in [('glb','asset.glb'),('ply','asset.ply'),('surface','surface.npz'),('metadata','metadata.json')]:
        row[role]='raised/A563/'+file
    write_json(series/'assets/asset_manifest.json',[row]);write_json(out/'run.json',dict(editions=['series']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);build(p.parse_args().out)
