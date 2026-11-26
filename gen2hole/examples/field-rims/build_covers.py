"""Rebuild a misaligned painted lid and a broken roadside telecom cover from two field photographs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'lane-damage'))
from photo_mesh import Mesh, Surface, face_polygon, face_up, flatten_scene, grid, make_one, rect, texture_uv, write_json, xyz


def circle(radius):
    a = np.linspace(0, 2*np.pi, 180, endpoint=False)
    return radius*np.column_stack([np.cos(a), np.sin(a)])


def wall(mesh, name, top, base, material, repeat=12):
    n=len(top)
    i=np.arange(n)
    # Duplicate the two seam vertices to preserve continuous UVs at the wrap.
    top=np.vstack([top,top[0]]);base=np.vstack([base,base[0]])
    v=np.vstack([top,base]);k=n+1
    f=np.vstack([np.c_[i,i+1,i+k+1],np.c_[i,i+k+1,i+k]])
    u=np.tile(np.linspace(0,repeat,k),2)
    mesh.add(name,v,f,np.c_[u,np.repeat([1.,0.],k)],material)


def painted(mesh, img, s):
    center=np.asarray(s['center'])
    scale=s['outer_diameter_m']*.5/np.asarray(s['ellipse_px'])*[1.,-1.]
    local=lambda p: (np.asarray(p)-center)*scale
    project=lambda p: texture_uv(p[:,:2]/scale+center,s['size'])
    pavement=mesh.material('Painted Bournemouth asphalt',img,.93,0,.8,lossless_color=True)
    iron=mesh.material('Cast iron and worn paint',img,.63,.21,1.1,lossless_color=True)
    shade=mesh.material('Dark recessed seat',img[170:181,486:492],.90,.08)
    outer=circle(s['outer_diameter_m']/2);inner=circle(s['rim_inner_radius_m'])
    lid=circle(s['lid_radius_m']);ledge_z=-.014
    # The tile widens only where the photo supplies neighbouring road paint.
    border=local(s['visible_road'])
    # Use concentric strips between the casting and the photographed polygon boundary.
    # Regular angular sampling keeps thin yellow paint accurately bounded without
    # recursively subdividing long, narrow triangles on the road outside the ring.
    angles=np.unique(np.r_[np.linspace(-np.pi,np.pi,721)[:-1],np.arctan2(border[:,1],border[:,0])])
    direction=np.c_[np.cos(angles),np.sin(angles)]
    def cross(a,b):
        return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    edges=np.roll(border,-1,axis=0)-border
    denominator=cross(direction[:,None],edges)
    with np.errstate(invalid='ignore',divide='ignore'):
        along=cross(border,edges)/denominator
        segment=cross(border,direction[:,None])/denominator
    distance=np.min(np.where((segment>=-1e-9)&(segment<=1+1e-9)&(along>0),along,np.inf),axis=1)
    assert np.isfinite(distance).all()
    # Reuse the exact straight edges of the 180-vertex casting at the inner boundary.
    step=2*np.pi/len(outer)
    inner_radius=s['outer_diameter_m']/2*np.cos(step/2)/np.cos((angles%step)-step/2)
    radii=np.linspace(inner_radius,distance,81)
    v=xyz(radii[:,:,None]*direction,0.).reshape(-1,3)
    ix=np.arange(radii.size).reshape(radii.shape)
    a=ix[:-1].ravel();b=np.roll(ix[:-1],-1,axis=1).ravel()
    c=np.roll(ix[1:],-1,axis=1).ravel();d=ix[1:].ravel()
    f=face_up(v,np.vstack([np.c_[a,b,c],np.c_[a,c,d]]))
    name='Road adjoining the fracture';material=pavement
    pixel=np.int32(np.rint(v[f].mean(axis=1)[:,:2]/scale+center))
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hue,sat,value=hsv[pixel[:,1],pixel[:,0]].T
    paint=(hue>=13)&(hue<=37)&(sat>51)&(value>77)
    uv=project(v)
    mesh.add(name,v,f[~paint],uv,material,True)
    mesh.add('Textured road marking edge',v,f[paint],uv,material)
    face_polygon(mesh,'Cast ring and cover seat',xyz(outer,0),[xyz(inner,0)],iron,project)
    wall(mesh,'Cast seat riser and ledge',xyz(inner,0),xyz(inner,ledge_z),shade)
    face_polygon(mesh,'Circular cover rebate',xyz(inner,ledge_z),[xyz(lid,ledge_z)],shade,lambda p:p[:,:2])
    face_polygon(mesh,'Lid face',xyz(lid,s['lid_z_m']),[],iron,project)
    wall(mesh,'Lid rolled edge',xyz(lid[::-1],s['lid_z_m']),xyz(lid[::-1],ledge_z),shade)
    span=2*np.max(abs(border),axis=0)+.015
    return span,dict(outer_diameter_model_m=s['outer_diameter_m'],lid_height_m=s['lid_z_m'],
                     pixel_to_local_scale=scale.tolist(),pixel_center=center.tolist())


def broken(mesh,img,s):
    world=rect(s['frame_m'])[:,:2].astype('f4')
    H=cv2.getPerspectiveTransform(np.float32(s['frame_quad']),world)
    def warp(points,matrix):
        return cv2.perspectiveTransform(np.float64(points)[None,:,:2],matrix)[0]
    local=lambda p:warp(p,H)
    project=lambda p:texture_uv(warp(p,np.linalg.inv(H)),s['size'])
    y,x=np.indices(img.shape[:2])
    # The safety barrier belongs to the street photograph, not the cover. Rebuild only the
    # parts it obscures from material sampled on the same cover and record the occluded pixels.
    mask=np.zeros(img.shape[:2],'u1')
    cv2.fillPoly(mask,[np.int32(poly) for poly in s['occluded_regions']],1)
    red=(img[:,:,0].astype(float)>img[:,:,1]*1.19)&(img[:,:,0].astype(float)>img[:,:,2]*1.18)
    mask|=cv2.dilate(red.astype('u1'),np.ones((7,7),'u1'))
    def donor(key):
        a,b,c,d=s[key];return img[b:d,a:c]
    def restore(chip):
        out=img.copy();fill=chip[y%chip.shape[0],x%chip.shape[1]]
        out[mask.astype(bool)]=fill[mask.astype(bool)];return out
    stone=donor('donor_concrete');edge=mesh.material('Concrete fracture edges',np.uint8(stone*.78),.98,0,.8)
    slab=mesh.material('Photographed fragments',restore(stone),.91,.01,.55,lossless_color=True)
    frame=mesh.material('Cast telecommunications frame',restore(donor('donor_frame')),.77,.18,.6)
    inner=local(s['socket'])
    face_polygon(mesh,'Cast ring and cover seat',xyz(world,0),[xyz(inner,0)],frame,project)
    dark=mesh.material('Inferred dark cavity',np.uint8(stone*.18),.95,0,.5)
    floor=-s['shaft_depth_m']
    wall(mesh,'Inspection shaft',xyz(inner[::-1],0),xyz(inner[::-1],floor),dark)
    face_polygon(mesh,'Shaft bottom',xyz(inner,floor),[],dark,lambda p:p[:,:2]*3)
    for poly,z in zip(s['pieces'],s['fragment_z_m']):
        shape=local(poly)
        top=xyz(shape,z)
        # Keep the three topological islands distinct inside each semantic primitive.
        face_polygon(mesh,'Lid fragment face',top,[],slab,project)
        area=np.cross(shape,np.roll(shape,-1,axis=0)).sum()
        if area>0:top=top[::-1]
        base=top.copy();base[:,2]-=s['thickness_m']
        wall(mesh,'Lid rolled edge',top,base,edge,6)
        from photo_mesh import rings_faces
        faces=face_up(base,rings_faces([base[:,:2]]))[:,::-1]
        mesh.add('Lid underside',base,faces,base[:,:2]*3,edge)
    merged=[]
    for name in dict.fromkeys(p[0] for p in mesh.parts):
        pieces=[q for q in mesh.parts if q[0]==name];verts=[];faces=[];uv=[]
        for _,v,f,u,m,road in pieces:
            faces.append(f+sum(map(len,verts)));verts.append(v);uv.append(u)
        merged.append((name,np.vstack(verts),np.vstack(faces),np.vstack(uv),m,road))
    mesh.parts=merged
    return np.asarray(s['frame_m'])+.02,dict(pixel_to_local_homography=H.tolist(),shaft_depth_model_m=-floor,
                model_thickness_m=s['thickness_m'],fragment_count=3,
                inferred_parts=['occluded lower perimeter','hidden underside','shaft walls and bottom'])


def build(out):
    specs=json.loads((HERE/'covers.json').read_text());series=out/'series';rows=[]
    import riverview
    specs['P562']=json.loads((HERE/'riverview.json').read_text())
    for key,s in specs.items():
        src=HERE/'references'/(key+'.jpg')
        assert hashlib.sha256(src.read_bytes()).hexdigest()==s['photo_sha256']
        img=np.asarray(Image.open(src).convert('RGB').resize(tuple(s['size']),Image.Resampling.LANCZOS))
        mesh=Mesh()
        span,extra={'A560':painted,'M561':broken,'P562':riverview.make}[key](mesh,img,s)
        cat={'A560':'cover','M561':'missing','P562':'pothole'}[key];folder=series/'assets'/cat/key
        xy=grid(span,max(span)/420);ext=np.ptp(xy,axis=(0,1))
        meta=dict(id=key,label=s['label'],source=s['source'],source_author=s['author'],source_license=s['license'],
                  source_sha256=s['photo_sha256'],category=cat,family='field_rims',state='photo_reconstruction',
                  seed=27000+int(key[1:]),surface_dimensions_m=ext.tolist(),scale_basis='Nominal single-photo scale; hidden geometry inferred.',**extra)
        cache=series/'mesh_cache'
        mesh.save(folder,cache,meta,dict(width=ext[0],length=ext[1],height=np.zeros(xy.shape[:2])))
        flatten_scene(folder/'asset.glb');surf=Surface(folder/'asset.glb')
        start=xyz(xy,1.);dist,part,_=surf.cast(start,np.broadcast_to([0.,0.,-1.],start.shape))
        hit=np.isfinite(dist);z=np.where(hit,1.-dist,0.)
        # The Redhill asset includes the two adjoining painted strips within the measured patch.
        install=part>1
        field=dict(width=ext[0],length=ext[1],height=z.astype('f4'),
            alpha=hit.astype('f4'),defect_mask=((part>1)&(part!=114)),installation_mask=install,
            opening_mask=np.isin(part,[6,7]),water_mask=part==4,photographic_rgb=True)
        if key=='P562':
            bare=Surface(folder/'asset.glb',exclude={4})
            depth,_,_=bare.cast(start,np.broadcast_to([0.,0.,-1.],start.shape))
            field['substrate_height']=np.where(np.isfinite(depth),1.-depth,0.).astype('f4')
        np.savez_compressed(folder/'surface.npz',**field)
        row=make_one(folder,cache,point_limit=380000,kernel_factors=(.76,.16))
        for role,file in [('glb','asset.glb'),('ply','asset.ply'),('surface','surface.npz'),('metadata','metadata.json')]:
            row[role]=f'{cat}/{key}/{file}'
        rows.append(row)
    write_json(series/'assets/asset_manifest.json',rows);write_json(out/'run.json',dict(editions=['series']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    build(parser.parse_args().out)
