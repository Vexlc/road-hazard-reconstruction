"""Photo-registered chamber surfaces and volumes. Source coordinates are pixel centres."""
import cv2
import mapbox_earcut
import numpy as np
from scipy.ndimage import distance_transform_edt


def rings_faces(rings):
    pixels = np.ascontiguousarray(np.vstack(rings), dtype='f8')
    ends = np.cumsum([len(ring) for ring in rings], dtype='u4')
    return mapbox_earcut.triangulate_float64(pixels, ends).reshape(-1, 3)


def refine(vertices, faces, resolution=.042):
    while True:
        edges=np.sort(np.stack((faces,np.roll(faces,-1,axis=1)),axis=2),axis=2)
        if np.linalg.norm(vertices[edges[:,:,0]]-vertices[edges[:,:,1]],axis=2).max() <= resolution:
            return vertices,faces
        unique, idx = np.unique(edges.reshape(-1,2),axis=0,return_inverse=True)
        a,b,c=faces.T
        ab,bc,ca=(idx.reshape(-1,3)+len(vertices)).T
        vertices=np.vstack([vertices,vertices[unique].mean(1)])
        faces=np.vstack([np.c_[a,ab,ca],np.c_[b,bc,ab],np.c_[c,ca,bc],np.c_[ab,bc,ca]])


def toward(vertices,faces,eye):
    a,b,c=vertices[faces].transpose(1,0,2)
    flip=np.sum(np.cross(b-a,c-a)*(eye-a),axis=1)<0
    faces=faces.copy()
    faces[flip]=faces[flip,::-1]
    return faces


def photographed(mesh, cam, name, rings, heights, material, verge=False):
    vertices=np.vstack([cam.world(p,z) for p,z in zip(rings,heights)])
    faces=toward(vertices,rings_faces(rings),cam.eye)
    vertices,faces=refine(vertices,faces)
    mesh.add(name,vertices,faces,cam.uv(vertices),material,verge)


def case(mesh,cam,name,shape,floor,top_material,side_material):
    poly=np.array(shape['poly'])
    if shape.get('smooth'):
        nxt=np.roll(poly,-1,axis=0)
        poly=np.stack([.85*poly+.15*nxt,.15*poly+.85*nxt],axis=1).reshape(-1,2)
    z=shape['z_m']
    width=shape['bevel_m']
    outer=cam.world(poly,z-width)
    # The visible footprint sets the outline. Extrude the hidden sides vertically.
    top=outer.copy()
    center=outer[:,:2].mean(0)
    vec=center-top[:,:2]
    top[:,:2]+=width*vec/np.linalg.norm(vec,axis=1,keepdims=True)
    top[:,2]=z
    base=outer.copy();base[:,2]=floor-.001
    n=len(poly);a=np.arange(n);b=np.roll(a,-1)
    strip=np.vstack([np.c_[a,b,b+n],np.c_[a,b+n,a+n]])
    vertices=np.vstack([top,outer])
    faces=np.vstack([rings_faces([poly]),strip])
    vertices,faces=refine(vertices,toward(vertices,faces,cam.eye),.012)
    if 'fold_px' in shape:
        # Preserve the photographed hinge between the sloped top and barcode face.
        pix=cam.pixels(vertices)
        row=pix[:,1]
        crest=shape['fold_px']
        drop=np.interp(row,[row.min(),crest,crest+4,row.max()],[.008,0,.005,.031])
        vertices=cam.world(pix,vertices[:,2]-drop)
        px=cam.pixels(outer)
        lower_rim=np.interp(px[:,1],[row.min(),crest,crest+4,row.max()],[.008,0,.005,.031])
        outer=cam.world(px,outer[:,2]-lower_rim)
        base=outer.copy();base[:,2]=floor-.001
    mesh.add(name,vertices,faces,cam.uv(vertices),top_material)
    v=np.vstack([outer,base])
    # Outward-facing side wall. Image polygons run clockwise in image coordinates.
    f=strip.copy()
    aa,bb,cc=v[f].transpose(1,0,2)
    outward=aa.copy();outward[:,:2]-=center;outward[:,2]=0
    flip=(np.cross(bb-aa,cc-aa)*outward).sum(1)<0
    f[flip]=f[flip,::-1]
    uv=np.c_[np.tile(np.linspace(.02,.98,n),2),np.repeat([.95,.05],n)]
    mesh.add(name+' side',v,f,uv,side_material)


def tube(mesh,cam,name,spec,material):
    pixels=np.array(spec['points'],float)
    centers=cam.world(pixels,spec['z_m'])
    radius=np.linalg.norm(cam.world(pixels+np.c_[spec['radius_px'],np.zeros(len(pixels))],spec['z_m'])-centers,axis=1)
    tangent=np.gradient(centers,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    across=np.cross(tangent,[0.,0.,1.])
    angles=np.arange(16)*2*np.pi/16
    vec=across[:,None,:]*np.cos(angles)[None,:,None]+np.array([0,0,1.])*np.sin(angles)[None,:,None]
    v=(centers[:,None,:]+radius[:,None,None]*vec).reshape(-1,3)
    a=(np.arange(len(centers)-1)[:,None]*16+np.arange(16)).ravel()
    b=(a//16)*16+(a+1)%16
    f=np.vstack([np.c_[a,b+16,b],np.c_[a,a+16,b+16]])
    va,vb,vc=v[f].transpose(1,0,2)
    flip=(np.cross(vb-va,vc-va)*(va-centers[f[:,0]//16])).sum(1)<0
    f[flip]=f[flip,::-1]
    # Close each end even where it meets the next fitting.
    ends=np.arange(1,15)
    caps=np.vstack([np.c_[0*ends,ends,ends+1],16*(len(centers)-1)+np.c_[0*ends,ends+1,ends]])
    for face in caps:
        a,b,c=v[face]
        target=tangent[-1] if face[0] else -tangent[0]
        if np.cross(b-a,c-a)@target<0:face[:]=face[::-1]
    mesh.add(name,v,np.vstack([f,caps]),cam.uv(v),material)


def paint_mask(shape,spec):
    mask=np.zeros(shape,'u1')
    for obj in spec['objects'].values():
        cv2.fillPoly(mask,[np.int32(obj['poly'])],255)
    for tube in spec['pipes'].values():
        line=np.array(tube['points'])
        for a,b,ra,rb in zip(line,line[1:],tube['radius_px'],tube['radius_px'][1:]):
            cv2.line(mask,tuple(a),tuple(b),255,round(2*max(ra,rb)+4))
    return cv2.dilate(mask,np.ones((5,5),'u1'))


def materials(mesh,rgb,spec):
    grass=[spec['pipes'][k] for k in ['dry_stem','right_stem']]
    def without_stems(image):
        m=np.zeros(image.shape[:2],'u1')
        for tube in grass:
            cv2.polylines(m,[np.int32(tube['points'])],False,255,4)
        return cv2.inpaint(image,m,3,cv2.INPAINT_TELEA)
    clean=without_stems(rgb)
    mask=paint_mask(rgb.shape[:2],spec)
    # Fill the unobserved leaf litter with patches sampled wholly within exposed leaf litter.
    allowed=np.zeros_like(mask);cv2.fillPoly(allowed,[np.int32(spec['floor_visible'])],255)
    allowed[mask>0]=0
    dist=distance_transform_edt(allowed>0)
    candidates=np.argwhere(dist>=15)
    rng=np.random.default_rng(627073)
    floor=clean.copy()
    for y in range(0,len(mask),18):
        for x in range(0,mask.shape[1],18):
            m=mask[y:y+18,x:x+18]>0
            if not m.any():continue
            row,col=candidates[rng.integers(len(candidates))]
            patch=clean[row-9:row+9,col-9:col+9][:m.shape[0],:m.shape[1]]
            floor[y:y+18,x:x+18][m]=patch[m]
    def new(name,photo,rough=.89):
        return mesh.material(name,photo,rough,.0,.30,lossless_color=True)
    return dict(frame=new('Concrete and grass from Richard Webb',clean), floor=new('Observed leaf litter and concealed fill',floor),
                equipment=new('Photographed equipment',clean,.66), stems=new('Crossing dry stems',rgb))


def shaft_wall(mesh,cam,spec,rgb):
    upper=cam.world(spec['rim_inner_top'],-spec['seat_drop_m'])
    lower=cam.world(spec['floor_visible'],-spec['floor_depth_m'])
    n=len(upper)
    # The foreground wall occupies fewer than a dozen source rows. Resample a clean
    # concrete patch on the far wall wherever the in-plane sampling collapses.
    texture=np.zeros((n*100,384,3),'u1')
    vertices=[];faces=[];uv=[]
    rows,cols=np.meshgrid((np.arange(100)+.5)/100,(np.arange(384)+.5)/384,indexing='ij')
    for i in range(n):
        j=(i+1)%n
        def interpolate(s,t):
            top=upper[i]+s[...,None]*(upper[j]-upper[i])
            bottom=lower[i]+s[...,None]*(lower[j]-lower[i])
            return top+(bottom-top)*t[...,None]
        points=interpolate(cols,rows)
        p=cam.pixels(points).astype('f4')
        observed=cv2.remap(rgb,p[:,:,0],p[:,:,1],cv2.INTER_LINEAR)
        up=cam.pixels(interpolate(cols,0*rows));down=cam.pixels(interpolate(cols,0*rows+1))
        tangent=cam.pixels(upper[j])-cam.pixels(upper[i]);tangent/=np.linalg.norm(tangent)
        projected=np.abs((up-down)@np.array([-tangent[1],tangent[0]]))
        confidence=np.clip((projected-13)/18,0,1)
        confidence=confidence**2*(3-2*confidence)
        donor=rgb[115:132,135:214]
        width=np.linalg.norm(upper[j]-upper[i])
        sx=(cols*width/.0017+i*21).astype('f4');sy=(rows*.114/.0017+i*9).astype('f4')
        restored=cv2.remap(donor,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        restored=restored*(1-.24*rows[...,None])
        mixture=observed*confidence[...,None]+restored*(1-confidence[...,None])
        texture[100*i:100*(i+1)]=np.uint8(mixture.clip(0,255))
        tt,ss=np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,33),indexing='ij')
        v=interpolate(ss,tt).reshape(-1,3)
        # Mesh UVs are bottom-up; each strip owns an inset atlas interval.
        texcoords=np.stack([(.5+383*ss)/384,1-(100*i+.5+99*tt)/(n*100)],axis=-1)
        a=np.arange(v.shape[0]).reshape(tt.shape)[:-1,:-1].ravel()
        f=np.vstack([np.c_[a,a+1,a+34],np.c_[a,a+34,a+33]])
        faces.append(toward(v,f,cam.eye)+sum(map(len,vertices)))
        vertices.append(v);uv.append(texcoords.reshape(-1,2))
    key=mesh.material('Original and restored concrete wall',texture,.95,0,.38,lossless_color=True)
    mesh.add('Inspection shaft',np.vstack(vertices),np.vstack(faces),np.vstack(uv),key)
