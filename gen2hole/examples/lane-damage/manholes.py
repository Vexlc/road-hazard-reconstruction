"""Build a brick-lined shaft and two individually traced fragments of a polymer cover."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from photo_mesh import xyz,rect,texture_uv,face_polygon,rings_faces,refine,face_up,remap


def pit(mesh,img,s):
    size=np.array(s['size']); top=rect([s['width_m'],s['length_m']])
    pix=np.float64(s['rim'])
    # A four-corner planar calibration constrains roll and inclination. Focal length and scale remain nominal.
    focal=s['focal_px_from_exif']
    K=np.array([[focal,0,size[0]/2],[0,focal,size[1]/2],[0,0,1.]])
    ok,rv,t=cv2.solvePnP(top,pix,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('Could not recover camera')
    observations=np.vstack([pix,s['wall_bottom_corner']])
    def residual(parameters):
        length,depth=parameters[-2:]
        upper=rect([s['width_m'],length]);corner=upper[1].copy();corner[2]=-depth
        points=np.vstack([upper,corner])
        predicted=cv2.projectPoints(points,parameters[:3],parameters[3:6],K,None)[0].reshape(-1,2)
        return (predicted-observations).ravel()
    solution=least_squares(residual,np.r_[rv.ravel(),t.ravel(),s['length_m'],1.])
    length,depth=solution.x[-2:];floor=-depth
    top=rect([s['width_m'],length])
    R=cv2.Rodrigues(solution.x[:3])[0];t=solution.x[3:6];eye=(-R.T@t).ravel()
    def project(v):
        v=(np.asarray(v)@R.T)+t.ravel()
        return v[...,:2]/v[...,2:]*[K[0,0],K[1,1]]+size/2
    def unproject(p,z):
        q=np.asarray(p,float);v=np.concatenate([(q-size/2)/[K[0,0],K[1,1]],np.ones(q.shape[:-1]+(1,))],-1)@R
        return eye+v*(z-eye[2])/v[...,2:]
    # Align the bottom of the far brick corner with its separately marked pixel.
    def discrepancy(z):
        return np.linalg.norm(project(xyz(top[1,:2],z))-s['wall_bottom_corner'])
    outer=unproject(s['outer_frame'],0);ledge=unproject(s['inner_footprint'],-.018)
    depth=float(-floor);collar=unproject(s['concrete_collar'],0);span=2*np.max(abs(collar[:,:2]),axis=0)+.14
    photo=mesh.material('Concrete surround and bearing from the original',img,.92,.02,.35,lossless_color=True)
    from photo_mesh import LIB
    from PIL import Image
    asphalt=np.asarray(Image.open(LIB/'examples/perforated-covers/split-iron/pavement.jpg').convert('RGB'))
    ground=mesh.material('Receiving asphalt adjoining original concrete',asphalt,.94,0,.5)
    face_polygon(mesh,'Road around the aperture',rect(span),[collar],ground,lambda v:.5+v[:,:2]/1.5)
    face_polygon(mesh,'Concrete frame on verge',collar,[top],photo,lambda v:texture_uv(project(v),size))
    # Build each face with a dedicated material; masked source projection identifies unseen brick.
    templates=[]
    rows,cols=np.meshgrid((np.arange(384)+.5)/384,(np.arange(512)+.5)/512,indexing='ij')
    def sheet(a,b,ss,tt,z0,z1):return xyz(a[:2]+(b[:2]-a[:2])*ss[...,None],z0+(z1-z0)*tt)
    for i in range(4):
        j=(i+1)%4;v=np.vstack([top[i],top[j],ledge[j],ledge[i]])
        # Upward lip has a slight inward offset.
        f=np.array([[0,1,2],[0,2,3]])
        a,b,c=v[f].transpose(1,0,2);flip=(np.cross(b-a,c-a)*(eye-a)).sum(1)<0;f[flip]=f[flip,::-1]
        name='Cast ring and cover seat'
        # Accumulate strips under a single GLB part.
        if not i:stripv=[];stripf=[]
        stripv.append(v);stripf.append(f+i*4)
        world=sheet(ledge[i],ledge[j],cols,rows,-.018,floor)
        tile=remap(img,project(world)*np.array(img.shape[1::-1])/size)
        templates.append(tile)
    v=np.vstack(stripv);mesh.add(name,v,np.vstack(stripf),texture_uv(project(v),size),photo)
    vv=[];ff=[];uv=[];atlas=[]
    for i in range(4):
        j=(i+1)%4
        donor=templates[i%2]
        # Only two walls face the survey camera. The opposite two use the observed brick courses.
        tile=templates[i].copy() if i in (0,1) else donor[:,::-1].copy()
        if i in (0,1):
            p=sheet(ledge[i],ledge[j],cols,rows,-.018,floor)
            crossing=p+(eye-p)*((-p[:,:,2])/(eye[2]-p[:,:,2]))[:,:,None]
            # Borrow from the middle of the same course where the near rim hides this wall.
            valid=(abs(crossing[:,:,:2])<np.array([s['width_m'],length])/2-.007).all(2)
            clean=templates[i].copy()
            for y,row in enumerate(valid):
                xs=np.flatnonzero(row)
                if len(xs)>20:
                    offset=np.arange(tile.shape[1])-xs[0]; clean[y]=tile[y,xs[np.mod(offset,len(xs))]]
                else:clean[y]=templates[i][max(0,y-125)]
            tile=np.where(valid[:,:,None],tile,clean)
        # Use restored course texture for the concealed rear wall as well.
        elif atlas:tile=atlas[i-2][:,::-1]
        atlas.append(tile)
        tt,ss=np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,21),indexing='ij')
        v=sheet(ledge[i],ledge[j],ss,tt,-.018,floor).reshape(-1,3)
        a=np.arange(tt.size).reshape(tt.shape)[:-1,:-1].ravel()
        f=np.vstack([np.c_[a,a+1,a+22],np.c_[a,a+22,a+21]])
        aa,bb,cc=v[f].transpose(1,0,2);flip=(np.cross(bb-aa,cc-aa)*(-aa*[1,1,0])).sum(1)<0;f[flip]=f[flip,::-1]
        ff.append(f+i*len(v));vv.append(v);uv.append(np.stack([(.5+511*ss)/512,1-(384*i+.5+383*tt)/1536],-1).reshape(-1,2))
    wall=mesh.material('Photographed brick courses and restored back faces',np.vstack(atlas),.96,0,.5)
    mesh.add('Inspection shaft',np.vstack(vv),np.vstack(ff),np.vstack(uv),wall)
    v=ledge.copy();v[:,2]=floor
    uvf=np.stack([cols,rows],axis=-1)
    # Bilinear floor atlas follows the metric rectangle and removes photographed foreground walls.
    surface=(1-rows[...,None])*((1-cols[...,None])*v[0]+cols[...,None]*v[1])+rows[...,None]*((1-cols[...,None])*v[3]+cols[...,None]*v[2])
    pp=project(surface)*np.array(img.shape[1::-1])/size;captured=remap(img,pp)
    cross=surface+(eye-surface)*(-floor/(eye[2]-floor))
    seen=(abs(cross[:,:,:2])<np.array([s['width_m'],length])/2-.012).all(2)
    patch=img[round(324*img.shape[0]/600):round(341*img.shape[0]/600),round(433*img.shape[1]/800):round(454*img.shape[1]/800)]
    yy,xx=np.indices(rows.shape);covered=patch[yy%len(patch),xx%patch.shape[1]]
    floor_mat=mesh.material('Visible shaft floor and concealed soil',np.where(seen[:,:,None],captured,covered),.95,0,.4)
    mesh.add('Shaft bottom',v,face_up(v,np.array([[0,1,2],[0,2,3]])),[[0,1],[1,1],[1,0],[0,0]],floor_mat)
    return span,dict(shaft_depth_model_m=depth,far_corner_residual_px=discrepancy(floor),source_camera_eye_m=eye.tolist(),source_camera_rotation=R.tolist(),source_camera_intrinsics=K.tolist(),opening_width_m=s['width_m'],opening_length_m=float(length),rim_rms_px=float(np.sqrt(np.mean(residual(solution.x)[:8]**2))))


def pieces(mesh,img,s,asphalt):
    factor=s['scale_m_per_pixel'];center=np.array([472.,200.])
    def px(v):return v[:,:2]/[factor,-factor]+center
    topmat=mesh.material('Photographed composite fragment faces',img,.87,.03,.5,lossless_color=True)
    broken=mesh.material('Dark broken polymer composite edge',img[round(img.shape[0]*.43):round(img.shape[0]*.445),round(img.shape[1]*.69):round(img.shape[1]*.72)],.92,0,.3)
    groups=[[],[],[]]
    for shape in s['pieces']:
        poly=np.array(shape['poly']);xy=(poly-center)*[factor,-factor]
        v=xyz(xy,shape['z']-.011);f=face_up(v,rings_faces([xy]));v,f=refine(v,f,.005)
        pixels=px(v);dist=np.full(len(v),1e5)
        for rib in shape['ribs']:
            for a,b in zip(np.array(rib),np.array(rib[1:])):
                ab=b-a;t=np.clip((pixels-a)@ab/(ab@ab),0,1)
                dist=np.minimum(dist,np.linalg.norm(pixels-a-t[:,None]*ab,axis=1))
        v[:,2]+=.011*np.exp(-.5*(dist/3.)**4)
        groups[0].append((v,f,texture_uv(pixels,s['size'])))
        # Extrude the actual refined top boundary. Split side and bottom edges at the
        # identical vertices so each fragment is a closed solid even at the broken teeth.
        directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
        _,first,count=np.unique(np.sort(directed,axis=1),axis=0,return_index=True,return_counts=True)
        boundary=directed[first[count==1]]
        ids,local=np.unique(boundary,return_inverse=True)
        a,b=local.reshape(-1,2).T;n=len(ids)
        bottom=v.copy();bottom[:,2]=0
        side=np.vstack([v[ids],bottom[ids]])
        sf=np.vstack([np.c_[b,a,a+n],np.c_[b,a+n,b+n]])
        groups[1].append((side,sf,np.c_[np.tile(np.linspace(0,1,n),2),np.repeat([.99,.01],n)]))
        groups[2].append((bottom,f[:,::-1],texture_uv(pixels,s['size'])))
    for name,group,material in zip(['Lid fragment face','Lid fragment aperture walls and perimeter','Lid underside'],groups,[topmat,broken,broken]):
        offsets=np.cumsum([0]+[len(v) for v,_,_ in group])
        mesh.add(name,np.vstack([v for v,_,_ in group]),np.vstack([f+o for (_,f,_),o in zip(group,offsets)]),np.vstack([uv for _,_,uv in group]),material)
    span=[.96,.52];ground=rect(span,-.00015)
    road=mesh.material('Asphalt031 receiving surface',asphalt,.94,0,.5)
    face_polygon(mesh,'Road around the aperture',ground,[],road,lambda v:.5+v[:,:2]/[1.4,1.4])
    return span,dict(fragment_count=2,nominal_m_per_source_pixel=factor,individual_footprints_m=[(np.ptp(p['poly'],axis=0)*factor).tolist() for p in s['pieces']])
