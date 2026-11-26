"""Trace three neighbouring concrete panels across the damaged Spurn Head road."""
import numpy as np
from photo_mesh import RoadCamera,texture_uv,grid_faces


def make(mesh,img,s):
    cam=RoadCamera(s);material=mesh.material('Photographed worn concrete',img,.92,0,.4,lossless_color=True)
    # The source shows coarse aggregate on the exposed front riser of the middle panel.
    chip=img[245:253,150:171]
    edge=mesh.material('Exposed aggregate from the broken front riser',chip,.97,0,.6)
    vertices=[];faces=[];tex=[];sides=[];side_faces=[];side_uv=[]
    cuts=[];bounds=[]
    for piece in s['sections']:
        uv=np.array([piece['back'][0],piece['back'][1],piece['front'][1],piece['front'][0]],float)
        z=np.array([*piece['back_z'],*piece['front_z'][::-1]])
        grade=cam.ground(uv)
        actual=cam.eye[:2]+(grade-cam.eye[:2])*(1-z[:,None]/cam.eye[2])
        corners=np.c_[actual,z]
        t,u=np.meshgrid(np.linspace(0,1,49),np.linspace(0,1,45),indexing='ij')
        def interpolate(q):
            return (1-t[...,None])*((1-u[...,None])*q[0]+u[...,None]*q[1])+t[...,None]*((1-u[...,None])*q[3]+u[...,None]*q[2])
        v=interpolate(corners);f=grid_faces(*t.shape)
        a,b,c=v.reshape(-1,3)[f[0]]
        if np.cross(b-a,c-a)[2]<0:f=f[:,::-1]
        vertices.append(v.reshape(-1,3));faces.append(f+sum(map(len,vertices[:-1])))
        tex.append(texture_uv(cam.project(v),s['size']).reshape(-1,2))
        for i in range(4):
            j=(i+1)%4
            # Make a contiguous rough section below the photographed wearing surface.
            top=np.linspace(corners[i],corners[j],40)
            base=top.copy();base[:,2]-=.155
            strip=np.vstack([top,base])
            k=np.arange(39)
            sf=np.vstack([np.c_[k,k+1,k+41],np.c_[k,k+41,k+40]])
            a,b,c=strip[sf].transpose(1,0,2);normal=a-corners.mean(0);normal[:,2]=0
            flip=(np.cross(b-a,c-a)*normal).sum(1)<0;sf[flip]=sf[flip,::-1]
            side_faces.append(sf+sum(map(len,sides)));sides.append(strip);side_uv.append(np.c_[np.tile(np.linspace(0,8,40),2),np.repeat([.98,.02],40)])
        bounds.extend(corners)
        cuts.append(dict(panel=piece['name'],corner_xyz_m=corners.tolist()))
    mesh.add('Detached pavement top',np.vstack(vertices),np.vstack(faces),np.vstack(tex),material)
    mesh.add('Detached pavement edges',np.vstack(sides),np.vstack(side_faces),np.vstack(side_uv),edge)
    # Inset dark substrate is revealed through truly open transverse seams.
    from photo_mesh import rect,face_polygon
    bounds=np.array(bounds);span=2*np.max(abs(bounds[:,:2]),axis=0)+.10
    dark=mesh.material('Subbase in open seams',np.uint8(chip*.36),.98,0,.4)
    for i in range(2):
        lo=np.array(cuts[i]['corner_xyz_m'])[:2]
        hi=np.array(cuts[i+1]['corner_xyz_m'])[3:1:-1]
        base=np.vstack([lo,hi[::-1]]);base[:,2]=min(lo[:,2].min(),hi[:,2].min())-.17
        if not i:vv=[];ff=[]
        from photo_mesh import face_up
        ff.append(face_up(base,np.array([[0,1,2],[0,2,3]]))+4*i);vv.append(base)
    mesh.add('Subgrade bearing',np.vstack(vv),np.vstack(ff),np.vstack(vv)[:,:2],dark)
    return span,dict(nominal_traced_span_m=s['major_m'],panels=cuts,source_camera_eye_m=cam.eye.tolist())
