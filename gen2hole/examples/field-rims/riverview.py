"""Fit a nested pavement patch, exposed ledge, waterline and five emergent pieces of aggregate."""
import cv2
import numpy as np
from photo_mesh import RoadCamera, face_polygon, face_up, refine, rings_faces, texture_uv, xyz


def make(mesh,img,s):
    cam=RoadCamera(s)
    def at(pixel,z):
        grade=cam.ground(pixel)
        return xyz(cam.eye[:2]+(grade-cam.eye[:2])*(1-z/cam.eye[2]),z)
    project=lambda v:texture_uv(cam.project(v),s['size'])
    def material(name,rgb,rough=.94):
        return mesh.material(name,rgb,rough,0,.55,lossless_color=True)
    asphalt=material('Riverview weathered asphalt',img)
    ring=at(s['rim'],0.);lip=at(s['lip'],-.009)
    crest=at(s['shelf'],-.047);water=at(s['water'],s['water_z_m'])
    # The external fringe only supplies neighbouring asphalt for host exposure matching.
    mid=ring.mean(axis=0)
    fringe=mid+(ring-mid)*1.09
    face_polygon(mesh,'Road adjoining the fracture',fringe,[ring],asphalt,project)
    # Triangulate the nested boundaries in the original image plane. The lower
    # rim can overhang in XY when the view grazes the near side of the pool.
    def band(name,outer,inner):
        vertices=np.vstack([outer,inner])
        faces=rings_faces([cam.project(outer),cam.project(inner)])
        vertices,faces=refine(vertices,faces,.045)
        a,b,c=vertices[faces].transpose(1,0,2)
        flip=np.sum(np.cross(b-a,c-a)*(cam.eye-a),axis=1)<0
        faces[flip]=faces[flip,::-1]
        mesh.add(name,vertices,faces,project(vertices),asphalt)
    band('Fracture lip and exposed bed',ring,lip)
    band('Measured rim shelf 01',lip,crest)
    band('Fracture sidewall in surface coordinates',crest,water)
    # The source water reflection supplies a colour reference. It is excluded from the unseen bed.
    a,b,c,d=s['bed_donor'];grain=img[b:d,a:c]
    floor=material('Concealed aggregate under the pool',np.uint8(grain*.57))
    bowl=water.copy();mean=water.mean(0);bowl[:,:2]=mean[:2]+.74*(water[:,:2]-mean[:2]);bowl[:,2]=s['bed_z_m']
    face_polygon(mesh,'Subgrade bearing',water,[bowl],floor,lambda v:v[:,:2]*8)
    face_polygon(mesh,'Subgrade bearing surface',bowl,[],floor,lambda v:v[:,:2]*8)
    sample=img[548:555,360:371];color=np.median(sample,axis=(0,1))
    # A neutral overcast tint keeps water separable from both dark lip and light stone at driving distance.
    tint=np.broadcast_to(np.uint8(color*.61), (16,16,3))
    liquid=mesh.material('Standing overcast water',tint,.07,0,0);mesh.ior[liquid]=1.333
    face_polygon(mesh,'Water surface',water,[],liquid,project)
    rock=material('Exposed individual aggregate',img)
    for i,(outline,height) in enumerate(zip(s['stones'],s['crest_m'])):
        edge=np.asarray(outline,float)
        z=s['water_z_m'];base=at(edge,z-.004);center=edge.mean(0)
        top=at(center+(edge-center)*.57,z+height*.69)
        tip=at(center[None,:],z+height);vertices=np.vstack([base,top,tip])
        n=len(edge);k=np.arange(n);j=(k+1)%n
        faces=face_up(vertices,np.vstack([np.c_[k,j,n+j],np.c_[k,n+j,n+k],np.c_[n+k,n+j,k*0+2*n]]))
        mesh.add(f'Loose stone {i+1:02}',vertices,faces,project(vertices),rock)
    span=2*np.max(abs(fringe[:,:2]),axis=0)+.02
    return span,dict(outer_patch_long_axis_model_m=s['major_m'],water_level_m=s['water_z_m'],
                     concealed_floor_m=s['bed_z_m'],source_camera_eye_m=cam.eye.tolist(),
                     emergent_stones=len(s['stones']),lip_xy_m=lip[:,:2].tolist())
