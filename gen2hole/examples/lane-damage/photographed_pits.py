"""Build depressions around the photographed rim and keep the water surface independent."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from photo_mesh import RoadCamera,grid,xyz,poly_mask,texture_uv,grid_faces,remap


def make(mesh,img,s):
    cam=RoadCamera(s);poly=cam.ground(s['rim']);span=np.ptp(poly,axis=0)+[.28,.28]
    pitch=.004 if 'water' in s else .005
    xy=grid(span,pitch)
    mask=poly_mask(xy,poly)
    distance=distance_transform_edt(mask)*pitch
    limit=max(.04,np.percentile(distance[mask],78))
    profile=np.sqrt(np.clip(distance/limit,0,1))
    z=-s['depth_m']*profile
    # A shallow rough floor retains the recess while avoiding a perfectly planar central patch.
    z*=.89+.11*np.sin(2.7*xy[:,:,1]+.6)*np.cos(4.2*xy[:,:,0])
    pos=xyz(xy,z);f=grid_faces(*mask.shape)
    defect=mask.ravel()[f].sum(1)>=2
    uv=texture_uv(cam.project(pos),s['size'])
    if 'water' in s:
        wetpoly=cam.ground(s['water'])
        # Intersect the source rays with the water plane before locating its physical perimeter.
        wetpoly=cam.eye[:2]+(wetpoly-cam.eye[:2])*(1-s['water_z_m']/cam.eye[2])
        wet=poly_mask(xy,wetpoly)
        # Grid exterior follows the original photographed rim. Keep submerged terrain below the interface.
        z[wet]=np.minimum(z[wet],s['water_z_m']-.006)
        pos=xyz(xy,z);uv=texture_uv(cam.project(pos),s['size'])
        # Concealed bottom uses observed dry aggregate. Remove the sky and branches from its colour map.
        wmask=np.zeros(img.shape[:2],'u1');cv2.fillPoly(wmask,[np.int32(s['water'])],1)
        yy,xx=np.indices(wmask.shape)
        donor=img[278:298,435:502]
        floor=img.copy();floor[wmask>0]=(donor[yy%20,xx%67][wmask>0]*.57).astype('u1')
    else:
        wet=np.zeros_like(mask);floor=img
    mat=mesh.material('Road surface from source photograph',floor,.91,0,.55,lossless_color=True)
    # Keep only a local photographed shoulder. Outside the camera raster the sampler
    # would otherwise repeat a roadside kerb or grass across the approach pavement.
    pad = distance_transform_edt(~mask)*pitch < .10
    observed = ((uv>=0)&(uv<=1)).all(axis=-1)
    shoulder = (~defect) & (pad&observed).ravel()[f].all(axis=1)
    mesh.add('Road adjoining the fracture',pos,f[shoulder],uv,mat,True)
    mesh.add('Fracture lip and exposed bed',pos,f[defect],uv,mat)
    if wet.any():
        # Split reflected radiance from albedo. A PBR viewer supplies the environment at the new view.
        rgb=np.zeros((16,16,3),'u1')+[37,42,38]
        water=mesh.material('Standing water, environment reflection',rgb,.075,0,0)
        mesh.ior[water]=1.333
        surface=xyz(xy,s['water_z_m']);wf=wet.ravel()[f].sum(1)>=2
        mesh.add('Water surface',surface,f[wf],texture_uv(cam.project(surface),s['size']),water)
    return np.ptp(xy,axis=(0,1)),dict(nominal_major_m=s['major_m'],nominal_depth_m=s['depth_m'],
             source_camera_eye_m=cam.eye.tolist(),photographed_extent_m=np.ptp(poly,axis=0).tolist(),
             water_level_m=s.get('water_z_m'),source_rim_xy_m=poly.tolist())
