"""Compare the native shaft mesh to a separately traced source silhouette and two Gaussian views."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from calibration import Camera
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lane-damage'))
from inspect_assets import inspect
from mesh_transfer import RoadMesh
from importlib import import_module
render=import_module('check-verge-chamber').render


def check(models,out):
    s=json.loads((HERE/'urayasu.json').read_text())
    def evidence(folder,surface,meta,spec,run):
        cal=meta['profile_calibration'];cam=Camera(s,cal['parameters'])
        scale=cal['metres_per_unit'];height=cal['profile']['cap']['height_above_road_m']
        eye=cam.eye*scale+[0,0,height]
        w,h=s['size'];y,x=np.indices((h,w))
        rays=cam.rays(np.stack([x,y],axis=-1));rgb,part,_=render(RoadMesh(folder),eye,rays)
        truth=np.zeros((h,w),'u1');cv2.fillPoly(truth,[np.int32(s['silhouette_px'])],1)
        found=part>1;truth=truth.astype(bool);iou=float((found&truth).sum()/(found|truth).sum())
        Image.fromarray(rgb).save(out/'source-view.jpg',quality=94)
        center=np.array([[0.,0.,1.]]);d,p,_=surface.cast(center,np.array([[0.,0.,-1.]]))
        assert p[0]==9 and abs(1.-d[0]-height)<.0001
        assert iou>.93,iou
        tip,foot=np.array(cal['pen_3d_endpoints_model_units'])
        # Sensitivity to where the pen touches the lid and the flange. The stated
        # perturbation is an illustrative displacement, not a statistical error bar.
        pen=np.asarray(s['pen']);param=np.asarray(cal['parameters'])
        heights=[]
        for dz_top in [-.005,0.,.005]:
            for dz_foot in [-.005,0.,.005]:
                top=cam.plane(pen[:1],dz_top/scale)[0]
                base=cam.plane(pen[1:],param[6]+dz_foot/scale)[0]
                heights.append(-param[10]*s['pen_m']/np.linalg.norm(top-base))
        return dict(source_silhouette_iou=iou,source_silhouette_pixels=int(truth.sum()),
            height_when_pen_contact_shifted_5mm_m=[float(min(heights)),float(max(heights))],
            reserved_ring_rms_px=cal['reserved_rms_px'],reserved_ring_max_px=cal['reserved_max_px'],
            cap_height_above_road_model_m=float(1.-d[0]),pen_constraint_m=float(np.linalg.norm(tip-foot)*scale))
    out.mkdir(parents=True,exist_ok=True)
    inspect(models,out,evidence,fit_tall=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('out',type=Path)
    check(**vars(p.parse_args()))
