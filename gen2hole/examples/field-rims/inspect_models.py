"""Measure withheld points, first-hit regions of the broken lid, and yellow-line disagreement."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lane-damage'))
from inspect_assets import inspect, scan_fit
from photo_mesh import RoadCamera, xyz
from road_geometry import read_json


def observations(folder,surf,meta,unused,run):
    key=folder.name
    if (run/'evaluation'/key/'survey.npz').is_file():
        return scan_fit(folder,surf,meta,run)
    if key=='P562':
        s=read_json(HERE/'riverview.json');cam=RoadCamera(s)
        w,h=s['size'];y,x=np.indices((h,w));pixel=np.stack([x,y],axis=-1)
        direction=xyz((pixel-[cam.cx,cam.cy])/cam.f,1.)@cam.R
        _,part,_=surf.cast(np.broadcast_to(cam.eye,direction.shape),direction)
        wet=np.zeros((h,w),'u1');cv2.fillPoly(wet,[np.int32(s['water'])],1)
        include=(part==4)|(part==44);truth=wet.astype(bool)
        iou=float((include&truth).sum()/(include|truth).sum())
        probes=[np.int32(np.mean(q,axis=0)) for q in s['stones']]
        stone_hits=[int(part[y,x]) for x,y in probes]
        wx,wy=s['water_witness'];assert part[wy,wx]==4 and stone_hits==[44]*5 and iou>.91,(stone_hits,iou)
        with np.load(folder/'surface.npz') as field:
            mask=field['water_mask'].astype(bool)
            depth=field['height'][mask]-field['substrate_height'][mask]
            assert np.median(depth)>.045
        return dict(source_water_and_stone_iou=iou,stone_center_part_ids=stone_hits,
                    water_depth_model_median_m=float(np.median(depth)))
    s=read_json(HERE/'covers.json')[key]
    w,h=s['size'];ys,xs=np.indices((h,w))
    pix=np.stack([xs,ys],axis=-1).astype('f8')
    if key=='A560':
        xy=(pix-meta['pixel_center'])*meta['pixel_to_local_scale']
    else:
        xy=cv2.perspectiveTransform(pix,np.array(meta['pixel_to_local_homography']))
    start=xyz(xy,1.0);dist,part,_=surf.cast(start,np.broadcast_to([0.,0.,-1.],start.shape))
    if key=='A560':
        img=np.asarray(Image.open(HERE/'references/A560.jpg').convert('RGB'))
        hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV);hue,sat,val=hsv.transpose(2,0,1)
        yellow=(hue>=13)&(hue<=37)&(sat>51)&(val>77)
        roi=np.zeros((h,w),'u1');cv2.fillPoly(roi,[np.int32(s['visible_road'])],1)
        road=(np.linalg.norm(xy,axis=2)>.352)&(roi>0)
        true=yellow&road;found=(part==114)&road
        iou=(true&found).sum()/(true|found).sum()
        assert iou>.92
        # Locate the centre of the same continuous stripe across independent scanlines.
        # Exclude the circular border: its tangent changes with azimuth.
        tilts={}
        for name,mask in [('lid',yellow&(part==9)),('road',found)]:
            p,q=np.asarray(s['angle_samples'][name],float)
            vertical=name=='lid'
            if vertical:
                p=p[::-1];q=q[::-1];mask=mask.T
            cols=(range(145,307,3) if vertical else list(range(15,120,3))+list(range(523,622,3)))
            samples=[]
            for x in cols:
                y=p[1]+(q[1]-p[1])*(x-p[0])/(q[0]-p[0])
                hit=np.flatnonzero(mask[:,x]);hit=hit[abs(hit-y)<30]
                if len(hit)>=12:
                    samples.append([x,np.median(hit)])
            samples=np.array(samples)
            assert len(samples)>28
            from scipy.stats import theilslopes
            slope=theilslopes(samples[:,1],samples[:,0])[0]
            angle=np.arctan2(1,slope) if vertical else np.arctan(slope)
            tilts[name]=float((np.rad2deg(angle)+90)%180-90)
        d=abs((tilts['lid']-tilts['road']+90)%180-90)
        assert 39<d<70 and part[232,319]==9, (d,tilts,int(part[232,319]))
        return dict(yellow_road_mask_iou=float(iou),direction_image_deg=tilts,angle_disagreement_image_deg=d,
                    center_first_hit_m=float(1.-dist[232,319]))
    ox,oy=s['opening_witness'];assert part[oy,ox]==7
    witness=np.zeros((h,w),'u1');cv2.fillPoly(witness,[np.int32(s['source_void'])],1)
    source=witness.astype(bool);inside=part[source];fraction=float(np.mean(np.isin(inside,[6,7])))
    assert fraction>.98
    tests=[]
    for mask,z in zip(s['pieces'],s['fragment_z_m']):
        stencil=np.zeros((h,w),'u1');cv2.fillPoly(stencil,[np.int32(mask)],1)
        core=cv2.erode(stencil,np.ones((7,7),'u1')).astype(bool)
        fraction_hit=float(np.mean(part[core]==9))
        assert fraction_hit>.995 and np.allclose(np.median(1.-dist[core]),z,atol=.0001)
        tests.append(dict(core_pixels=int(core.sum()),lid_hit_fraction=fraction_hit,median_top_m=float(np.median(1.-dist[core]))))
    return dict(visible_void_witness_pixels=int(source.sum()),opening_hit_fraction=fraction,
                center_void_floor_depth_m=float(dist[oy,ox]-1.),fragment_checks=tests)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();inspect(a.models,a.out,observations)
