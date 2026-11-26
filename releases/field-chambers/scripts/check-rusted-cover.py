"""Trace every photographed corrosion opening through the exported steel and its sealed control."""
import argparse
from importlib import import_module
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from road_geometry import Surface, glb_triangles, read_json, save_json, sha256

topology=import_module("check-perforated-covers").topology
BASE=Path(__file__).resolve().parents[1]/"gen2hole/examples/rusted-cover"


def check(native, control, photo, dest, topology_check=True):
    rec=read_json(BASE/"survey.json");dest.mkdir(exist_ok=True,parents=True)
    assert sha256(photo)==rec["sha256"]
    metal=[p for p in glb_triangles(native) if p[0] in (163,164,10,11)]
    solid=topology(metal) if topology_check else {}
    raw=np.asarray(Image.open(photo).convert("L"))
    height,width=raw.shape
    l,t,r,b=rec["crop"]
    u,v=np.meshgrid(np.arange(l,r+1),np.arange(t,b+1))
    scale=rec["plate_width_m"]/(r-l)
    xy=np.stack([(u-(r+l)/2)*scale,((b+t)/2-v)*scale],axis=-1)
    orig=np.dstack([xy,np.ones(u.shape)*.04]);direction=np.broadcast_to([0.,0.,-1.],orig.shape)
    mesh=Surface(native);sealed=Surface(control)
    dist,part,_=mesh.cast(orig,direction)
    ds,ps,_=sealed.cast(orig,direction)
    hole=(part==7)|(part==6)|(part==16)
    hole[[0,-1],:]=False
    hole[:,[0,-1]]=False
    assert not (ps[2:-2,2:-2]!=163).any()
    factor=np.array([width,height])/rec["reference_size"]
    regions=[];thresholds={}
    for threshold in (22,30,38):
        # Segment at original camera resolution; the builder works on the 1200-pixel copy.
        _,labels=cv2.connectedComponents((raw<threshold).astype("u1"))
        ids=[int(labels[tuple(np.rint((np.asarray(seed["seed_uv"])+.5)*factor-.5).astype(int)[::-1])]) for seed in rec["seeds"]]
        assert min(ids)>0 and len(set(ids))==len(ids)
        idx=np.zeros(labels.max()+1,'u2');idx[ids]=np.arange(1,len(ids)+1)
        rr=np.rint((v+.5)*factor[1]-.5).astype(int);cc=np.rint((u+.5)*factor[0]-.5).astype(int)
        source=idx[labels[rr,cc]]
        correct=binary_fill_holes(source>0)
        thresholds[str(threshold)]=float((correct&hole).sum()/(correct|hole).sum())
        if threshold==30:
            reference=source.copy()
            for i,(x,y) in enumerate([s["seed_uv"] for s in rec["seeds"]],1):
                mask=binary_fill_holes(source==i)
                recall=float(hole[mask].mean())
                # The annulus lies beyond the antialiased metal edge of every aperture.
                ring=(distance_transform_edt(~mask)<=4)&(distance_transform_edt(~mask)>=2)&~correct
                regions.append(dict(id=i,seed_uv=[x,y],source_pixels=int(mask.sum()),recall=recall,
                                    metal_ring_agreement=float((part[ring]==163).mean()),
                                    center_first_part=int(part[y-t,x-l]),center_clearance_mm=float((dist-ds)[y-t,x-l]*1000)))
    np.savez_compressed(dest/"overhead.npz",source_id=reference,mesh_part=part,open_range_m=dist,sealed_range_m=ds)
    if not all(s["center_first_part"]==7 for s in regions):
        raise ValueError("A photographed seed no longer opens to the chamber")
    report=dict(glb_sha256=sha256(native),control_sha256=sha256(control),photo_sha256=sha256(photo),
                topology=solid,source_threshold_iou=thresholds,holes=regions,packet_sha256=sha256(dest/"overhead.npz"))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("native","control","photo","dest","report"):p.add_argument(arg,type=Path)
    a=p.parse_args()
    save_json(a.report,check(a.native,a.control,a.photo,a.dest))
