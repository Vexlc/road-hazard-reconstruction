"""Reconstruct the photographed patch of a corroded chequer-plate cover with open bores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, map_coordinates

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[1]
sys.path[:0] = [str(GEN), str(GEN/"recipes/common"), str(GEN/"recipes/series"), str(HERE.parent/"road-distress"), str(HERE.parent/"parted-inlets")]
from build_chamber import flatten_scene
from export_splats import make_one
from mesh_io import Mesh, grid_faces, write_json
from refine import boundary_mesh, sample
from edge_kernels import align_edges

TOP = "Photographed corroded chequer plate"
EDGE = "Rust perforation cut edge"
COMMON = HERE.parent/"perforated-covers"


def photo_fields(photo, spec):
    rgb = np.asarray(photo.resize(spec["reference_size"],Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Segment before downsampling. Averaging grey values first erodes the smallest pores.
    fine_size=tuple(2*np.asarray(spec["reference_size"]))
    fine=np.asarray(photo.convert("L").resize(fine_size,Image.Resampling.LANCZOS))
    _, labels = cv2.connectedComponents((fine<30).astype("u1"))
    ids = [labels[2*y+1,2*x+1] for x,y in (s["seed_uv"] for s in spec["seeds"])]
    if 0 in ids or len(set(ids)) != len(ids):
        raise ValueError("A documented perforation no longer has a unique silhouette")
    coverage=cv2.resize(np.isin(labels,ids).astype("f4"),tuple(spec["reference_size"]),interpolation=cv2.INTER_AREA)
    mask = binary_fill_holes(coverage>=.5)
    # A dark folded lip is attached on the left. An isolated bright thread
    # inside another void gives no evidence of a load-bearing metal bridge.
    mask = mask.astype("u1")
    for key,value in (("connected_metal_uv",0),("unresolved_filament_uv",1)):
        cv2.fillPoly(mask,[np.int32(spec["edge_interpretation"][key])],value)
    mask = mask.astype(bool)
    signed = (distance_transform_edt(~mask)-distance_transform_edt(mask)).astype("f4")
    near = distance_transform_edt(mask,return_distances=False,return_indices=True)
    extended = gray[tuple(near)].astype("f4")
    # Along-rib averages suppress isotropic rust mottling. Compare each response to
    # both flanks so colour changes across a broad patch do not imply raised metal.
    yy,xx = np.indices(gray.shape,dtype="f4")
    responses=[]
    for angle in np.deg2rad(spec["tread_angles_deg"]):
        normal=np.array([-np.sin(angle),np.cos(angle)])
        kernel=np.zeros((35,35),"f4")
        off=np.rint(16*np.array([np.cos(angle),np.sin(angle)])).astype(int)
        cv2.line(kernel,tuple(17-off),tuple(17+off),1.,3,cv2.LINE_AA)
        along=cv2.filter2D(extended,-1,kernel/kernel.sum())
        flanks=[cv2.remap(along,xx+float(t*normal[0]),yy+float(t*normal[1]),cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT) for t in (-9,9)]
        responses.append((along-.5*(flanks[0]+flanks[1])-1.)/14.)
    tread=cv2.GaussianBlur(np.maximum(*responses).clip(0,1),(0,0),1.2)
    # The final one to two pixels near the void may include defocus from its background.
    # Extend only into the aperture. The intact photographed colour is retained elsewhere.
    l,t,r,b=spec["crop"]
    crop=np.s_[t:b+1,l:r+1]
    return rgb[crop],signed[crop],tread[crop]


def clipped_plate(rgb,signed,tread,rec,pitch,sealed):
    h,w=signed.shape
    width=rec["plate_width_m"]
    length=width*(h-1)/(w-1)
    rows,cols=np.ceil(np.array([length,width])/pitch).astype(int)+1
    yy,xx=np.meshgrid(np.linspace(0,h-1,rows),np.linspace(0,w-1,cols),indexing="ij")
    uv=np.c_[xx.ravel(),yy.ravel()]
    faces=grid_faces(rows,cols)
    seeds=np.array([s["seed_uv"] for s in rec["seeds"]])-rec["crop"][:2]
    uv,faces,level=boundary_mesh(uv,faces,signed,seeds,2)
    vertices=np.column_stack(((uv[:,0]/(w-1)-.5)*width,(.5-uv[:,1]/(h-1))*length,
                              sample(tread,uv)*rec["tread_relief_m"]))
    if sealed:
        return vertices,faces,np.empty((0,2),int),np.array([width,length])
    positive=level[faces]>=0
    counts=positive.sum(1)
    result=[faces[counts==3]]
    new_vertices=[vertices]
    edges=[];cursor=len(vertices)
    for inside in (1,2):
        f=faces[counts==inside]
        first=(positive[counts==inside] if inside==1 else ~positive[counts==inside]).argmax(1)
        a,b,c=np.take_along_axis(f,(first[:,None]+np.arange(3))%3,axis=1).T
        def intersect(i,j):
            i,j=np.minimum(i,j),np.maximum(i,j)
            t=level[i]/(level[i]-level[j])
            return vertices[i]+t[:,None]*(vertices[j]-vertices[i])
        new_vertices.extend([intersect(a,b),intersect(a,c)])
        e=cursor+np.arange(len(f));g=e+len(f);cursor+=2*len(f)
        if inside==1:
            result.append(np.c_[a,e,g]);edges.append(np.c_[e,g])
        else:
            result.extend([np.c_[e,b,c],np.c_[e,c,g]]);edges.append(np.c_[g,e])
    return np.vstack(new_vertices),np.vstack(result),np.vstack(edges),np.array([width,length])


def build(source,run,spec,*,sealed=False,pitch=.00155,mesh_only=False):
    path=source/spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec["sha256"]:
        raise ValueError("Source image checksum differs")
    rgb,signed,tread=photo_fields(Image.open(path).convert("RGB"),spec)
    v,faces,edges,size=clipped_plate(rgb,signed,tread,spec,pitch,sealed)
    mesh=Mesh()
    iron=mesh.material("Rust and raised tread from the original photograph",rgb,.87,.24,0,lossless_color=True)
    around=np.asarray(Image.open(COMMON/"split-iron/pavement.jpg").convert("RGB"))
    road=mesh.material("Asphalt031 road sample",around,.94,0,.8)
    near=distance_transform_edt(signed<2,return_distances=False,return_indices=True)
    cut=mesh.material("Dark rust at the broken steel edge",np.uint8(rgb[tuple(near)]*.66),.94,.16,0)
    liner=np.asarray(Image.open(COMMON/"lining.jpg").convert("RGB"))
    dark=mesh.material("Recess lining beneath shaded plate",np.uint8(liner*.10+4),.97,0,.4)
    frame=mesh.material("Oxidised steel mounting rebate",rgb[22:103,155:235],.86,.20,.2)
    uv=lambda q:.5+q[:,:2]/size
    mesh.add(TOP,v,faces,uv(v),iron)
    bottom=v.copy();bottom[:,2]=-spec["plate_thickness_m"]
    mesh.add("Lid underside",bottom,faces[:,::-1],uv(bottom),cut)
    def edges_to_wall(edges,name):
        a,b=edges.T
        ring=np.stack([v[b],v[a],bottom[a],bottom[b]],1).reshape(-1,3)
        ix=np.arange(len(a))[:,None]*4
        wall_faces=np.vstack([ix+[0,1,2],ix+[0,2,3]])
        mesh.add(name,ring,wall_faces,uv(ring),cut)
    if len(edges):
        edges_to_wall(edges,EDGE)
    def rect(half,z):
        return np.c_[np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half,np.full(4,z)]
    def band(name,outer,inner,material,isroad=False):
        # Each boundary runs counterclockwise viewed from above.
        p=np.vstack([outer,inner])
        a=np.arange(4);b=np.roll(a,-1)
        f=np.vstack([np.c_[a,b,b+4],np.c_[a,b+4,a+4]])
        mesh.add(name,p,f,.5+p[:,:2]/span,material,isroad)
    half=size/2
    span=size+.20
    band("Road around the aperture",rect(span/2,0),rect(half+.014,0),road,True)
    band("Cast ring and cover seat",rect(half+.014,0),rect(half+.0015,-.001),frame)
    rows,cols=np.ceil(size[::-1]/pitch).astype(int)+1
    grid=np.arange(rows*cols).reshape(rows,cols)
    boundary=np.r_[grid[0,::-1],grid[1:,0],grid[-1,1:],grid[-2:0:-1,-1]]
    edges_to_wall(np.c_[boundary,np.roll(boundary,-1)],"Lid rolled edge")
    band("Subgrade bearing",rect(half+.0015,-.001),rect(half-.004,-.010),frame)
    depth=spec["shaft_depth_m"]
    band("Inspection shaft",rect(half-.004,-.010),rect(half-.004,-depth),dark)
    p=rect(half-.004,-depth)
    mesh.add("Shaft bottom",p,[[0,1,2],[0,2,3]],.5+p[:,:2]/size,dark)
    y,x=np.meshgrid(np.linspace(span[1]/2,-span[1]/2,385),np.linspace(-span[0]/2,span[0]/2,513),indexing="ij")
    rr=(.5-y/size[1])*(rgb.shape[0]-1);cc=(.5+x/size[0])*(rgb.shape[1]-1)
    lid=(abs(x)<=half[0])&(abs(y)<=half[1])
    hole=lid&(map_coordinates(signed,[rr,cc],order=1,mode="nearest")<0)&(not sealed)
    z=np.where(hole,-depth,np.where(lid,map_coordinates(tread,[rr,cc],order=1,mode="nearest")*spec["tread_relief_m"],0.))
    installed=(abs(x)<=half[0]+.014)&(abs(y)<=half[1]+.014)
    field=dict(width=span[0],length=span[1],height=z,alpha=installed.astype(float),opening_mask=hole,
               defect_mask=installed,installation_mask=installed)
    key=spec["id"];series=run/"series";dest=series/"assets/missing"/key
    refs=series/"references";refs.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(path,refs/(key+".jpg"))
    meta=dict(id=key,category="missing",family="rusted-cover",label=spec["label"],state="sealed-control" if sealed else "corroded",
              source=spec["page"],source_author=spec["author"],source_license=spec["license"],source_sha256=spec["sha256"],
              source_image=f"references/{key}.jpg",seed=18544,surface_dimensions_m=span.tolist(),plate_dimensions_m=size.tolist(),
              depth_p95_m=depth,scale_basis="Single cropped photograph; plate extent, thickness, ridge height and shaft are inferred.")
    mesh.save(dest,series/"mesh_cache",meta,field)
    flatten_scene(dest/"asset.glb")
    row=json.loads((dest/"metadata.json").read_text())
    if not mesh_only:
        row=make_one(dest,series/"mesh_cache",part_importance={TOP:16,EDGE:25,"Shaft bottom":1.6,"Lid underside":.25},
                     point_limit=980000,kernel_factors=(.79,.14))
        row["edge_aligned_splats"]=align_edges(dest,(163,164,10),falloff=2.8,minimum_ratio=.12)
        write_json(dest/"metadata.json",row)
    for role,name in [("glb","asset.glb"),("metadata","metadata.json"),("surface","surface.npz")]+([] if mesh_only else [("ply","asset.ply")]):
        row[role]=f"missing/{key}/{name}"
    write_json(series/"assets/asset_manifest.json",[row])
    write_json(run/"run.json",dict(editions=["series"],assets=[key]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","run"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--sealed",action="store_true")
    parser.add_argument("--mesh-only",action="store_true")
    parser.add_argument("--pitch",type=float,default=.00155)
    a=parser.parse_args()
    build(a.source,a.run,json.loads((HERE/"survey.json").read_text()),sealed=a.sealed,pitch=a.pitch,mesh_only=a.mesh_only)
