"""Metric geometry and image projection helpers used by the field-photo assets."""
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / 'releases/field-chambers/gen2hole'
sys.path[:0] = [str(LIB/'recipes/series'),str(LIB/'recipes/common'),str(LIB/'examples/road-distress'),
                str(LIB/'examples/verge-chamber'),str(LIB/'downstream/sensors'),str(LIB)]
from mesh_io import Mesh, grid_faces, write_json
from surfaces import rings_faces, refine
from build_chamber import flatten_scene
from road_geometry import Surface
from export_splats import make_one


def normalize(a):
    return a / np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1.e-12)


class RoadCamera:
    def __init__(self,s):
        self.cx,self.cy=np.asarray(s['size'])/2
        self.f=s['focal']
        t=np.arctan((self.cy-s['horizon'])/self.f)
        self.R=np.array([[1.,0.,0.],[0.,-np.sin(t),-np.cos(t)],[0.,np.cos(t),-np.sin(t)]])
        self.eye=np.array([0.,0.,1.])
        outline=self.ground(s['rim'])
        rect=cv2.minAreaRect(outline.astype('f4'))
        self.eye[2]=s['major_m']/max(rect[1])
        xy=self.ground(s['rim'])
        self.eye[:2]=-((xy.min(0)+xy.max(0))/2)

    def ground(self,pixels):
        q=np.asarray(pixels,float)
        r=np.concatenate([(q-[self.cx,self.cy])/self.f,np.ones(q.shape[:-1]+(1,))],axis=-1) @ self.R
        return self.eye[:2]-self.eye[2]*r[...,:2]/r[...,2:]
    def project(self,xyz):
        d=(xyz-self.eye)@self.R.T
        return d[...,:2]/d[...,2:]*self.f+[self.cx,self.cy]


def texture_uv(pixels,size):
    return ((np.asarray(pixels)+.5)/size)*[1,-1]+[0,1]


def grid(span,pitch):
    half=np.ceil(np.asarray(span)/2/pitch).astype(int)
    x=np.arange(-half[0],half[0]+1)*pitch
    y=np.arange(half[1],-half[1]-1,-1)*pitch
    return np.stack(np.meshgrid(x,y),-1)


def poly_mask(xy,poly):
    low=xy[0,0];pitch=xy[0,1,0]-low[0]
    rc=(np.array(poly)-low)/[pitch,-pitch]
    m=np.zeros(xy.shape[:2],'u1');cv2.fillPoly(m,[np.int32(np.round(rc*16))],1,shift=4)
    return m.astype(bool)


def face_up(v,f):
    a,b,c=v[f].transpose(1,0,2);flip=np.cross(b-a,c-a)[:,2]<0
    f=f.copy();f[flip]=f[flip,::-1]
    return f


def face_polygon(mesh,name,outer,holes,mat,uvfun):
    rings=[outer,*holes];v=np.vstack(rings)
    f=face_up(v,rings_faces([q[:,:2] for q in rings]))
    v,f=refine(v,f,.08)
    mesh.add(name,v,f,uvfun(v),mat,name.startswith('Road '))


def xyz(xy,z):
    xy=np.asarray(xy);return np.concatenate([xy,np.broadcast_to(z,xy.shape[:-1])[...,None]],axis=-1)


def rect(span,z=0):
    return xyz(np.array([[-1,1],[1,1],[1,-1],[-1,-1]])*np.array(span)/2,z)


def remap(img,pix):
    return cv2.remap(img,pix[...,0].astype('f4'),pix[...,1].astype('f4'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
