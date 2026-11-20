"""Fit an inspection camera to complete frame/lid geometry at physical scale."""
from itertools import product
import numpy as np


def unit(value):
    return value/max(float(np.linalg.norm(value)),1e-12)


def structural_support(mesh):
    with np.load(mesh) as data:
        names=data['part_names'].astype(str)
        parts=np.array([name.startswith('Lid') or name=='Cast ring and cover seat' for name in names])
        faces=data['faces'][parts[data['face_part']]]
        if not len(faces):raise ValueError('No explicit frame/lid geometry')
        points=data['xyz'][np.unique(faces)]
    # Include a short look into the aperture; do not fit a 1m shaft below an
    # opaque road into an above-ground inspection shot.
    lo=points.min(0);hi=points.max(0);lo[2]=min(lo[2],-.10)
    padding=.015
    corners=np.array(list(product(*zip(lo-padding,hi+padding))))
    return points,corners


def fit(center,basis,corners,angle,width=960,height=540,focal=873.6,margin=28):
    world=corners@basis.T+center;target=(world.min(0)+world.max(0))*.5
    away=unit(basis@np.array([np.sin(angle),-np.cos(angle),.896]))
    forward=-away;right=unit(np.cross(forward,[0.,0.,1.]))
    orientation=np.stack([right,np.cross(forward,right),forward],axis=1)
    q=(world-target)@orientation
    distance=max(.075,float(np.max(abs(q[:,0])*focal/(width*.5-margin)-q[:,2])),
                 float(np.max(abs(q[:,1])*focal/(height*.5-margin)-q[:,2])),float(.05-q[:,2].min()))
    return target+away*distance,target


def projection(points,position,target,width=960,height=540,focal=873.6):
    forward=unit(target-position);right=unit(np.cross(forward,[0.,0.,1.]))
    rotation=np.stack([right,np.cross(forward,right),forward],axis=1)
    q=(points-position)@rotation
    uv=q[:,:2]*focal/q[:,2:3]+[width*.5,height*.5]
    margin=np.minimum.reduce([uv[:,0],width-uv[:,0],uv[:,1],height-uv[:,1]])
    return q[:,2],margin
