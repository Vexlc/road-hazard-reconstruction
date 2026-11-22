"""Sun direction from recorded time/GPS, checked against source-road pole shadows."""
import argparse
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_scans import camera_track
from road_geometry import SITES,load_inventory,read_json,save_json

NOAA='https://gml.noaa.gov/grad/solcalc/solareqns.PDF'


def sun_enu(lat,lon,timestamp):
    now=datetime.fromtimestamp(timestamp,timezone.utc)
    hour=now.hour+now.minute/60+(now.second+now.microsecond/1e6)/3600
    a=(2*np.pi/365)*(now.timetuple().tm_yday-1+(hour-12)/24)
    e=229.18*(.000075+.001868*np.cos(a)-.032077*np.sin(a)-.014615*np.cos(2*a)-.040849*np.sin(2*a))
    d=(.006918-.399912*np.cos(a)+.070257*np.sin(a)-.006758*np.cos(2*a)+.000907*np.sin(2*a)
       -.002697*np.cos(3*a)+.001480*np.sin(3*a))
    ha=np.deg2rad(((hour*60+e+4*lon)%1440)/4-180);la=np.deg2rad(lat)
    return np.array([-np.cos(d)*np.sin(ha),np.sin(d)*np.cos(la)-np.cos(d)*np.cos(ha)*np.sin(la),
                     np.sin(d)*np.sin(la)+np.cos(d)*np.cos(ha)*np.cos(la)])


# Regions of a cast pole shadow and the immediately adjoining, unmarked pavement.
# Coordinates use half-resolution inspection images and are scaled before reading.
PATCHES={'front_camera':('110',24,[(90,476,300,512),(345,480,440,514)]),
         'back_camera':('109',42,[(347,327,413,369),(412,363,451,390)])}
TRACES=[('front_camera',37,[(891,405),(764,382)]),('back_camera',42,[(338,323),(422,373)])]


def measure(raw,parent,out):
    _,_,anchors=load_inventory(parent);lights={}
    for camera,(site,frame,patches) in PATCHES.items():
        with Image.open(raw/site/'camera'/camera/f'{frame:02d}.jpg') as im:rgb=np.asarray(im)/255.
        ratios=[]
        for rect in patches:
            l,t,r,b=np.array(rect)*2;v=rgb[t:b,l:r].reshape(-1,3);lum=v@np.array([.2126,.7152,.0722])
            q=np.quantile(lum,[.04,.13,.62,.82])
            dark=np.median(v[(lum>=q[0])&(lum<=q[1])],axis=0)
            lit=np.median(v[(lum>=q[2])&(lum<=q[3])],axis=0)
            ratios.append((dark/lit).tolist())
        lights[camera]=dict(measured_shade_to_sun_srgb=np.median(ratios,axis=0).tolist(),samples=ratios,
                            site=site,frame=frame,rects_at_half_resolution=patches)
    details={}
    for site in SITES:
        src=raw/site;gps=read_json(src/'meta/gps.json');stamps=read_json(src/'meta/timestamps.json')
        lat=np.asarray([g['lat'] for g in gps]);lon=np.asarray([g['long'] for g in gps])
        pos=np.column_stack(((lon-lon.mean())*np.cos(np.deg2rad(lat.mean()))*111320,(lat-lat.mean())*111132))
        native=np.array([[p['position']['x'],p['position']['y']] for p in read_json(src/'lidar/poses.json')]);native-=native.mean(axis=0)
        U,_,V=np.linalg.svd(pos.T@native);R=U@V
        if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
        residual=np.linalg.norm(pos@R-native,axis=1)
        sun=sun_enu(lat.mean(),lon.mean(),np.median(stamps))
        world=np.r_[sun[:2]@R,sun[2]]
        details[site]=dict(sun_world=world.tolist(),sun_enu=sun.tolist(),sun_elevation_deg=float(np.rad2deg(np.arcsin(sun[2]))),
            gps_to_world_yaw_deg=float(np.rad2deg(np.arctan2(R[0,1],R[0,0]))),gps_track_fit_rms_m=float(np.sqrt(np.mean(residual**2))),
            direct_light_used=site in ('109','110'))
    center,B=anchors[('109','05')]
    traces=[]
    for camera,frame,points in TRACES:
        _,poses=camera_track(raw/'109',camera);pose=poses[frame]
        k=read_json(raw/'109'/'camera'/camera/'intrinsics.json')
        xy=np.array(points)*2;v=np.column_stack(((xy[:,0]-k['cx'])/k['fx'],(xy[:,1]-k['cy'])/k['fy'],np.ones(2)))@pose[:3,:3].T
        d=(center-pose[:3,3])@B[:,2]/(v@B[:,2]);g=pose[:3,3]+v*d[:,None]
        observed=(g[1]-g[0])@B
        expected=-np.array(details['109']['sun_world'])@B
        angle=np.rad2deg(np.arccos(np.clip(observed[:2]@expected[:2]/(np.linalg.norm(observed[:2])*np.linalg.norm(expected[:2])),-1,1)))
        traces.append(dict(site='109',camera=camera,frame=frame,line_at_half_resolution=points,shadow_azimuth_difference_deg=float(angle)))
    save_json(out/'tools/daylight.json',dict(sites=details,contrast=lights,checked_pole_shadows=traces,solar_formula=NOAA))


@lru_cache(maxsize=1)
def lighting():
    return read_json(Path(__file__).with_name('daylight.json'))


def cast_shadows(site,camera,models,pose,K,rect,part,which,src,protected):
    shape=part.shape;coverage=np.zeros(shape,'f4')
    info=lighting();config=info['sites'][site]
    if not config['direct_light_used']:return coverage,np.ones(3,'f4')
    factor=np.asarray(info['contrast'][camera]['measured_shade_to_sun_srgb'],dtype='f4').clip(.42,.88)
    l,t,r,b=rect
    for i,(p,mesh,_) in enumerate(models,1):
        if p['category']!='raised' and not (13 in mesh.parts and mesh.high[2]>.0035):continue
        keep=(part==1)&(which==i)&~protected
        y,x=np.where(keep)
        if not len(y):continue
        B=np.array(p['basis_columns']);c=np.asarray(p['center_world'])
        sun=np.asarray(config['sun_world'])@B
        if sun[2]<.15:continue
        eye=(pose[:3,3]-c)@B
        camera_rays=np.column_stack(((x+l+.5-K[0,2])/K[0,0],(y+t+.5-K[1,2])/K[1,1],np.ones(len(y))))
        ray=(camera_rays@pose[:3,:3].T)@B
        above=ray[:,2]<-1e-5
        if not above.any():continue
        y,x,ray=y[above],x[above],ray[above]
        ground=eye+ray*((.0035-eye[2])/ray[:,2])[:,None]
        distance,hit,_=mesh.cast(ground,np.broadcast_to(sun,ground.shape))
        shade=np.zeros(shape,'f4')
        shade[y,x]=((hit>=9)&(hit<=15)&(distance>0)&(distance<3.)).astype('f4')
        if not shade.any():continue
        soft=cv2.GaussianBlur(shade,(0,0),1.15)
        # Suppress additional darkening on strongly pre-shadowed asphalt; paint can receive light.
        lum=src@np.array([.2126,.7152,.0722]);reference=np.median(lum[keep])
        luminance=cv2.GaussianBlur(lum.astype('f4'),(0,0),2.)
        confidence=np.clip((luminance/max(reference,1)-.49)/.37,0,1)
        coverage=np.maximum(coverage,(soft*keep*confidence).astype('f4'))
    return coverage,factor


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['raw','parent','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();measure(a.raw,a.parent,a.out)
