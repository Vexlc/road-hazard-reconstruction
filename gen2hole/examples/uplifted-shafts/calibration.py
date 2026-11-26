"""Fit concentric horizontal ring projections, then recover scale from a photographed reference pen."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent


class Camera:
    def __init__(self, spec, params):
        pitch,roll=params[:2]
        c,s=np.cos(pitch),np.sin(pitch)
        base=np.array([[1.,0.,0.],[0.,-s,-c],[0.,c,-s]])
        cr,sr=np.cos(roll),np.sin(roll)
        self.R=np.array([[cr,-sr,0.],[sr,cr,0.],[0.,0.,1.]])@base
        self.t=np.asarray(params[2:5])
        self.center=np.asarray(spec['size'])/2
        self.f=spec['focal_35mm']*np.linalg.norm(self.center*2)/np.hypot(36.,24.)
        self.eye=-self.t@self.R

    def rays(self,p):
        p=np.asarray(p,float)
        return np.concatenate([(p-self.center)/self.f,np.ones((*p.shape[:-1],1))],axis=-1)@self.R

    def plane(self,p,z):
        r=self.rays(p)
        return self.eye+(z-self.eye[2])*r/r[...,2:]

    def project(self,v):
        q=np.asarray(v)@self.R.T+self.t
        return self.f*q[...,:2]/q[...,2:]+self.center


def rings(par):
    return dict(cap=(1.,0.),flange=(par[5],par[6]),joint=(par[7],par[8]),base=(par[9],par[10]))


def ring_error(cam,p,r,z,offset):
    def residual(q):
        return np.linalg.norm(cam.plane(q,z)[:,:2]-offset,axis=1)-r
    value=residual(p)
    gradient=np.stack([residual(p+step)-value for step in np.eye(2)*.2],axis=1)/.2
    return value/np.maximum(np.linalg.norm(gradient,axis=1),1e-6)


def fit(spec):
    pts={k:np.asarray(v,float) for k,v in spec['rings'].items()}
    def errors(par,hold):
        cam=Camera(spec,par);all_rings=rings(par);res=[]
        for k,p in pts.items():
            reserved=np.isin(np.arange(len(p)),spec['reserved_indices'][k])
            r,z=all_rings[k];offset=z*par[11:13]
            res.extend(ring_error(cam,p[reserved if hold else ~reserved],r,z,offset))
            if not hold:
                a=np.linspace(0,2*np.pi,361)
                uv=cam.project(np.c_[r*np.cos(a)+offset[0],r*np.sin(a)+offset[1],np.full_like(a,z)])
                ends=[uv[:,0].min(),uv[:,0].max()]
                res.extend(np.repeat(np.array(ends)-spec['lateral_extrema_px'][k],3))
        return np.array(res)
    x0=[.64,0.,-.06,-1.,4.,1.21,-.42,1.20,-.75,1.3,-2.15,0.,0.]
    low=[.25,-.3,-1.,-2.,2.,1.04,-1.,1.,-1.5,.8,-4.,-.35,-.35]
    high=[1.12,.3,1.,-.2,7.,1.45,-.08,1.44,-.2,1.9,-1.,.35,.35]
    result=least_squares(lambda v:errors(v,False),x0,bounds=(low,high),loss='soft_l1',f_scale=1.8)
    par=result.x;cam=Camera(spec,par);pen=np.array(spec['pen'],float)
    top=cam.plane(pen[:1],0.)[0];foot=cam.plane(pen[1:],par[6])[0]
    scale=spec['pen_m']/np.linalg.norm(top-foot)
    profile={k:dict(center_xy_m=(z*par[11:13]*scale).tolist(),radius_m=float(r*scale),height_above_road_m=float((z-par[10])*scale)) for k,(r,z) in rings(par).items()}
    return dict(parameters=par.tolist(),metres_per_unit=float(scale),camera_focal_px=float(cam.f),
                profile=profile,pen_3d_endpoints_model_units=[top.tolist(),foot.tolist()],
                fit_rms_px=float(np.mean(errors(par,False)**2)**.5),
                reserved_rms_px=float(np.mean(errors(par,True)**2)**.5),
                reserved_max_px=float(abs(errors(par,True)).max()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    args=p.parse_args();s=json.loads((HERE/'urayasu.json').read_text());q=fit(s)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(q['profile']),q['reserved_rms_px'])
