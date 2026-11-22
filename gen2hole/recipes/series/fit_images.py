"""Fit depression outlines and frame/lid landmarks in 480-pixel normalized photo coordinates."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/'calibration'
cv2.setNumThreads(2)
Y,X=np.mgrid[:480,:480].astype('f4')
def main_component(mask):
    n,l,stats,c=cv2.connectedComponentsWithStats(np.uint8(mask),8)
    if n<2:raise ValueError('empty foreground')
    score=stats[1:,4]*np.exp(-np.sum((c[1:]-[239,239])**2,1)/(2*130**2))
    return ndi.binary_fill_holes(l==1+int(np.argmax(score)))
def poly(mask):
    cs=cv2.findContours(mask.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    return cv2.approxPolyDP(max(cs,key=cv2.contourArea),1.0,True)[:,0,:].tolist()
def fit_rgb_bowl(cfg, image, shape, patch):
    """Segment a single dry depression using a loose box marked on the RGB."""
    if cfg['model'] != 'pothole':
        raise ValueError("designed_bowl requires a dry pothole")
    box = patch.get('box')
    if not isinstance(box, list) or len(box) != 4 or not all(type(x) is int for x in box):
        raise ValueError("Specify box as integer [left, top, right, bottom] in 480 px coordinates")
    left, top, right, bottom = box
    if not (0 < left < right < 479 and 0 < top < bottom < 479):
        raise ValueError("The box must enclose the pit with intact road outside each edge")
    if any(not np.isfinite(cfg[k]) or cfg[k] <= 0 for k in ('span', 'depth')):
        raise ValueError("span and depth must be positive finite metres")
    classes = np.zeros(image.shape[:2], dtype=np.uint8)
    rect = (left, top, right-left+1, bottom-top+1)
    cv2.setRNGSeed(47)
    cv2.grabCut(image, classes, rect, np.zeros((1,65)), np.zeros((1,65)), 6, cv2.GC_INIT_WITH_RECT)
    mask = main_component(ndi.binary_closing(classes % 2 > 0, iterations=3))
    if mask.sum() < 100:
        raise ValueError(f"{cfg['id']}: insufficient foreground in the supplied RGB box")
    full_mask = cv2.resize(mask.astype('u1'), (shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(full_mask*255).save(CAL / (cfg['id']+'_outline.png'))
    cfg.update(outline=poly(mask), box=box, family=cfg.get('family','ravelled'), mask_fraction=float(mask.mean()))
    return cfg


def fit_pit(cfg,img,full,patch):
    if cfg.get('depth_profile') in ('masked_groove','scanned_relief'):
        path=CAL/(cfg['id']+'_outline.png')
        with Image.open(path) as image:
            mask=np.asarray(image.convert('L'))>0
        if mask.shape!=full[:2] or not mask.any() or mask.all():
            raise ValueError(f"Invalid registered crack mask {path}")
        Image.fromarray(np.uint8(mask)*255).save(path)
        cfg.update(outline=path.name, family=cfg.get('family','crack_network'))
        return cfg
    if cfg.get('depth_profile') == 'designed_bowl':
        return fit_rgb_bowl(cfg,img,full,patch)
    if cfg.get('depth_profile') is not None:
        raise ValueError(f"Unknown depth_profile for {cfg['id']}")
    key=cfg['id']
    depth=np.load(CAL/(key+'_depth.npz'))['inverse_depth'].astype('f4')
    dep=cv2.resize(depth,(480,480),interpolation=cv2.INTER_AREA)
    x=(X-239.5)/239.5;y=(Y-239.5)/239.5
    outer=(abs(x)>.91)|(abs(y)>.91)|(x*x+y*y>1.12)
    a=np.stack([np.ones_like(x),x,y,x*y,x*x,y*y],-1)
    use=outer.copy()
    for _ in range(4):
        coef=np.linalg.lstsq(a[use],dep[use],rcond=None)[0];err=dep-a@coef
        mad=max(np.median(abs(err[use]-np.median(err[use])))*1.4826,.012)
        use=outer&(abs(err-np.median(err[use]))<2.8*mad)
    d=ndi.gaussian_filter(a@coef-dep,1.7)
    ref=max(np.percentile(d,98),.06)
    seed=d>max(.10*ref,1.5*min(mad,.1),.03)
    seed[[0,-1],:]=False;seed[:,[0,-1]]=False
    mask=main_component(ndi.binary_closing(seed,iterations=4))
    if 'outline' in patch:
        mask=np.zeros((480,480),'u1');cv2.fillPoly(mask,[np.asarray(patch['outline'],'i4')],1);mask=mask.astype(bool)
    dist=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
    band=patch.get('band',12)
    classes=np.where(mask,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('u1')
    classes[dist>band]=cv2.GC_FGD;classes[dist<-band]=cv2.GC_BGD
    cv2.setRNGSeed(47)
    cv2.grabCut(img,classes,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
    mask=main_component(ndi.binary_closing(classes%2>0,iterations=2))
    def save_mask(mask,name):
        large=cv2.resize(mask.astype('u1')*255,(full[1],full[0]),interpolation=cv2.INTER_CUBIC)
        Image.fromarray(np.uint8(large>127)*255).save(CAL/(key+'_'+name+'.png'))
    cfg['outline']=poly(mask);save_mask(mask,'outline')
    cfg['family']=cfg.get('family') or {'wet':'wet_pothole','spall':'delamination','pothole':'ravelled'}[cfg['model']]
    if cfg['model']=='wet':
        lum=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype('f4')
        tex=np.sqrt(np.maximum(0,ndi.uniform_filter(lum**2,7)-ndi.uniform_filter(lum,7)**2))
        inside=ndi.binary_erosion(mask,iterations=6)
        if 'water' in patch:
            water=np.zeros_like(mask,'u1');cv2.fillPoly(water,[np.array(patch['water'],'i4')],1);water=(water>0)&mask
        else:
            thresh=min(np.percentile(tex[inside],30),9.)
            wk=ndi.binary_closing((tex<thresh)&inside&(d>ref*.32),iterations=4)
            water=main_component(wk) if wk.any() else np.zeros_like(mask)
            water &=mask
        if water.sum()>mask.sum()*.025 and not patch.get('dry'):
            cfg['water']=poly(water);save_mask(water,'water')
    cfg['mask_fraction']=float(mask.mean())
    return cfg
def grey_edges(rgb):
    grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    smooth=cv2.GaussianBlur(grey,(5,5),1.4).astype('f4')
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    return grey,smooth,gx,gy
def hough(grey,m,rect=False):
    lo,hi=(56,213) if m!='offset' else (38,152)
    det=cv2.HoughCircles(cv2.medianBlur(grey,7),cv2.HOUGH_GRADIENT,1.2,12,param1=85,param2=45,minRadius=lo,maxRadius=hi)
    return [] if det is None else det[0].tolist()
def concentric(img,edges,kind,shape):
    grey,sm,gx,gy=edges
    prx=128 if kind=='offset' else 240
    search=[(prx,240)] if shape=='rect' else [(v[0],v[1]) for v in hough(grey,kind) if abs(v[0]-prx)<(74 if kind=='offset' else 66) and abs(v[1]-240)<74][:12]+[(prx,240)]
    if kind in ('raised','hatch'):search.extend([(250,252),(245,263)])
    if shape=='rect':
        xx=np.arange(480)
        def peaks(line,lo,hi):
            a=ndi.gaussian_filter1d(line.astype(float),1.35)
            pp=(a==ndi.maximum_filter1d(a,9))&(xx>=lo)&(xx<=hi)
            return sorted(np.flatnonzero(pp),key=lambda i:-a[i])[:12]
        sx=np.mean(abs(gx[182:298]),0)+np.mean(abs(gx[256:321]),0)*.4
        sy=np.mean(abs(gy[:,169:312]),1)
        if kind=='hatch':
            sy=np.mean(abs(gy[:,159:321]),1)
        sides=[peaks(sx,11,161),peaks(sx,321,468),peaks(sy,10,162),peaks(sy,320,470)]
        sides=[v[:5] for v in sides]
        for l in sides[0]:
            for r in sides[1]:
                cx=(int(l)+int(r))*.5
                if abs(cx-240)<50:
                    for t in sides[2][:3]:
                        for b in sides[3][:3]:
                            cy=(int(t)+int(b))*.5
                            if abs(cy-240)<58:search.append((cx,cy))
    ang=np.linspace(0,2*np.pi,188,endpoint=False)
    if kind=='raised':ang=ang[np.cos(ang)>-.12]
    if kind=='hatch':ang=ang[np.sin(ang)>-.16]
    co,si=np.cos(ang),np.sin(ang)
    if shape=='rect':
        sc=np.maximum(abs(co),abs(si));co/=sc;si/=sc
    rs=np.arange(50 if kind=='offset' else 76,159 if kind=='offset' else 226)
    if shape=='rect': ratios=[.82,.9,.96,1.,1.05,1.11]
    else:ratios=[.96,1.,1.04]
    unique=set();best=None
    for cx,cy in search:
        if (round(cx/2),round(cy/2)) in unique:continue
        unique.add((round(cx/2),round(cy/2)))
        for ratio in ratios:
            xp=np.clip(np.rint(cx+rs[:,None]*co).astype(int),0,479)
            yp=np.clip(np.rint(cy+rs[:,None]*ratio*si).astype(int),0,479)
            vals=sm[yp,xp];val=np.median(vals,axis=1)
            if shape=='rect':
                j=abs(co)>.99
                grad=np.where(j[None,:],gx[yp,xp]*np.sign(co),gy[yp,xp]*np.sign(si))
            else:grad=gx[yp,xp]*co+gy[yp,xp]*si
            strength=np.mean(np.minimum(abs(grad),150),1)
            # Brightening moving outward usually locates the dark throat.
            pos=np.mean(np.clip(grad,-60,130),1)
            score=ndi.gaussian_filter1d(strength,1.1)
            shift=round(np.clip(rs.mean()*.08,5,13))
            for outer in np.argsort(score)[-9:]:
                rv=rs[outer]
                if min(cx-rv,cy-ratio*rv,479-cx-rv,479-cy-ratio*rv)<2:continue
                low=max(0,outer-round(rv*.24));high=max(0,outer-max(4,round(rv*.052)))
                if high<=low:continue
                edge=pos[low:high]+.38*score[low:high]
                inner=low+int(np.argmax(edge))
                value=.80*score[outer]+.50*score[inner]+.28*pos[inner]+.16*(val[outer]-val[max(0,inner-shift)])-.030*(abs(cx-prx)+abs(cy-240))-.11*abs((rv-rs[inner])/rv-.105)*100
                if best is None or value>best[0]:best=(value,cx,cy,rv,rv*ratio,rs[inner],rs[inner]*ratio)
    if best is None: raise ValueError('no frame')
    _,cx,cy,rx,ry,ix,iy=best
    return [round(float(v),1) for v in (cx,cy,rx,ry)],[round(float(v),1) for v in (cx,cy,ix,iy)]
def fit_chamber(cfg,img,patch):
    kind=cfg['model'];shape=cfg['shape'];edges=grey_edges(img)
    if 'outer' in patch:
        outer=patch['outer'];inner=patch.get('inner',[outer[0],outer[1],outer[2]*.87,outer[3]*.87])
    else:outer,inner=concentric(img,edges,kind,shape)
    if 'inner' in patch:inner=patch['inner']
    cfg.update(outer=outer,inner=inner)
    if 'road_fill' in patch:cfg['road_fill']=patch['road_fill']
    if cfg['category']=='missing' and shape=='rect':
        cfg['length']=round(cfg['width']*outer[3]/outer[2],4)
    cx,cy,rx,ry=outer;xi,yi,ix,iy=inner
    if kind=='brick' or 'collar' in patch:cfg['collar']=min(patch.get('collar',1.21),max(1.,(min(cx,cy,479-cx,479-cy)-4)/max(rx,ry)))
    if cfg['category']=='raised' and kind!='offset':cfg['wall_source']=cfg.pop('donor')
    else:cfg.pop('donor',None)
    cfg['wall_end']=[xi,yi+.22*iy,ix*.55,iy*.5]
    cfg['bottom_photo']=[xi,yi+.23*iy,ix*.33,iy*.32]
    if cfg['category']=='raised':
        thick=round(.025+(int(cfg['id'][1:])%6)*.002,3)
        if kind=='offset':
            cir=hough(edges[0],kind)
            opts=[(x,y,r) for x,y,r in cir if x>cx+rx*1.15 and x<465 and abs(y-cy)<ry*.69 and ix*.73<r<rx*1.2]
            x,y,r=min(opts,key=lambda c:abs(c[1]-cy)+.68*abs(c[2]-ix)+.10*abs(c[0]-(cx+2.25*rx))) if opts else (cx+2.18*rx,cy,ix)
            ell=patch.get('ellipse',[x,y,r,r]);dia=cfg['width']*np.clip(ell[2]/rx,.80,1.03)
            cfg['lid']=dict(type='displaced',ellipse=list(np.round(ell,1)),diameter=float(dia),thickness=thick)
        elif kind=='hatch':
            h=2*iy*np.cos(np.deg2rad(cfg['angle']))
            q=patch.get('quad',[[xi-ix*.94,yi-iy],[xi+ix*.94,yi-iy],[xi+ix,yi-iy+h],[xi-ix,yi-iy+h]])
            cfg['lid']=dict(type='end_hinge',quad=np.round(q,1).tolist(),width=cfg['width']*ix/rx*1.065,length=cfg['length']*iy/ry*1.065,thickness=thick)
        else:
            ex=ix*np.cos(np.deg2rad(cfg['angle']));ell=patch.get('ellipse',[xi-ix+ex,yi,ex,iy])
            cfg['lid']=dict(type='side_hinge',ellipse=list(np.round(ell,1)),bow_x=patch.get('bow_x',0),diameter=cfg['width']*ix/rx*1.062,thickness=thick)
    for key in ('wall_end','bottom_photo','wall_source'):
        if key in patch:cfg[key]=patch[key]
    return cfg
def fit(cfg,over):
    cfg={k:v for k,v in cfg.items() if k!='prompt'}
    key=cfg['id'];f=ROOT/'references'/(key+'.png')
    img=np.array(Image.open(f).convert('RGB'))
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected a square nadir image: {f}")
    small=cv2.resize(img,(480,480),interpolation=cv2.INTER_AREA)
    if cfg['category']=='pothole':cfg=fit_pit(cfg,small,img.shape,over)
    else:cfg=fit_chamber(cfg,small,over)
    (CAL/(key+'_fit.json')).write_text(json.dumps(cfg,ensure_ascii=False,indent=1)+'\n')
    return cfg
def sync():
    rows=[json.loads(p.read_text()) for p in sorted(CAL.glob('*_fit.json'))]
    p=CAL/'layout.tmp';p.write_text(json.dumps({r['id']:r for r in rows},ensure_ascii=False,indent=1)+'\n');p.replace(CAL/'layout.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('ids',nargs='*');ap.add_argument('--redo',action='store_true');a=ap.parse_args()
    overrides=json.loads((CAL/'overrides.json').read_text()) if (CAL/'overrides.json').exists() else {}
    for cfg in json.loads((CAL/'designs.json').read_text()):
        k=cfg['id']
        if a.ids and k not in a.ids:continue
        if not (ROOT/'references'/(k+'.png')).exists():continue
        if k[0]=='P' and cfg.get('depth_profile') not in ('designed_bowl','masked_groove','scanned_relief') and not (CAL/(k+'_depth.npz')).exists():continue
        if (CAL/(k+'_fit.json')).exists() and not a.redo:continue
        fit(cfg,overrides.get(k,{}));print(k,flush=True)
    sync()
if __name__=='__main__':main()
