"""Render real RGB, exported crack models, signed metric profiles and calibrated camera crops."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.ndimage import map_coordinates

SHARED=runpy.run_path(str(Path(__file__).with_name('render-asset-plates.py')))
label=SHARED['label']
PAPER,INK,LINE=SHARED['PAPER'],SHARED['INK'],SHARED['LINE']


def fit(sheet,im,x,y,w,h):
    im=ImageOps.contain(im.convert('RGB'),(w,h),Image.Resampling.LANCZOS)
    sheet.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))


def profile(sheet,draw,meta,field,x,y,w,h):
    heights=field['height']
    mask=field['defect_mask']>.5
    n=mask.shape[0];pitch=float(field['width'])/(n-1)
    # Choose a depth-bearing pixel away from the crop border, then intersect
    # the fissure between two road shoulders using its sampled depth profile.
    inside=mask.copy()
    cut=int(n*.12)
    inside[:cut]=inside[-cut:]=False;inside[:,:cut]=inside[:,-cut:]=False
    rows,cols=np.where(inside)
    if not len(rows):
        rows,cols=np.where(mask)
    deep=-heights[rows,cols]
    score=deep/max(deep.max(),1e-9)-.25*np.hypot((rows-n/2)/n,(cols-n/2)/n)
    positions=np.linspace(-.075,.075,151)
    candidates=np.argsort(-score)[::max(1,len(score)//300)]
    best=None
    for at in candidates:
        row,col=rows[at],cols[at]
        for angle in np.arange(8)*np.pi/8:
            dx,dy=np.cos(angle),np.sin(angle)
            z=map_coordinates(heights,[row+dy*positions/pitch,col+dx*positions/pitch],order=1,mode='nearest')
            if min(z[0],z[-1]) < -.0001:
                continue
            width_at_half=int(np.sum(z<z[len(z)//2]*.5))
            value=score[at]-.25*width_at_half/len(z)
            if best is None or value>best[0]:
                best=(value,row,col,dx,dy,z)
    if best is None:raise ValueError(meta['id']+' has no 150 mm transect between road shoulders')
    _,row,col,dx,dy,z=best
    top=y+30;bottom=y+h-43;left=x+45;right=x+w-16
    limit=.025
    for mm in (0,10,20):
        gy=round(top+(bottom-top)*mm/(limit*1000))
        draw.line((left,gy,right,gy),fill=LINE,width=1)
        label(draw,(x+4,gy-5),str(-mm),14)
    xy=[(left+(right-left)*(q+.075)/.15,top+(bottom-top)*(-v)/limit) for q,v in zip(positions,z)]
    draw.line(xy,fill='#a2492c',width=4)
    label(draw,(x+4,y+2),'高程 mm',15)
    label(draw,(x+48,y+h-24),'←  150 mm 截面  →',16)
    half=.075/pitch
    return (col-dx*half,row-dy*half,col+dx*half,row+dy*half)


def plate(run,out):
    bundle=run/'series'
    rows=json.loads((bundle/'assets/asset_manifest.json').read_text())
    size=(1680,170+len(rows)*310+50)
    page=Image.new('RGB',size,PAPER);draw=ImageDraw.Draw(page)
    label(draw,(32,20),'路面裂缝的实拍重建与自驾视图',43,True)
    label(draw,(32,83),'照片与标注决定缝迹。横向尺度、侧壁坡度与深度上限由设计给定，剖面按米制网格绘制。',23)
    for x,t in [(32,'输入照片与取样截线'),(420,'GLB 斜视'),(830,'网格截面'),(1215,'PandaSet 前视 · 迁入后')]:label(draw,(x,131),t,24,True)
    for i,row in enumerate(rows):
        key=row['id'];y=174+i*310
        label(draw,(32,y),key+'    '+row['label'],24,True)
        with np.load(bundle/'assets'/row['surface']) as field:
            pts=profile(page,draw,row,field,827,y+72,345,171)
        with Image.open(bundle/row['source_image']) as src:
            marked=src.convert('RGB');pen=ImageDraw.Draw(marked)
            pen.line(pts,fill='#ffce48',width=4)
            fit(page,marked,32,y+39,353,233)
        with Image.open(bundle/'views'/(key+'_inspection.jpg')) as im:fit(page,im,420,y+39,372,233)
        view=next(f for p in sorted((run/'sensors/camera').glob('*/views.json')) for f in json.loads(p.read_text())
                  if f['asset_id']==key and f['available'] and f['camera']=='front_camera')
        with Image.open(run/'sensors/camera'/view['case']/view['after_stem']) as im:
            x0,y0,x1,y1=view['xyxy_target_in_crop']
            crop=(max(0,x0-30),max(0,y0-42),min(im.width,x1+30),min(im.height,y1+35))
            fit(page,im.crop(crop),1215,y+39,424,233)
        label(draw,(1215,y+270),f"{view['site']} · 帧 {view['frame']} · 原相机裁切",16)
        draw.line((32,y+303,size[0]-32,y+303),fill=LINE,width=1)
    label(draw,(32,size[1]-34),'图像与标注来自 Rossini 等人的 Asphalt Pavement Crack Segmentation Dataset，CC BY 4.0。',17)
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,quality=94,subsampling=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plate(a.run,a.out)
