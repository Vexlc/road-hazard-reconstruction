from pathlib import Path
from functools import lru_cache
import json
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1]
from road_geometry import read_json
from browser_data import base_points
from camera_geometry import crop_box

PAPER='#f4f3eb';INK='#1c2e28';MUTED='#51665b';LINE='#ccd2c6';DARK='#192a24';WHITE='#fafaf3'
ORANGE='#ee9163';CYAN='#58bed0';GOLD='#dfbd69';GREY='#a4afa8'
@lru_cache(maxsize=50)
def font(n,bold=False):
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-'+('Bold' if bold else 'Regular')+'.ttc',n,index=2)
@lru_cache(maxsize=10)
def case_data(case):
    info=read_json(ROOT/'cases'/(case+'.json'));d=ROOT/'cases'/(case+'.npz')
    with np.load(d) as f:z={k:f[k] for k in ('row','frame_offsets','instance','xyz','action','part','sensor')}
    return info,z

def cloud_section(case,aid,frame,w,h,annot=True):
    info,d=case_data(case);a=next(a for a in info['assets'] if a['asset_id']==aid)
    B=np.array(a['basis_columns']);c=np.asarray(a['center_world']);full,_=base_points(str(ROOT),case[:3],frame)
    lo,hi=d['frame_offsets'][frame:frame+2];sel=(d['instance'][lo:hi]==a['instance'])&(d['action'][lo:hi]==1)
    idx=d['row'][lo:hi][sel];orig=(full[idx]-c)@B;new=(d['xyz'][lo:hi][sel]-c)@B;pr=d['part'][lo:hi][sel]
    low,high=a['z_range_m'];lo_z=np.floor(min(-.035,low-.035)*20)/20;hi_z=np.ceil(max(.04,high+.035)*20)/20
    pad=48 if annot else 43;area=(pad,13,w-15,h-28)
    other=np.flatnonzero(np.all(abs(full-c)<2.,axis=1));context=(full[other]-c)@B
    ok=(abs(context[:,0])<1.05)&(abs(context[:,1])<1.12)&(abs(context[:,2])<.055)
    context=context[ok][::max(1,int(ok.sum())//1000)]
    def coords(v):
        return np.column_stack([area[0]+(v[:,1]+1.12)/2.24*(area[2]-area[0]),
            area[3]-(v[:,2]-lo_z)/(hi_z-lo_z)*(area[3]-area[1])])
    im=Image.new('RGB',(w,h),DARK);dr=ImageDraw.Draw(im)
    for value in (lo_z,0,hi_z):
        q=float(coords(np.array([[0,0,value]]))[0,1]);dr.line((pad,q,w-10,q),fill='#40574a',width=1)
        dr.text((4,q-8),f'{value:.2f}',font=font(12),fill='#bdcbbf')
    def dots(points,col,r):
        if not len(points):return
        for x,y in coords(points):
            if area[0]<=x<=area[2] and area[1]<=y<=area[3]:
                dr.ellipse((x-r,y-r,x+r,y+r),fill=col)
    dots(context,'#667c70',1.1);dots(orig,GREY,1.4)
    for category,color in [(pr<5,CYAN),((pr==6)|(pr==7),CYAN),((pr==5)|(pr==8),GOLD),(pr>=9,ORANGE)]:
        dots(new[category],color,2)
    dr.text((pad,h-23),'纵向 −1.12 至 +1.12 m',font=font(12),fill='#bdcbbf')
    return im,len(idx)

def photo(p,size):
    im=Image.open(p).convert('RGB')
    return ImageOps.fit(im,size,method=Image.Resampling.LANCZOS,centering=(.5,.5))

def poster():
    samples=[('109_23','P23','积水坑洞'),('034_34','M34','缺盖井'),('110_30','A30','翻起的方井盖')]
    w,h=1660,946;im=Image.new('RGB',(w,h),PAPER);d=ImageDraw.Draw(im)
    d.text((38,22),'路面病害的相机与雷达对照',font=font(51,True),fill=INK)
    d.text((40,99),'前视和后视来自同一次通过。最右侧显示前视帧编号对应的雷达扫描，高度以所在路面为零。',font=font(20),fill=MUTED)
    xs=[183,540,897,1254];cw=337;ch=199
    d.line((38,147,w-38,147),fill=LINE,width=2)
    for x,label in zip(xs,['前视 · 原路面','前视 · 迁入后','后视 · 驶过后','雷达回波 · 局部侧视']):
        d.text((x,160),label,font=font(22,True),fill=INK)
    for i,(case,aid,label) in enumerate(samples):
        y=210+i*230;pack=ROOT/'camera'/case;vs=[r for r in read_json(pack/'views.json') if r['asset_id']==aid]
        f=next(r for r in vs if r['camera']=='front_camera');b=next(r for r in vs if r['camera']=='back_camera')
        d.text((39,y+8),aid,font=font(30,True),fill=INK);d.text((39,y+56),label,font=font(19,True),fill=INK)
        d.text((39,y+96),'路段 '+case[:3],font=font(16),fill=MUTED)
        d.text((39,y+127),'前视 %03d'%f['frame'],font=font(15),fill=MUTED);d.text((39,y+154),'后视 %03d'%b['frame'],font=font(15),fill=MUTED)
        for x,img in zip(xs,[f['before_stem'],f['after_stem'],b['after_stem']]):im.paste(photo(pack/img,(cw,ch)),(x,y))
        sect,n=cloud_section(case,aid,f['frame'],cw,ch);im.paste(sect,(xs[-1],y));d.line((38,y+214,w-38,y+214),fill=LINE)
    d.text((39,905),'灰色为原始回波。蓝色为坑壁或井内表面，金色为井沿，橙色为盖板。侧视图纵轴单位为米，各行高度范围见刻度。',font=font(17),fill=MUTED)
    im.save(ROOT/'相机与雷达对照.jpg',quality=94,subsampling=0)

if __name__=='__main__':poster()
