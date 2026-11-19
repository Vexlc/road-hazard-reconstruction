"""Render all catalog PLYs as five paginated 4x4 turntable videos."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / 'outputs/road_hazard_expansion'
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs/道路病害_五类4x4视频'
WORK = BASE / 'work/visualization'
CATS = [('stereo','01_双目实拍坑洞'),('rgbd','02_RGB-D实拍坑洞'),
        ('pothole','03_程序化建模坑洞'),('missing','04_缺盖窨井'),('raised','05_掀盖与移位窨井')]
TITLES = dict((c, n[3:]) for c,n in CATS)
COLORS = {'stereo':'#66ddd0','rgbd':'#7abaff','pothole':'#e4b773','missing':'#b79cf2','raised':'#ef929d'}
MAT = {'fine_asphalt':'细沥青','aged_asphalt':'老化沥青','chip_seal':'碎石封层','concrete':'水泥'}
FAMILY = {'spall':'浅层剥落','bowled':'碗状坑','deep_break':'深度破损','crescent':'月牙坑',
          'wheel_rut':'车辙长坑','joined':'连通坑','cluster':'群坑','patch_break':'修补破损',
          'slab_corner':'板角破损','waterlogged':'积水坑',
          'round':'圆形','oval':'椭圆形','square':'方形','rectangular':'长方形'}
STATE = {'flush':'齐平井沿','masonry':'砖砌井壁','broken_flange':'破损井沿','settled':'沉降井框',
         'ajar14':'微开 14°','tilted29':'倾斜 29°','propped43':'支起 43°','raised62':'掀开 62°',
         'upright82':'近直立 82°','slid':'平移','skewed':'斜移','removed_side':'移至井旁'}
W,H = 2560,1920
X0,Y0,GAP,CW,CH = 24,124,16,616,420
IW,IH = 616,330
FPS,DURATION = 24,4
FRAMES=FPS*DURATION
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

def font(n,bold=False): return ImageFont.truetype(BOLD if bold else FONT,n)
def tensor(a): return torch.as_tensor(np.ascontiguousarray(a),device='cuda',dtype=torch.float32)

def load_asset(m):
    a=PlyData.read(LIB/m['ply'],mmap='r')['vertex'].data
    def cols(names): return np.stack([a[n] for n in names],1).astype('f4')
    xyz=cols(['x','y','z'])
    opacity=1/(1+np.exp(-cols(['opacity'])[:,0]))
    dc=cols([f'f_dc_{i}' for i in range(3)])[:,None,:]
    rest=cols([f'f_rest_{i}' for i in range(9)]).reshape(-1,3,3).transpose(0,2,1)
    state=dict(means=tensor(xyz),quats=tensor(cols([f'rot_{i}' for i in range(4)])),
               scales=tensor(np.exp(cols([f'scale_{i}' for i in range(3)]))),
               opacities=tensor(opacity),colors=tensor(np.concatenate([dc,rest],1)))
    visible=xyz[opacity>.45]
    if len(visible)<30: visible=xyz
    lo,hi=np.quantile(visible,[.005,.995],axis=0)
    # The wells extend underground; frame the opening and raised lid, not the shaft bottom.
    if m['category'] in ('missing','raised'): lo[2]=max(lo[2],-.16)
    center=(lo+hi)/2
    # Fit sampled visible geometry rather than empty corners of its bounding box.
    support=visible[np.all((visible>=lo)&(visible<=hi),axis=1)]
    corners=support[::max(1,len(support)//4000)]
    return state,center,corners

def cameras(center,corners,cat):
    elevation=math.radians(58 if cat in ('stereo','rgbd','pothole') else 51)
    focal=IH*1.35
    angles=np.arange(FRAMES)/FRAMES*2*np.pi + math.radians(-61)
    bases=[]; distances=[]
    for theta in angles:
        away=np.array([math.cos(theta)*math.cos(elevation),math.sin(theta)*math.cos(elevation),math.sin(elevation)])
        forward=-away
        right=np.cross(forward,[0.,0.,1.]); right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        basis=np.stack([right,down,forward],1)
        q=(corners-center)@basis
        required=np.maximum(np.abs(q[:,0])*focal/(IW*.45)-q[:,2], np.abs(q[:,1])*focal/(IH*.44)-q[:,2])
        distances.append(required.max()); bases.append((basis,away))
    distance=max(max(distances),.01)
    views=[]
    for basis,away in bases:
        pose=np.eye(4); pose[:3,:3]=basis; pose[:3,3]=center+away*distance
        views.append(np.linalg.inv(pose))
    k=np.array([[focal,0,IW/2],[0,focal,IH/2],[0,0,1]],dtype='f4')
    return tensor(np.array(views)),tensor(k)[None]

@torch.no_grad()
def render_asset(m,batch=8):
    state,center,corners=load_asset(m)
    views,k=cameras(center,corners,m['category'])
    images=[]
    for start in range(0,FRAMES,batch):
        n=min(batch,FRAMES-start)
        rgb,alpha,_=rasterization(**state,viewmats=views[start:start+n],Ks=k.expand(n,-1,-1),
            width=IW,height=IH,sh_degree=1,packed=False,
            backgrounds=tensor([[.23,.27,.30]]).expand(n,-1),rasterize_mode='antialiased')
        if not bool(torch.isfinite(rgb).all()): raise ValueError('Non-finite '+m['id'])
        images.append((rgb.clamp(0,1)*255+.5).to(torch.uint8).cpu().numpy())
    return np.concatenate(images)

def description(m):
    cat=m['category']
    if cat=='stereo': return f"双目采集  ·  数据集 {m['source_dataset']} / 帧 {m['source_frame']}"
    if cat=='rgbd': return f"RGB-D 深度重建  ·  {m['id'].replace('rgbd_pothole_','')}"
    title=FAMILY.get(m['family'],m['family'])
    if m.get('state'): title+=' · '+STATE[m['state']]
    return title+' · '+MAT.get(m['material'],m['material'])

def dims(m):
    text=f"坑口 / 井口 {m['width_m']*100:.1f} × {m['length_m']*100:.1f} cm"
    if m['category'] in ('stereo','rgbd'): text+=f"  |  深度 P95 ≈ {m['depth_p95_m']*100:.1f} cm"
    else: text+=f"  |  设计深度 {m['depth_design_m']*100:.1f} cm"
    return text

def page_base(cat,page,total,items,start,count):
    im=Image.new('RGB',(W,H),'#101923'); d=ImageDraw.Draw(im)
    d.rectangle((24,30,31,84),fill=COLORS[cat])
    d.text((48,22),TITLES[cat],font=font(43,True),fill='#f3f6fa')
    d.text((50,79),f'4 × 4 全库巡览   /   {count} 个资产   /   独立 3DGS 旋转渲染',font=font(22),fill='#9eb2c4')
    d.text((W-400,35),f'第 {page:02d} / {total:02d} 页',font=font(32,True),fill=COLORS[cat])
    d.text((W-400,79),f'编号 {start+1:03d}–{start+len(items):03d}  ·  每页 4 秒',font=font(21),fill='#aabccb')
    for slot in range(16):
        x=X0+(slot%4)*(CW+GAP); y=Y0+(slot//4)*(CH+GAP)
        d.rectangle((x,y,x+CW-1,y+CH-1),fill='#1b2733')
        if slot>=len(items):
            d.text((x+CW/2-38,y+CH/2-14),'—',font=font(40),fill='#3f5162'); continue
        m=items[slot]
        d.text((x+12,y+6),f'{start+slot+1:03d}  '+description(m),font=font(21,True),fill='#e7eff6')
        d.text((x+12,y+374),dims(m),font=font(18),fill='#bdcbd6')
        identifier=m['id']
        # Full IDs remain available in the index, while the on-screen label stays readable.
        size=15
        while d.textbbox((0,0),identifier,font=font(size))[2]>CW-24 and size>10: size-=1
        d.text((x+12,y+399),identifier,font=font(size),fill='#8198ac')
    note='按来源帧计数；相同病害可能重复出现。RGB-D 尺寸采用估计内参。' if cat in ('stereo','rgbd') else '程序化三维结构 + 摄影测量来源表面材质；尺寸为建模设定。'
    d.text((24,H-42),note,font=font(20),fill='#99adbf')
    d.text((W-535,H-42),'每格独立取景  ·  画面大小不代表实物大小',font=font(19),fill='#99adbf')
    return np.array(im)

def encode_command(path):
    return [imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','veryfast',
            '-crf','20','-pix_fmt','yuv420p','-threads','6','-movflags','+faststart',str(path)]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--test',action='store_true'); p.add_argument('--category'); args=p.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
    rows=json.loads((LIB/'asset_manifest.json').read_text())
    report=[]; coverage=[]
    for cat,filename in CATS:
        if args.category and cat!=args.category: continue
        entries=sorted([m for m in rows if m['category']==cat],key=lambda x:x['id'])
        total=math.ceil(len(entries)/16)
        if args.test:
            m=entries[0]; t=time.time(); images=render_asset(m)
            Image.fromarray(images[0]).save(WORK/f'test_{cat}.jpg')
            print(json.dumps({'test':cat,'seconds':round(time.time()-t,2),'shape':list(images.shape)}),flush=True)
            continue
        dest=OUT/(filename+'_4x4.mp4')
        log=open(WORK/(cat+'_ffmpeg.log'),'w')
        process=subprocess.Popen(encode_command(dest),stdin=subprocess.PIPE,stderr=log)
        t=time.time()
        try:
            for page,start in enumerate(range(0,len(entries),16),1):
                items=entries[start:start+16]
                base=page_base(cat,page,total,items,start,len(entries))
                page_frames=np.broadcast_to(base,(FRAMES,H,W,3)).copy()
                for slot,m in enumerate(items):
                    images=render_asset(m)
                    x=X0+(slot%4)*(CW+GAP); y=Y0+(slot//4)*(CH+GAP)+40
                    page_frames[:,y:y+IH,x:x+IW]=images
                    coverage.append({'category':cat,'video':dest.name,'page':page,'slot':slot+1,'row':slot//4+1,
                                     'column':slot%4+1,'start_seconds':(page-1)*DURATION,'end_seconds':page*DURATION,
                                     'number':start+slot+1,'id':m['id'],'description':description(m),'source_ply':str(LIB/m['ply'])})
                if page==1 or page==total:
                    Image.fromarray(page_frames[0]).save(OUT/(filename+f'_第{page:02d}页.jpg'),quality=94)
                for frame in page_frames: process.stdin.write(frame.tobytes())
                del page_frames,images
                progress={'category':cat,'page':page,'pages':total,'assets_done':start+len(items),
                          'assets_total':len(entries),'seconds':round(time.time()-t,1)}
                (WORK/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2))
                print(json.dumps(progress,ensure_ascii=False),flush=True)
            process.stdin.close(); code=process.wait(); log.close()
            if code: raise RuntimeError('ffmpeg failed: '+str(code))
        except BaseException:
            process.kill(); raise
        report.append({'category':cat,'title':TITLES[cat],'file':dest.name,'assets':len(entries),'pages':total,
                       'seconds':total*DURATION,'width':W,'height':H,'fps':FPS,'frames':total*FRAMES,'bytes':dest.stat().st_size})
        (WORK/(cat+'_index.json')).write_text(json.dumps([x for x in coverage if x['category']==cat],ensure_ascii=False,indent=2))
        (WORK/(cat+'_report.json')).write_text(json.dumps(report[-1],ensure_ascii=False,indent=2))
    if not args.test:
        all_coverage=[]; all_report=[]
        for cat,_ in CATS:
            if (WORK/(cat+'_index.json')).exists():
                all_coverage+=json.loads((WORK/(cat+'_index.json')).read_text())
                all_report.append(json.loads((WORK/(cat+'_report.json')).read_text()))
        (OUT/'资产视频索引.json').write_text(json.dumps({'videos':all_report,'assets':all_coverage},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
