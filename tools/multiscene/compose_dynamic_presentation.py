"""Use real consecutive driving frames and stronger stereo inspection views."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import time
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import compose_presentation as c

WORK=c.WORK
OUT=c.ROOT/'逐帧行车与双目增强版'
CACHE=WORK/'composite4'

def project(site,frame,center,calibration):
    pose,k=calibration[site][0][frame],calibration[site][1]
    q=(np.array(center)-pose[:3,3])@pose[:3,:3]
    if q[2]<=.01:return None
    uv=np.array([q[0]*k[0]/q[2]+k[2],q[1]*k[1]/q[2]+k[3]])
    x,y=uv*np.array([864/1920,486/1080])
    return (float(x),float(y)) if 0<x<864 and 0<y<486 else None

def decorate_global(picture,point,stereo):
    picture=picture.copy();raw=picture.copy();d=ImageDraw.Draw(picture)
    if stereo:
        d.rounded_rectangle((8,8,220,253),radius=7,fill='#101923',outline='#ffd06e',width=2)
        d.text((19,17),'8× 跟随放大',font=c.font(20,True),fill='#ffd06e')
        if point:
            x,y=point;cx=int(np.clip(round(x),12,851));cy=int(np.clip(round(y),12,473))
            crop=raw.crop((cx-12,cy-12,cx+13,cy+13)).resize((200,200),Image.Resampling.BICUBIC)
            picture.paste(crop,(14,47));d=ImageDraw.Draw(picture)
        else:d.text((27,126),'坑洞不在画内',font=c.font(20),fill='#c1d1dc')
    if point:
        x,y=point;radius=28;arm=12
        for sx in (-1,1):
            for sy in (-1,1):
                px=x+sx*radius;py=y+sy*radius
                d.line((px,py,px-sx*arm,py),fill='#ffd06e',width=3)
                d.line((px,py,px,py-sy*arm),fill='#ffd06e',width=3)
        d.text((max(4,min(754,x-38)),y-58 if y>414 else y+32),'当前资产',font=c.font(18,True),fill='#ffd06e',stroke_width=2,stroke_fill='#111923')
    return picture

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((c.LIB/'asset_manifest.json').read_text(encoding='utf-8'))}
    calibration={}
    for site in c.SITES:
        data=WORK/'data/processed'/site
        calibration[site]=([np.loadtxt(data/'extrinsics'/f'{i:03d}_0.txt') for i in range(80)],np.loadtxt(data/'intrinsics/0.txt').ravel())
    videos=[];segments=[]
    for cat,name in c.CATS:
        path=OUT/(name+'_逐帧行车与局部对照.mp4');count=0;started=time.time()
        with imageio.get_writer(str(path),fps=24,codec='libx264',quality=None,macro_block_size=2,
             output_params=['-preset','veryfast','-crf','20','-threads','8','-movflags','+faststart']) as video:
            for sample,asset in enumerate(plan['selected_assets'][cat]):
                orbitpath=(CACHE/'orbits' if cat=='stereo' else c.CACHE/'orbits')/(asset+'.mp4')
                while not orbitpath.exists():time.sleep(2)
                orbit=c.read_frames(orbitpath,(760,570));assert len(orbit)==96
                for site in c.SITES:
                    key=f'{site}_{sample+1:02d}';source=c.ROOT/'迁入场景'/key
                    old=c.CACHE/'views'/key
                    rec=json.loads((old/'complete.json').read_text(encoding='utf-8'))
                    view=next(v for v in rec['views'] if v['category']==cat)
                    pair=c.read_frames(source/'行车迁入前后.mp4',(864,972));assert len(pair)==40
                    if cat=='stereo':
                        detail=CACHE/'stereo'/key
                        while not (detail/'complete.json').exists():time.sleep(2)
                        before=c.read_frames(detail/'before.mp4',(864,486));after=c.read_frames(detail/'after.mp4',(864,486))
                        assert len(before)==len(after)==96
                    else:
                        before=c.read_frames(old/f'{cat}_local_before.mp4',(864,486))
                        after=c.read_frames(source/f'{cat}_close.mp4',(864,486));assert len(before)==len(after)==72
                    base=c.base_frame(cat,name[3:],asset,sample,site,plan,models[asset],view,old)
                    draw=ImageDraw.Draw(base)
                    draw.rectangle((800,710,2545,781),fill='#101923')
                    draw.text((800,723),'全局：同一连续行车序列逐帧同步播放；局部：同一检查轨迹前后对照',font=c.font(22),fill='#a7c4d4')
                    draw.rectangle((24,1381,2545,1439),fill='#101923')
                    note='双目坑洞：仅放大观察视角和图像窗口，实物尺寸未改变。' if cat=='stereo' else '全局逐帧定位当前资产；迁入后保留同组其余资产。'
                    draw.text((24,1391),note,font=c.font(21),fill='#a4bbcc')
                    if cat=='stereo':
                        draw.rectangle((24,175,770,278),fill='#182532')
                        draw.text((38,184),'坑洞放大环绕 360°',font=c.font(32,True),fill='#86dcce')
                        draw.text((38,234),'收近镜头 · 保持实物尺寸',font=c.font(23),fill='#acc3d2')
                    center=plan['slots_by_site'][site][plan['category_order'].index(cat)]['center_world']
                    start=plan['drive_start_by_site'][site];t0=count/24
                    visible_frames=0
                    for i in range(96):
                        j=i*40//96;native_frame=start+j
                        point=project(site,native_frame,center,calibration)
                        if point:visible_frames+=1
                        whole=pair[j]
                        first=decorate_global(whole.crop((0,0,864,486)),point,cat=='stereo')
                        second=decorate_global(whole.crop((0,486,864,972)),point,cat=='stereo')
                        local=i if cat=='stereo' else i*72//96
                        image=base.copy();image.paste(orbit[i],(20,312));image.paste(first,(800,208));image.paste(second,(1680,208))
                        image.paste(before[local],(800,848));image.paste(after[local],(1680,848))
                        draw=ImageDraw.Draw(image)
                        draw.text((45,879),f'环绕角度 {i*360/96:05.1f}°',font=c.font(23),fill='#a8c9d7')
                        draw.text((800,759),f'行车帧 {native_frame:02d} / {start+39:02d}   ·   原序列 10 fps',font=c.font(20),fill='#86dcce')
                        video.append_data(np.asarray(image));count+=1
                        if sample==0 and site=='034' and i==24:image.save(OUT/(name+'_新版预览.jpg'),quality=95)
                    segments.append({'category':cat,'asset_id':asset,'scene':site,'file':path.name,'start_seconds':t0,
                         'end_seconds':count/24,'global_native_frame_indices':list(range(start,start+40)),
                         'global_source_fps':10,'global_before_after_synchronized':True,'scale_multiplier':1.,
                         'tracking_visible_output_frames':visible_frames,'stereo_tracking_zoom':8 if cat=='stereo' else None,
                         'stereo_close_camera':cat=='stereo'})
                    print('DYNAMIC COMPOSE',cat,sample+1,site,round(time.time()-started,1),flush=True)
                del orbit,before,after,pair
        videos.append({'category':cat,'file':path.name,'frames':count,'fps':24,'seconds':count/24,'resolution':[2560,1440],'assets':10,'scenes':3})
    (OUT/'视频与资产索引.json').write_text(json.dumps({'videos':videos,'segments':segments},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('DYNAMIC PRESENTATION COMPLETE',flush=True)

if __name__=='__main__':main()
