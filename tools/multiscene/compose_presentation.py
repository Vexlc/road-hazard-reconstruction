"""Compose asset orbit + matched global pair + matched local pair in one frame."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import time
from functools import lru_cache

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

WORK=Path(__file__).parent
ROOT=PROJECT/'outputs/道路病害_多场景迁入'
OUT=ROOT/'环绕与迁入前后综合展示'
CACHE=WORK/'composite3'
LIB=PROJECT/'outputs/road_hazard_expansion'
CATS=[('stereo','01_双目实拍坑洞'),('rgbd','02_RGB-D实拍坑洞'),('pothole','03_建模坑洞'),
      ('missing','04_缺盖窨井'),('raised','05_掀盖与移位窨井')]
SITES=['034','109','110']
REG='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
W,H=2560,1440
FRAMES=96

@lru_cache(maxsize=32)
def font(size,bold=False):return ImageFont.truetype(BOLD if bold else REG,size)
def read_frames(path,size):
    reader=imageio.get_reader(str(path),size=size,input_params=['-threads','1'])
    result=[Image.fromarray(f) for f in reader];reader.close()
    return result

def global_picture(path,uv):
    picture=Image.open(path).convert('RGB').resize((864,486),Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(picture);x,y=uv[0]*864/1280,uv[1]*486/720
    assert 0<x<864 and 0<y<486,(path,uv)
    color='#ffd06e';radius=27;arm=11
    for sx in (-1,1):
        for sy in (-1,1):
            px=x+sx*radius;py=y+sy*radius
            d.line((px,py,px-sx*arm,py),fill=color,width=3)
            d.line((px,py,px,py-sy*arm),fill=color,width=3)
    d.text((max(6,min(x-35,750)),y-58 if y>415 else y+32),'当前资产',font=font(18,True),fill=color,stroke_width=2,stroke_fill='#111923')
    return picture

def base_frame(cat,title,asset,index,site,plan,meta,view,dest):
    im=Image.new('RGB',(W,H),'#101923');d=ImageDraw.Draw(im)
    d.text((24,18),f'{title}  /  样本 {index+1:02d} / 10',font=font(39,True),fill='#f3f7fa')
    d.text((24,75),asset,font=font(25),fill='#b7cada')
    d.text((24,116),f'PandaSet {site} · {plan["scene_titles"][site]}',font=font(24,True),fill='#79c8d6')
    d.text((1980,26),f'道路 {SITES.index(site)+1} / 3   ·   每段 4 秒',font=font(25),fill='#b7cada')
    d.rounded_rectangle((16,160,783,1352),radius=12,fill='#182532')
    d.text((38,184),'资产 360° 环绕',font=font(32,True),fill='#86dcce')
    d.text((38,231),'独立资产视角',font=font(23),fill='#a8bfce')
    depth=meta.get('depth_p95_m',meta.get('depth_design_m'))
    d.text((38,930),f"原尺寸  {meta['width_m']*100:.1f} × {meta['length_m']*100:.1f} cm",font=font(28),fill='#d7e4ec')
    d.text((38,977),f"{'深度 P95 ≈' if cat in ('stereo','rgbd') else '设计深度'} {depth*100:.1f} cm",font=font(25),fill='#aabfce')
    d.text((38,1065),'右上：在道路中的位置与迁入效果',font=font(23),fill='#acc3d2')
    d.text((38,1110),'右下：同相机轨迹的局部前后对照',font=font(23),fill='#acc3d2')
    d.text((38,1198),'原始资产尺寸保持不变',font=font(23),fill='#86dcce')
    for x,y,label,color in [(800,160,'全局 · 迁入前','#a5c7dc'),(1680,160,'全局 · 迁入后','#eab67b'),
                            (800,800,'局部 · 迁入前','#a5c7dc'),(1680,800,'局部 · 迁入后','#eab67b')]:
        d.rounded_rectangle((x,y,x+863,y+547),radius=10,fill='#1b2936')
        d.text((x+14,y+10),label,font=font(28,True),fill=color)
    im.paste(global_picture(dest/f'{cat}_global_before.jpg',view['global_center_pixels']),(800,208))
    im.paste(global_picture(dest/f'{cat}_global_after.jpg',view['global_center_pixels']),(1680,208))
    d.text((800,725),f'全局与局部前后均使用相同相机和时间；全局参考帧 {view["frame"]:02d}',font=font(22),fill='#9fb8ca')
    d.text((24,1391),'定位框对应当前资产；迁入后保留同组其余资产。各面板独立取景。',font=font(21),fill='#a4bbcc')
    return im

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((LIB/'asset_manifest.json').read_text(encoding='utf-8'))}
    reports=[];segments=[]
    for cat,name in CATS:
        path=OUT/(name+'_环绕与迁入前后.mp4');count=0;start=time.time()
        with imageio.get_writer(str(path),fps=24,codec='libx264',quality=None,macro_block_size=2,
             output_params=['-preset','veryfast','-crf','21','-threads','8','-movflags','+faststart']) as video:
            for index,asset in enumerate(plan['selected_assets'][cat]):
                orbit=read_frames(CACHE/'orbits'/(asset+'.mp4'),(760,570));assert len(orbit)==96
                for site in SITES:
                    dest=CACHE/'views'/f'{site}_{index+1:02d}'
                    record=json.loads((dest/'complete.json').read_text(encoding='utf-8'))
                    view=next(v for v in record['views'] if v['category']==cat)
                    assert view['asset_id']==asset
                    before=read_frames(dest/f'{cat}_local_before.mp4',(864,486))
                    after_path=ROOT/'迁入场景'/f'{site}_{index+1:02d}'/f'{cat}_close.mp4'
                    after=read_frames(after_path,(864,486));assert len(before)==len(after)==72
                    base=base_frame(cat,name[3:],asset,index,site,plan,models[asset],view,dest)
                    t0=count/24
                    for i in range(96):
                        image=base.copy();j=min(71,int(i*72/96))
                        image.paste(orbit[i],(20,312));image.paste(before[j],(800,848));image.paste(after[j],(1680,848))
                        draw=ImageDraw.Draw(image)
                        draw.text((45,879),f'环绕角度  {i*360/96:05.1f}°',font=font(23),fill='#a8c9d7')
                        video.append_data(np.asarray(image));count+=1
                        if index==0 and site=='110' and i==48:image.save(OUT/(name+'_版式预览.jpg'),quality=95)
                    segments.append({'file':path.name,'category':cat,'asset_id':asset,'sample':index+1,'scene':site,
                         'start_seconds':t0,'end_seconds':count/24,'global_reference_frame':view['frame'],
                         'source_after_checkpoint':str(ROOT/'迁入场景'/f'{site}_{index+1:02d}'/'trained_model/iteration_7000.pth'),
                         'full_orbit_degrees':360,'matched_global_camera':True,'matched_local_camera_sweep':True})
                    print('COMPOSE',cat,index+1,site,'elapsed',round(time.time()-start,1),flush=True)
                del before,after,orbit
        reports.append({'file':path.name,'category':cat,'assets':10,'scenes':3,'segments':30,'frames':count,
                        'fps':24,'seconds':count/24,'resolution':[W,H]})
        (CACHE/f'{cat}_composition_complete.json').write_text(json.dumps(reports[-1],indent=2),encoding='utf-8')
    (OUT/'视频与资产索引.json').write_text(json.dumps({'layout':['asset_360_orbit','global_before','global_after',
           'local_before','local_after'],'videos':reports,'segments':segments},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ALL COMPOSITIONS COMPLETE',flush=True)

if __name__=='__main__':main()
