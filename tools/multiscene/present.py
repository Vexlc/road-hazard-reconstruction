"""Make synchronized three-road comparisons for all fifty sampled assets."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

WORK=Path(__file__).parent
OUT=PROJECT/'outputs/道路病害_多场景迁入'
LIB=PROJECT/'outputs/road_hazard_expansion'
SITES=['034','109','110']
CATS=[('stereo','01_双目实拍坑洞'),('rgbd','02_RGB-D实拍坑洞'),('pothole','03_建模坑洞'),
      ('missing','04_缺盖窨井'),('raised','05_掀盖与移位窨井')]
COLORS=['#69d9cc','#76b3f3','#e7b875','#b99cf0','#ee98a8']
REG='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

def font(n,b=False):return ImageFont.truetype(BOLD if b else REG,n)
def writer(path,fps):return imageio.get_writer(str(path),fps=fps,codec='libx264',quality=8,macro_block_size=2,output_params=['-threads','6','-movflags','+faststart'])
def reader(path):return imageio.get_reader(str(path),input_params=['-threads','1'])
def titlebar(title,subtitle,plan,height):
    im=Image.new('RGB',(2880,height),'#101923');d=ImageDraw.Draw(im)
    d.text((24,14),title,font=font(37,True),fill='#f2f6fa')
    d.text((26,65),subtitle,font=font(22),fill='#9fb6c8')
    for j,site in enumerate(SITES):
        d.text((j*960+24,104),f'PandaSet {site}  ·  {plan["scene_titles"][site]}',font=font(27,True),fill=COLORS[j])
    return im

def category_videos(plan,models):
    results=[]
    for cat,name in CATS:
        path=OUT/(name+'_三场景迁入.mp4');count=0
        with writer(path,24) as video:
            for i,asset in enumerate(plan['selected_assets'][cat]):
                meta=models[asset]
                canvas=titlebar(f'{name[3:]}  /  随机样本 {i+1:02d} / 10',asset,plan,780)
                d=ImageDraw.Draw(canvas)
                depth=meta.get('depth_p95_m',meta.get('depth_design_m'))
                caption=f"原尺寸：{meta['width_m']*100:.1f} × {meta['length_m']*100:.1f} cm   |   {'估计深度 P95' if cat in ('stereo','rgbd') else '设计深度'}：{depth*100:.1f} cm"
                d.text((24,710),caption,font=font(25),fill='#d4e1eb')
                d.text((24,750),'同一资产 / 三个独立重建场景 / 联合 3DGS 渲染 / 每段 3 秒',font=font(20),fill='#95aabd')
                d.text((2150,750),'各格独立取景，未放大实物尺寸',font=font(20),fill='#95aabd')
                readers=[reader(OUT/'迁入场景'/f'{s}_{i+1:02d}'/f'{cat}_close.mp4') for s in SITES]
                frames=0
                for group in zip(*readers):
                    frame=canvas.copy()
                    for j,picture in enumerate(group):frame.paste(Image.fromarray(picture),(j*960,154))
                    video.append_data(np.asarray(frame));frames+=1;count+=1
                    if i==0 and frames==36:frame.save(OUT/(name+'_迁入预览.jpg'),quality=94)
                for r in readers:r.close()
                assert frames==72,(cat,i,frames)
                print('COMPOSE',cat,i+1,flush=True)
        results.append({'category':cat,'file':path.name,'assets':10,'scenes':3,'placements':30,
                        'frames':count,'fps':24,'seconds':count/24,'resolution':[2880,780]})
    return results

def projection(data,frame,point):
    pose=np.loadtxt(data/'extrinsics'/f'{frame:03d}_0.txt')
    k=np.loadtxt(data/'intrinsics/0.txt').ravel()
    q=(np.array(point)-pose[:3,3])@pose[:3,:3]
    return [q[0]*k[0]/q[2]+k[2],q[1]*k[1]/q[2]+k[3],q[2]]

def grid_videos(plan,models):
    """Keep the requested 4x4 overview, with ten sampled assets on each road page."""
    results=[]
    for cat,name in CATS:
        path=OUT/(name+'_4x4迁入总览.mp4');total=0
        with writer(path,24) as video:
            for site in SITES:
                base=Image.new('RGB',(2560,1920),'#101923');d=ImageDraw.Draw(base)
                d.text((24,22),name[3:]+' / 4 × 4 迁入总览',font=font(42,True),fill='#f0f5f9')
                d.text((26,82),f'PandaSet {site} · {plan["scene_titles"][site]} · 五类各随机抽样 10 个，本页展示本类 10 个',font=font(25),fill='#a9c3d4')
                ids=plan['selected_assets'][cat]
                for slot in range(16):
                    x=24+(slot%4)*632;y=124+(slot//4)*436
                    d.rectangle((x,y,x+615,y+419),fill='#1b2733')
                    if slot>=10:
                        d.text((x+289,y+180),'—',font=font(32),fill='#3a4b5a');continue
                    m=models[ids[slot]]
                    d.text((x+12,y+7),f'随机样本 {slot+1:02d} / 10',font=font(22,True),fill='#edf4f8')
                    depth=m.get('depth_p95_m',m.get('depth_design_m'))
                    label=f"{m['width_m']*100:.1f} × {m['length_m']*100:.1f} cm  |  {'深度 P95 ≈' if cat in ('stereo','rgbd') else '设计深度'} {depth*100:.1f} cm"
                    d.text((x+12,y+374),label,font=font(18),fill='#c0cfdb')
                    d.text((x+12,y+402),m['id'],font=font(13),fill='#8ca4b7')
                d.text((24,1878),'10 个唯一资产 / 保持原尺寸 / 每条道路一页 / 其余格留空',font=font(21),fill='#a4bbcc')
                readers=[reader(OUT/'迁入场景'/f'{site}_{i+1:02d}'/f'{cat}_close.mp4') for i in range(10)]
                count=0
                for pictures in zip(*readers):
                    im=base.copy()
                    for slot,picture in enumerate(pictures):
                        # Preserve the 16:9 camera view inside each grid cell.
                        image=Image.fromarray(picture).resize((586,330),Image.Resampling.LANCZOS)
                        x=24+(slot%4)*632+15;y=124+(slot//4)*436+40
                        im.paste(image,(x,y))
                    video.append_data(np.asarray(im));count+=1;total+=1
                    if count==36:im.save(OUT/(name+f'_4x4迁入_{site}.jpg'),quality=94)
                for r in readers:r.close()
                assert count==72
                print('GRID COMPOSE',cat,site,flush=True)
        results.append({'category':cat,'kind':'grid_overview','file':path.name,'assets':10,'scenes':3,'placements':30,
                        'frames':total,'fps':24,'seconds':total/24,'resolution':[2560,1920]})
    return results

def drive_comparison(plan):
    path=OUT/'三种道路_迁入前后.mp4';total=0
    with writer(path,10) as video:
        for batch in range(1,11):
            canvas=titlebar(f'道路病害迁入前后  /  随机组 {batch:02d} / 10',
                            '上：迁入前    下：迁入后    |    每组五类各 1 个，共 50 个资产 × 3 条道路',plan,1260)
            legend=ImageDraw.Draw(canvas)
            legend.text((24,1237),'位置编号：1 双目坑洞    2 RGB-D 坑洞    3 建模坑洞    4 缺盖窨井    5 掀盖与移位窨井',font=font(18),fill='#a7bdcd')
            readers=[reader(OUT/'迁入场景'/f'{s}_{batch:02d}'/'行车迁入前后.mp4') for s in SITES]
            frames=0
            for pictures in zip(*readers):
                im=canvas.copy()
                for j,picture in enumerate(pictures):
                    resized=Image.fromarray(picture).resize((960,1080),Image.Resampling.LANCZOS)
                    im.paste(resized,(j*960,154))
                d=ImageDraw.Draw(im)
                for j,site in enumerate(SITES):
                    data=WORK/'data/processed'/site
                    for slot,rec in enumerate(plan['slots_by_site'][site]):
                        x,y,z=projection(data,plan['drive_start_by_site'][site]+frames,rec['center_world'])
                        if not (z>1 and 0<x<1920 and 0<y<1080):continue
                        x=j*960+x*.5;y=154+y*.5
                        for delta in (0,540):
                            d.ellipse((x-5,y+delta-5,x+5,y+delta+5),fill=COLORS[slot],outline='#121925',width=1)
                            d.text((x+7,y+delta-15),str(slot+1),font=font(20,True),fill=COLORS[slot],stroke_width=2,stroke_fill='#0a1016')
                video.append_data(np.asarray(im));frames+=1;total+=1
                if batch==3 and frames==15:im.save(OUT/'三种道路_迁入前后.jpg',quality=94)
            for r in readers:r.close()
            assert frames==40
            print('DRIVE COMPOSE',batch,flush=True)
    return {'file':path.name,'frames':total,'fps':10,'seconds':total/10,'resolution':[2880,1260]}

def reconstruction_comparison(plan):
    path=OUT/'三种道路_原图与重建对照.mp4'
    canvas=titlebar('三个新增道路场景 / 原图与 3DGS 重建',
                    '上：原始采集图像    下：重建渲染    |    每段 80 帧，7,000 轮独立训练',plan,1260)
    readers=[reader(OUT/'重建场景'/s/'原图与重建对照.mp4') for s in SITES]
    count=0
    with writer(path,10) as video:
        for pictures in zip(*readers):
            im=canvas.copy()
            for j,picture in enumerate(pictures):
                im.paste(Image.fromarray(picture[:,:1280]).resize((960,540),Image.Resampling.LANCZOS),(960*j,154))
                im.paste(Image.fromarray(picture[:,1280:]).resize((960,540),Image.Resampling.LANCZOS),(960*j,694))
            video.append_data(np.asarray(im));count+=1
            if count==40:im.save(OUT/'三种道路_原图与重建对照.jpg',quality=94)
    for r in readers:r.close()
    assert count==80
    return {'file':path.name,'frames':count,'fps':10,'seconds':count/10,'resolution':[2880,1260]}

def main():
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((LIB/'asset_manifest.json').read_text(encoding='utf-8'))}
    results=category_videos(plan,models)
    results.extend(grid_videos(plan,models))
    results.append(drive_comparison(plan));results.append(reconstruction_comparison(plan))
    (OUT/'可视化视频索引.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PRESENTATION COMPLETE',flush=True)

if __name__=='__main__':main()
