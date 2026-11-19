"""Show the genuinely enlarged stereo geometry with native driving comparisons."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import ImageDraw
import torch
import compose_dynamic_presentation as p

WORK=p.WORK
OUT=p.c.ROOT/'双目坑洞_几何放大10倍'
CACHE=WORK/'enlarged10'
sys.path.insert(0,str(WORK.parent/'visualization'))
import render_catalog as g
g.IW,g.IH,g.FRAMES=760,570,96

@torch.no_grad()
def orbit(meta):
    path=CACHE/'orbits'/(meta['id']+'.mp4');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return path
    state,_,_=g.load_asset(meta);state['means']*=10;state['scales']*=10
    xyz=state['means'].cpu().numpy()
    support=xyz[(abs(xyz[:,0])<meta['width_m']*.78)&(abs(xyz[:,1])<meta['length_m']*.78)]
    low,high=np.quantile(support[:,2],[.005,.995]);center=np.array([0.,0.,(low+high)*.5])
    support=support[::max(1,len(support)//4000)];views,k=g.cameras(center,support,'stereo')
    with imageio.get_writer(str(path),fps=24,codec='libx264',quality=8,macro_block_size=2,
         output_params=['-threads','2','-movflags','+faststart']) as video:
        for start in range(0,96,8):
            rgb,_,_=g.rasterization(**state,viewmats=views[start:start+8],Ks=k.expand(8,-1,-1),width=760,height=570,
                 sh_degree=1,packed=False,backgrounds=g.tensor([[.23,.27,.30]]).expand(8,-1),rasterize_mode='antialiased')
            assert bool(torch.isfinite(rgb).all())
            for frame in (rgb.clamp(0,1)*255+.5).to(torch.uint8).cpu().numpy():video.append_data(frame)
    return path

def main():
    torch.set_num_threads(4)
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models=json.loads((OUT/'放大资产尺寸.json').read_text(encoding='utf-8'))
    calibration={}
    for site in p.c.SITES:
        data=WORK/'data/processed'/site
        calibration[site]=([np.loadtxt(data/'extrinsics'/f'{i:03d}_0.txt') for i in range(80)],np.loadtxt(data/'intrinsics/0.txt').ravel())
    path=OUT/'双目来源坑洞_几何放大10倍_逐帧前后对照.mp4';count=0;segments=[]
    with imageio.get_writer(str(path),fps=24,codec='libx264',quality=None,macro_block_size=2,
         output_params=['-preset','veryfast','-crf','20','-threads','8','-movflags','+faststart']) as video:
        for sample,asset in enumerate(plan['selected_assets']['stereo']):
            meta=models[asset];turn=p.c.read_frames(orbit(meta),(760,570));assert len(turn)==96
            for site in p.c.SITES:
                key=f'{site}_{sample+1:02d}';folder=OUT/'迁入场景'/key
                summary=json.loads((folder/'scene.json').read_text(encoding='utf-8'));assert summary['stereo_geometry_scale']==10
                d=summary['placements'][0];assert d['asset_id']==asset and d['size_multiplier']==10
                pair=p.c.read_frames(folder/'行车迁入前后.mp4',(864,972));assert len(pair)==40
                before=p.c.read_frames(folder/'stereo_local_before.mp4',(864,486));after=p.c.read_frames(folder/'stereo_close.mp4',(864,486))
                assert len(before)==len(after)==72
                old=p.c.CACHE/'views'/key;views=json.loads((old/'complete.json').read_text(encoding='utf-8'))
                view=next(x for x in views['views'] if x['category']=='stereo')
                base=p.c.base_frame('stereo','双目来源坑洞 · 几何放大10×',asset,sample,site,plan,meta,view,old)
                draw=ImageDraw.Draw(base)
                draw.rectangle((24,175,770,280),fill='#182532')
                draw.text((38,184),'放大坑洞 360° 环绕',font=p.c.font(31,True),fill='#86dcce')
                draw.text((38,234),'三维几何统一放大 10×',font=p.c.font(24,True),fill='#f0bd77')
                draw.rectangle((24,915,770,1325),fill='#182532')
                draw.text((38,930),f"原坑口 {meta['original_width_m']*100:.1f} × {meta['original_length_m']*100:.1f} cm",font=p.c.font(25),fill='#a9bfce')
                draw.text((38,980),f"放大后 {meta['width_m']:.2f} × {meta['length_m']:.2f} m",font=p.c.font(31,True),fill='#ffd06e')
                draw.text((38,1040),f"深度 {meta['original_depth_p95_m']*100:.1f} → {meta['depth_p95_m']*100:.1f} cm",font=p.c.font(24),fill='#c7d8e4')
                draw.text((38,1118),'坑口、深度均按原模型放大 10 倍',font=p.c.font(23),fill='#a9bfce')
                draw.text((38,1190),'演示放大，非原实测尺寸',font=p.c.font(25,True),fill='#f0bd77')
                draw.rectangle((800,710,2545,782),fill='#101923')
                draw.text((800,723),'全局前后：同步逐帧行车；坑洞已实际放大到场景中',font=p.c.font(23,True),fill='#e7bd81')
                draw.rectangle((24,1380,2545,1439),fill='#101923')
                draw.text((24,1391),'演示放大版：坑洞几何实际放大10倍；原实拍资产和原尺寸版本均保留。',font=p.c.font(21),fill='#b3c7d5')
                start=plan['drive_start_by_site'][site];t0=count/24
                for i in range(96):
                    j=i*40//96;frame=start+j;point=p.project(site,frame,d['center_world'],calibration)
                    whole=pair[j]
                    first=p.decorate_global(whole.crop((0,0,864,486)),point,False)
                    second=p.decorate_global(whole.crop((0,486,864,972)),point,False)
                    image=base.copy();image.paste(turn[i],(20,312));image.paste(first,(800,208));image.paste(second,(1680,208))
                    image.paste(before[i*72//96],(800,848));image.paste(after[i*72//96],(1680,848))
                    text=ImageDraw.Draw(image);text.text((45,879),f'环绕角度 {i*360/96:05.1f}°',font=p.c.font(23),fill='#a8c9d7')
                    text.text((800,759),f'行车帧 {frame:02d}/{start+39:02d} · 原序列 10 fps',font=p.c.font(20),fill='#86dcce')
                    video.append_data(np.asarray(image));count+=1
                    if sample==0 and site=='034' and i==24:image.save(OUT/'放大10倍_展示预览.jpg',quality=95)
                segments.append({'asset_id':asset,'scene':site,'file':path.name,'start_seconds':t0,'end_seconds':count/24,
                     'geometry_scale':10,'original_width_m':meta['original_width_m'],'original_length_m':meta['original_length_m'],
                     'display_width_m':meta['width_m'],'display_length_m':meta['length_m'],
                     'global_frame_indices':list(range(start,start+40)),'checkpoint':summary['checkpoint']})
                print('COMPOSE ENLARGED',sample+1,site,flush=True)
    (OUT/'视频与资产索引.json').write_text(json.dumps({'file':path.name,'frames':count,'fps':24,'seconds':count/24,
         'resolution':[2560,1440],'geometry_scale':10,'segments':segments},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ENLARGED COMPOSITION COMPLETE',flush=True)

if __name__=='__main__':main()
