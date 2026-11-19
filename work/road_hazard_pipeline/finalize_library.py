from pathlib import Path
import os,json,shutil,hashlib,zipfile,subprocess,concurrent.futures
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs';LIB=OUT/'road_hazards';INST=OUT/'road_hazard_insertions'
FONT=os.environ.get("ROAD_LABEL_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BASE=['asphalt_pothole_deep','asphalt_missing_intact','asphalt_cover_raised65','concrete_pothole_deep','concrete_missing_broken','concrete_cover_offset']
LABELS={'shallow':'浅层剥落','deep':'深坑','elongated':'长条车辙坑','intact':'缺盖·混凝土井壁','broken':'缺盖·破损井沿与砖壁','raised30':'井盖翘起 30°','raised65':'井盖掀起 65°','offset':'井盖错位 28cm','leaning75':'井盖靠放 75°'}
def font(n):return ImageFont.truetype(FONT,n)
def link(p,label):return f'[{label}]({p})'
def title(cfg):return ('沥青' if cfg['kind']=='asphalt' else '水泥')+' · '+LABELS[cfg['variant']]
def instance(cfg):
    lab,sid=('B','055') if cfg['kind']=='asphalt' else ('A','056')
    return INST/f'{lab}_{sid}_{cfg["id"]}'

def verify_video(p,n,size):
    r=imageio.get_reader(p);m=r.get_meta_data();actual=r.count_frames();r.close()
    assert actual==n,(p,actual);assert tuple(m['size'])==size,(p,m['size'])
    return {'file':str(p.relative_to(OUT)),'frames':actual,'size':m['size'],'fps':m['fps']}

def asset_check(cfg):
    name=cfg['id'];d=LIB/name;g=d/'gaussians';p=PlyData.read(g/'asset.ply')['vertex'];arr=p.data.view('f4').reshape(len(p),-1)
    assert np.isfinite(arr).all(),name
    q=np.stack([p[f'rot_{i}'] for i in range(4)],1);error=float(abs(np.linalg.norm(q,axis=1)-1).max());assert error<1e-5
    xyz=np.stack([p[k] for k in ['x','y','z']],1);bounds=np.stack([xyz.min(0),xyz.max(0)])
    assert bounds[0,2]<(-.035 if cfg['type']=='pothole' else -.9)
    if cfg['type']=='cover' and cfg['angle']>20:assert bounds[1,2]>.25
    target=instance(cfg);s=json.loads((target/'summary.json').read_text());assert s.get('road_appearance_transfer');assert s['validation']['checkpoint_reload_max_rgb_error']<1e-6
    for required in ['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','f_dc_0','f_rest_8']:assert required in p.data.dtype.names
    return {'id':name,'title':title(cfg),'kind':cfg['kind'],'hazard_type':cfg['type'],'variant':cfg['variant'],'parameters':cfg,'gaussians':len(p),'bounds_m':bounds.tolist(),'quaternion_norm_max_error':error,'gaussian_ply':str((g/'asset.ply').relative_to(OUT)),'gaussian_parameters':str((g/'asset.pth').relative_to(OUT)),'preview':str((g/'preview.png').relative_to(OUT)),'scene_directory':str(target.relative_to(OUT)),'checkpoint_reload_max_rgb_error':s['validation']['checkpoint_reload_max_rgb_error'],'render_resolution':s['render_resolution'],'geometry_origin':'authored 3D hazard geometry + CC0 source materials','license':'CC0-1.0'}

def make_contacts(catalog):
    canvas=Image.new('RGB',(1800,1060),'#18202a');draw=ImageDraw.Draw(canvas);byid={c['id']:c for c in catalog}
    for i,name in enumerate(BASE):
        x=i%3*600;y=i//3*530;canvas.paste(Image.open(LIB/name/'gaussians/preview.png').resize((600,450)),(x,y+68));draw.text((x+12,y+8),title(byid[name]),font=font(26),fill='white');draw.text((x+12,y+39),'实际 3DGS 渲染',font=font(20),fill='#a5c7d8')
    canvas.save(OUT/'道路病害_六类高斯资产预览.jpg',quality=96)
    canvas=Image.new('RGB',(1800,3120),'#18202a');draw=ImageDraw.Draw(canvas)
    for i,cfg in enumerate(catalog):
        x=i%3*600;y=i//3*520;canvas.paste(Image.open(LIB/cfg['id']/'gaussians/preview.png').resize((600,450)),(x,y+64))
        draw.text((x+10,y+7),f'{i+1:02d} · '+title(cfg),font=font(26),fill='white')
        detail=f"坑口约 {2*cfg['rx']:.2f}×{2*cfg['ry']:.2f}m，深 {cfg['depth']*100:.1f}cm" if cfg['type']=='pothole' else f"井口直径约 0.65m，井深 {cfg['depth']:.2f}m"
        draw.text((x+10,y+38),detail,font=font(20),fill='#a5c7d8')
    canvas.save(OUT/'道路病害_18种高斯资产总览.jpg',quality=95)

def projection_center(cfg,frame):
    d=instance(cfg);p=json.loads((d/'summary.json').read_text())['assets'][0]['placement'];sid='055' if cfg['kind']=='asphalt' else '056'
    c=np.loadtxt(ROOT/'work/data/processed'/sid/'extrinsics'/f'{frame:03d}_0.txt');q=np.linalg.inv(c)@np.r_[p['center_world'],1]
    intr=np.loadtxt(ROOT/'work/data/processed'/sid/'intrinsics/0.txt');return intr[0]*q[0]/q[2]+intr[2],intr[1]*q[1]/q[2]+intr[3]

def showcase(catalog):
    byid={c['id']:c for c in catalog};writer=imageio.get_writer(str(OUT/'道路病害_高斯插入演示.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
    for index,name in enumerate(BASE):
        cfg=byid[name];d=instance(cfg);canonical=Image.open(LIB/name/'gaussians/preview.png').resize((384,288))
        for frame in range(25,40):
            im=Image.open(d/'frames'/f'{frame:03d}.jpg');canvas=Image.new('RGB',(1920,1080),'#18202a');draw=ImageDraw.Draw(canvas)
            canvas.paste(im.resize((1280,720)),(0,64));cx,cy=projection_center(cfg,frame)
            left=int(np.clip(cx-200,0,1920-400));top=int(np.clip(cy-265,0,1080-450));crop=im.crop((left,top,left+400,top+450)).resize((640,720));canvas.paste(crop,(1280,64));canvas.paste(canonical,(0,792))
            draw.text((18,12),f'{index+1}/6  '+title(cfg)+'  ·  1080p 场景高斯渲染',font=font(34),fill='white')
            draw.text((408,810),'同一场景统一高斯渲染，保留原车辆轨迹',font=font(30),fill='white')
            draw.text((408,866),'左下：独立高斯资产    右侧：行车画面局部放大',font=font(27),fill='#a5c7d8')
            draw.text((408,918),'此演示按 0.5× 播放；目录内另有原速完整视频',font=font(27),fill='#a5c7d8')
            detail=f"标称坑口 {cfg['rx']*2:.2f}×{cfg['ry']*2:.2f}m，深度 {cfg['depth']*100:.1f}cm" if cfg['type']=='pothole' else f"可见井壁与真实凹陷几何，井深设定 {cfg['depth']:.2f}m"
            draw.text((408,974),detail,font=font(27),fill='#dce9f0');arr=np.asarray(canvas);writer.append_data(arr);writer.append_data(arr)
    writer.close()

def docs(catalog,assets):
    rows=[]
    for i,(cfg,asset) in enumerate(zip(catalog,assets),1):
        d=instance(cfg);g=LIB/cfg['id']/'gaussians';detail=f"{2*cfg['rx']:.2f}×{2*cfg['ry']:.2f}m / {cfg['depth']*100:.1f}cm深" if cfg['type']=='pothole' else f"井深{cfg['depth']:.2f}m";rows.append(f"| {i:02d} | {title(cfg)} | {detail} | {asset['gaussians']:,} | {link(g/'asset.ply','高斯PLY')} | {link(d/'pure_3dgs.mp4','1080p视频')} / {link(d/'parallax_0.6m.mp4','横移')} |")
    text=f'''# 道路病害高斯资产库

已完成 **9 种结构/状态 × 2 种路面 = 18 个资产**，均已转换为实际各向异性 3DGS，并分别插入已有场景。水泥版放在 A / 056，沥青版放在 B / 055。

{link(OUT/'道路病害_六类高斯资产预览.jpg','六个基础方向近景')} · {link(OUT/'道路病害_18种高斯资产总览.jpg','全部18个实际高斯近景')} · {link(OUT/'道路病害_高斯插入演示.mp4','高斯插入与局部放大演示')}

## 多样性

- 坑洞：浅层剥落、深坑、长条车辙坑。标称坑口从 0.72×0.50m 到 1.32×0.46m，深度 4.5/9/13cm；有不规则破损边缘、断层和骨料。
- 缺盖井口：混凝土井壁与破损井沿/砖砌井壁两类，井深分别设为 1.05/1.35m，有井框、井壁、井底与内部踏步。
- 异常井盖：翘起30°、掀开65°、横向错位28cm、靠放75°。井盖包含厚度、底面与加强筋。
- 两种路面各有上述9种结构/状态。不同路面下相同状态共享宏观设计，材质及表面起伏不同；这不是18个独立实物的测量重建。

## 文件目录

| 序号 | 路面与病害 | 尺寸/深度设定 | 高斯数 | 独立资产 | 已插入场景 |
| --- | --- | --- | ---: | --- | --- |
'''+ '\n'.join(rows)+f'''

## 使用

每个 road_hazards/<ID>/gaussians 目录提供 asset.ply、asset.pth、实际高斯渲染的 preview.png 及 metadata.json。PLY 包含透明度 logit、对数尺度、WXYZ四元数和SH颜色，SH为1阶，单位米，Z-up；应使用支持3DGS的查看器，而非普通点云着色模式。

{link(OUT/'道路病害_18种高斯资产.zip','打包的18个独立高斯资产')} 包含全部PLY、参数与预览，不包含重复的场景检查点。

每个 road_hazard_insertions/<场景>_<ID> 目录包含：

- pure_3dgs.mp4：原速80帧、10fps、1920×1080行车结果。
- before_after.mp4：相同分辨率、相同相机的插入前后对比。
- parallax_0.6m.mp4：48帧、12fps、1920×1080横移检查。
- scene_with_assets_frame040.ply：已包含资产的完整高斯场景，第40帧静态快照。
- trained_model/iteration_7000.pth：包含原动态车辆及轨迹的原生检查点，配套render_config.yaml。7000是原背景的训练步数。
- *_world.ply、summary.json：场景坐标下资产及位置/颜色适配参数。

渲染时井壁、井盖底面等完整三维表面均已采样为高斯，与场景统一投影、排序和累计透明度。天空沿用原场景cubemap。资产没有用网格或固定二维图层叠加到最终画面上。

## 质量处理与来源

使用完整三维网格采样，包含竖直井壁和翻起井盖的背面；按几何射线烘焙环境遮蔽和盖板阴影。插入时清理覆盖井口的大路面高斯，补齐细路面高斯，并将原场景多视角路面颜色/纹理离线烘焙到路面高斯，避免明显方形材质边界。最终视频由1920×1080高斯光栅化直接生成，不是把低分辨率视频放大。

材质来源为 [ambientCG Asphalt031](https://ambientcg.com/view?id=Asphalt031)、[Concrete034](https://ambientcg.com/view?id=Concrete034)、[Concrete049](https://ambientcg.com/view?id=Concrete049)、[ManholeCover009](https://ambientcg.com/view?id=ManholeCover009)，均为CC0。沥青、骨料和井盖使用摄影测量来源的材质/高度；Concrete034为PBR近似材质。病害形状、井筒、底面和盖板结构由几何建模补全，尺度是建模设定，不是现场测量精度。原始多视角实拍照片未用于重新训练这些资产；也未执行StreetCrafter扩散蒸馏。

所有资产和插入参数集中在 {link(LIB/'asset_manifest.json','asset_manifest.json')}。参数化生成器支持 --catalog 自定义JSON；扩展时使用新的唯一ID，修改坑口尺寸、深度、种子、井壁状态或开盖角度。源码和验证记录位于 {link(OUT/'reproducibility/road_hazards','reproducibility/road_hazards')}。
'''
    (OUT/'道路病害资产库说明.md').write_text(text)

def main():
    catalog=json.loads((LIB/'catalog.json').read_text());assets=[asset_check(c) for c in catalog]
    (LIB/'asset_manifest.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2));make_contacts(catalog);showcase(catalog);docs(catalog,assets)
    videos=[]
    for cfg in catalog:
        d=instance(cfg);videos.extend([(d/'pure_3dgs.mp4',80,(1920,1080)),(d/'before_after.mp4',80,(3840,1080)),(d/'parallax_0.6m.mp4',48,(1920,1080))])
    with concurrent.futures.ThreadPoolExecutor(4) as pool:checks=list(pool.map(lambda t:verify_video(*t),videos))
    repro=OUT/'reproducibility/road_hazards';repro.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,repro/p.name)
    for name in ['insert_gaussian_assets.py','polish_road_hazard.py']:shutil.copy2(ROOT/'work/street_crafter'/name,repro/name)
    for name in ['gaussian_assets.py','gaussian_road.py']:shutil.copy2(ROOT/'work/asset_pipeline'/name,repro/name)
    shutil.copy2(ROOT/'work/street_crafter/street_gaussian/utils/camera_utils.py',repro/'camera_utils.py')
    shutil.copy2(ROOT/'work/road_hazard_sources/selected_sources.json',repro/'source_metadata.json')
    (repro/'validation.json').write_text(json.dumps({'asset_count':len(assets),'assets':assets,'videos':checks},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(OUT/'道路病害_18种高斯资产.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        z.write(OUT/'道路病害资产库说明.md','README.md');z.write(LIB/'asset_manifest.json','asset_manifest.json');z.write(OUT/'道路病害_18种高斯资产总览.jpg','preview.jpg')
        for cfg in catalog:
            d=LIB/cfg['id'];g=d/'gaussians'
            for name in ['asset.ply','metadata.json','preview.png']:z.write(g/name,cfg['id']+'/'+name)
            z.write(d/'metadata.json',cfg['id']+'/geometry.json')
    hashes={}
    for p in [OUT/'道路病害_18种高斯资产.zip']+[LIB/c['id']/'gaussians/asset.ply' for c in catalog]:
        with p.open('rb') as f:hashes[str(p.relative_to(OUT))]=hashlib.file_digest(f,'sha256').hexdigest()
    (repro/'sha256.json').write_text(json.dumps(hashes,indent=2));print('FINALIZED',len(assets),len(checks),'videos',flush=True)

if __name__=='__main__':main()
