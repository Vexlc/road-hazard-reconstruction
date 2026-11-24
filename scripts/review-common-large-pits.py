"""Verify and assemble the twenty dry-pit models, their inputs and the matched road replay."""
import argparse
import json
import os
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gen2hole.large_pits.audit import camera_summary, check_movie, read, review, sha256

OUTPUT = ROOT/'outputs/photo-large-potholes'
COLLECTION = OUTPUT/'common-20'
RECIPE = ROOT/'gen2hole/examples/oblique-road-pits/common-large'
NEW = [c['id'] for c in read(RECIPE/'cases.json')]
KEYS = NEW + ['P190', 'P198', 'P199', 'P200', 'P201']
SIZE = (1600, 800)
PANEL = (1110, 562)


def locations(key):
    if key == 'P190':
        return ROOT/'runs/large-pits-v3/series', OUTPUT/'P190-v3/scene-109', ROOT/'work/large-pits-v3/inputs'
    if key in NEW:
        return ROOT/'runs/common-large-pits'/key/'series', COLLECTION/key/'scene-109', ROOT/'work/common-large-20/inputs'/key
    return ROOT/'runs/lane-pits-batch'/key/'series', OUTPUT/'lane-batch'/key/'scene-109', ROOT/'work/large-pits-extra/inputs'/key


def verify(key):
    bundle, folder, _ = locations(key)
    meta, = read(bundle/'assets/asset_manifest.json')
    native = (bundle/'assets'/meta['glb']).parent
    checks = read(bundle/'checks.json')[key]
    record = read(folder/'placement.json')
    if meta['id'] != key or record['asset'] != key or record['site'] != '109':
        raise ValueError(f'{key}: asset or site mismatch')
    for filename, digest in checks['sha256'].items():
        if sha256(native/filename) != digest:
            raise ValueError(f'{key}: different native {filename}')
    for ext in ('ply', 'glb'):
        if checks['sha256']['asset.'+ext] != record[ext+'_sha256']:
            raise ValueError(f'{key}: different model in scene')
    if sha256(bundle/'scenes'/record['case']/'trained_model/iteration_7000.pth') != record['source_checkpoint_sha256']:
        raise ValueError(f'{key}: different scene checkpoint')
    cameras = camera_summary(record)
    if (record['size_multiplier'] != 1 or record['reload_max_pixel_error'] or
            any(c['outside_edits'] or c['protected_edits'] for c in cameras.values())):
        raise ValueError(f'{key}: scale, reload or protected pixels failed')
    ref = COLLECTION/key/'reference'
    ref.mkdir(parents=True, exist_ok=True)
    projection = review.source_check(bundle, meta, ref, context=False, detailed=True)
    data = {k: meta[k] for k in ('label','oriented_extent_m','outline_area_m2','depth_p95_m','source_sha256')}
    data.update(reference_kind=meta.get('reference_kind','photograph'),
                geometry_sha256=checks['geometry_sha256'], native_hashes=checks['sha256'],
                source_check=projection, cameras=cameras, movie=check_movie(folder/'drive.mp4',record),
                source=(bundle/meta['source_image']).relative_to(ROOT).as_posix(),
                model=(native/'asset.glb').relative_to(ROOT).as_posix(),
                drive=(folder/'drive.mp4').relative_to(ROOT).as_posix())
    return record, data


def photo(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def contact_sheet(data):
    tile = (450, 298)
    board = Image.new('RGB', (1850, 1868), '#f5f3ed')
    review.label(board, (18, 14), '20 个车行道坑洞', 33, True)
    for i,key in enumerate(KEYS):
        x,y = 14+(i%4)*458, 72+(i//4)*350
        im = photo(locations(key)[0]/'views'/(key+'_inspection.jpg'))
        board.paste(ImageOps.fit(im,tile,method=Image.Resampling.LANCZOS), (x,y))
        review.label(board,(x,y+307),key+'  '+data[key]['label'],21,True)
    review.label(board,(16,1832),'P205–P219 为本批新增    ·    原有五件列于最后    ·    每格展示独立网格的斜视图',19)
    board.save(COLLECTION/'overview.jpg',quality=94,subsampling=0)


def shot_crop(records, name, number):
    boxes = np.array([next(r['bbox'] for r in p['frames'][name] if r['frame']==number) for p in records.values()])*.75
    lo, hi = boxes[:,:2].min(0), boxes[:,2:].max(0)
    w = min(1440, max((hi[0]-lo[0])*1.20, (hi[1]-lo[1])*PANEL[0]/PANEL[1]*1.20))
    h = w*PANEL[1]/PANEL[0]
    x = np.clip((hi[0]+lo[0]-w)*.5,0,1440-w)
    y = np.clip((hi[1]+lo[1]-h)*.5,75,810-h)
    return np.array([x,y,x+w,y+h])


def card(key, data, ordinal):
    bundle, _, _ = locations(key)
    page = Image.new('RGB',SIZE,'#f5f3ed')
    review.label(page,(24,20),f"{key}   {data['label']}",37,True)
    review.label(page,(26,75),f"坑口 {data['outline_area_m2']:.2f} m²     长轴 {data['oriented_extent_m'][0]:.2f} m     凹深 P95  {data['depth_p95_m']*100:.1f} cm",21)
    review.label(page,(1350,35),f'{ordinal:02d}  /  20',28,True)
    kind='生成参考图' if data['reference_kind']=='generated' else '实拍参考图'
    review.label(page,(24,126),kind,20,True)
    review.label(page,(24,426),'三维模型',20,True)
    # The checked image-space crop is essential for the small opening in the P198 street photograph.
    ref=photo(COLLECTION/key/'reference'/(key+'-source.jpg'))
    view=photo(bundle/'views'/(key+'_inspection.jpg'))
    for im, top in [(ref,156),(view,456)]:
        tile=ImageOps.contain(im,(426,252),method=Image.Resampling.LANCZOS)
        page.paste(tile,(24+(426-tile.width)//2,top+(252-tile.height)//2))
    review.label(page,(473,126),'PandaSet 109   ·   实拍车道',20,True)
    if key=='P198':
        note='参考照片 © Jennifer Petrie / Geograph · CC BY-SA 2.0'
    elif key=='P190':
        note='用户提供的道路照片    ·    坑口尺度由单张照片估计'
    else:
        note='ImageGen 参考图    ·    模型采用设计尺度'
    review.label(page,(24,750),note,17)
    return page


def make_movie(records, data):
    # Select common frame numbers in both camera tracks once for the entire collection.
    common = {}
    for name in ('front_camera','back_camera'):
        common[name]=sorted(set.intersection(*[{r['frame'] for r in rec['frames'][name]} for rec in records.values()]))
    front=common['front_camera'][-14:]
    back=common['back_camera'][:9]
    if len(front)!=14 or len(back)!=9:
        raise ValueError('Insufficient shared camera coverage')
    full=np.array([0,810-1440*PANEL[1]/PANEL[0],1440,810])
    close=shot_crop(records,'front_camera',front[-1])
    stage=COLLECTION/'common-20.stage.mp4'
    frames=0
    with imageio.get_writer(str(stage),fps=10,codec='libx264',quality=8,macro_block_size=2,
                            output_params=['-threads','4','-movflags','+faststart']) as writer:
        for i,key in enumerate(KEYS):
            page=card(key,data[key],i+1)
            cap=cv2.VideoCapture(str(ROOT/data[key]['drive']))
            for name,selection in [('front_camera',front),('back_camera',back)]:
                track=records[key]['frames'][name]
                offset=0 if name=='front_camera' else len(records[key]['frames']['front_camera'])+33
                for number in selection:
                    index=offset+next(j for j,r in enumerate(track) if r['frame']==number)
                    if round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=index:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                    ok,bgr=cap.read()
                    if not ok:
                        raise ValueError(f'Cannot decode {key} frame {number}')
                    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                    zooms=[0.]
                    if name=='front_camera' and number==front[-1]:
                        zooms += list(np.linspace(.1,1.,8)) + [1.]*13
                    for zoom in zooms:
                        crop=np.rint(full*(1-zoom)+close*zoom).astype(int)
                        view=im.crop(tuple(crop)).resize(PANEL,Image.Resampling.LANCZOS)
                        frame=page.copy()
                        frame.paste(view,(470,156))
                        title='前视' if name=='front_camera' else '后视'
                        review.label(frame,(1245,749),f'{title}    ·    第 {number} 帧',19)
                        writer.append_data(np.asarray(frame))
                        frames += 1
            cap.release()
    cap = cv2.VideoCapture(str(stage))
    decoded_frames = 0
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        if tuple(frame.shape[1::-1]) != SIZE:
            raise ValueError('Unexpected collection movie size')
        decoded_frames += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if decoded_frames != frames or fps != 10 or decoded_frames != 44 * len(KEYS):
        raise ValueError('Incomplete collection movie')
    stage.replace(COLLECTION/'common-20.mp4')
    return dict(frames=decoded_frames, seconds=decoded_frames/fps, fps=fps,
                front=front, back=back)


def package(data):
    credits=readme_sources()
    for kind in ('models-glb','inputs'):
        stage=COLLECTION/(kind+'.stage.zip')
        with ZipFile(stage,'w',compression=ZIP_STORED) as z:
            z.writestr('SOURCES.md',credits)
            z.write(RECIPE/'cases.json','generated-cases.json')
            for key in KEYS:
                bundle, _, kit=locations(key)
                if kind=='models-glb':
                    folder=(ROOT/data[key]['model']).parent
                    for f in ['asset.glb','metadata.json']:
                        z.write(folder/f,key+'/'+f,compress_type=ZIP_DEFLATED,compresslevel=2)
                else:
                    checks=read(kit/'checksums.json')
                    for name,digest in checks.items():
                        if sha256(kit/name) != digest:
                            raise ValueError(f'Input mismatch: {key}/{name}')
                        z.write(kit/name,key+'/'+name)
                    z.write(kit/'checksums.json',key+'/checksums.json')
        with ZipFile(stage) as z:
            if z.testzip():
                raise ValueError('Invalid ZIP '+kind)
        stage.replace(COLLECTION/(kind+'.zip'))


def readme_sources():
    old = (RECIPE.parent/'lane-batch/SOURCES.md').read_text()
    return ('P205–P219 各使用一张独立的 ImageGen 生成参考图，尺度按设计设定。P190 使用用户提供的道路照片。'
            '模型采用 MoGe 深度估计、坑沿及断壁标注和扫描材质。车载回放背景来自 PandaSet 109。\n\n'+old)


def write_index(data):
    lines=['# 车行道坑洞合集', '', '[合集视频](common-20.mp4) · [模型总览](overview.jpg) · [全部 GLB](models-glb.zip) · [重建输入包](inputs.zip)',
           '', '| 编号 | 形态 | 坑口面积 | 凹深 P95 | 参考图 | 模型 | 车载回放 |',
           '|---|---|---:|---:|---|---|---|']
    for key in KEYS:
        row=data[key]
        source,model,drive=[os.path.relpath(ROOT/row[k],COLLECTION) for k in ('source','model','drive')]
        kind='实拍' if row['reference_kind']=='photograph' else '生成'
        lines.append(f"| {key} | {row['label']} | {row['outline_area_m2']:.2f} m² | {row['depth_p95_m']*100:.1f} cm "
                     f"| [{kind}]({source}) | [GLB]({model}) | [播放]({drive}) |")
    (COLLECTION/'list.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-one',choices=KEYS)
    args=ap.parse_args()
    if args.audit_one:
        key=args.audit_one
        _,data=verify(key)
        (COLLECTION/key/'audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        return
    data,records={},{}
    for key in KEYS:
        _,folder,_=locations(key)
        saved=COLLECTION/key/'audit.json'
        if saved.exists():
            records[key]=read(folder/'placement.json')
            data[key]=read(saved)
        else:
            records[key],data[key]=verify(key)
            saved.write_text(json.dumps(data[key],ensure_ascii=False,indent=2)+'\n')
        if data[key]['native_hashes']['asset.glb']!=records[key]['glb_sha256']:
            raise ValueError(f'Stale model audit: {key}')
    for field in ('centre_world','basis_columns','site'):
        if any(r[field]!=records[KEYS[0]][field] for r in records.values()):
            raise ValueError('Different placement: '+field)
    for field in ('geometry_sha256','source_sha256'):
        if len({v[field] for v in data.values()}) != len(KEYS):
            raise ValueError('Duplicate '+field)
    contact_sheet(data)
    clip=make_movie(records,data)
    package(data)
    result=dict(site='109',count=len(KEYS),added=NEW,movie=clip,assets=data)
    (COLLECTION/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_index(data)


if __name__ == '__main__':
    main()
