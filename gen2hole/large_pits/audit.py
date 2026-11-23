"""Check exported GLBs in the source camera and inspect matched road frames."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
review = import_module('review-oblique-pits')
from road_geometry import sha256


def read(path):
    return json.loads(path.read_text())


def source_pair(bundles, key, out):
    records = {}
    for version,bundle in bundles.items():
        folder = out/version
        folder.mkdir(parents=True,exist_ok=True)
        row = next(m for m in read(bundle/'assets/asset_manifest.json') if m['id']==key)
        checks = read(bundle/'checks.json')[key]
        directory = (bundle/'assets'/row['glb']).parent
        for name,digest in checks['sha256'].items():
            if sha256(directory/name)!=digest:
                raise ValueError('Native asset differs from checks.json: '+str(directory/name))
        measured = review.source_check(bundle,row,folder,context=False,detailed=True)
        records[version] = dict(source_check=measured, native_checks=checks, source_sha256=row['source_sha256'])
    if records['v2']['source_sha256'] != records['v3']['source_sha256']:
        raise ValueError('Different reference images')
    if records['v2']['native_checks']['geometry_sha256'] != records['v3']['native_checks']['geometry_sha256']:
        raise ValueError('Expected identical geometry for this material comparison')
    board = Image.new('RGB',(1300,630),'#f5f3ed')
    review.label(board,(24,20),key+'  /  原始相机位姿',29,True)
    photos = [('原始照片',out/'v3'/(key+'-source.jpg')),
              ('上一版',out/'v2'/(key+'-source-pose.jpg')),
              ('本版',out/'v3'/(key+'-source-pose.jpg'))]
    for i,(name,path) in enumerate(photos):
        im = Image.open(path)
        top = 66+i*184
        review.label(board,(24,top+70),name,22,True)
        review.place(board,im,(142,top,642,180))
        # Identical central crop of the far broken edge, enlarged at right.
        w,h = im.size
        crop = im.crop((round(.24*w),round(.17*h),round(.64*w),round(.53*h)))
        review.place(board,crop,(805,top,465,180))
    board.save(out/'source-comparison.jpg',quality=96,subsampling=0)
    return records


def check_movie(path, placement):
    cap = cv2.VideoCapture(str(path))
    total = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    shape = [1440,810]
    while True:
        ok,im = cap.read()
        if not ok:
            break
        if list(im.shape[1::-1])!=shape:
            raise ValueError('Unexpected frame dimensions: '+str(path))
        total += 1
    cap.release()
    frames = sum(len(rows) for rows in placement['frames'].values())
    if total != frames+33 or fps != placement['playback_fps']:
        raise ValueError('Incomplete or incorrect movie: '+str(path))
    return dict(decoded_frames=total, fps=fps, seconds=total/fps, size=shape)


def camera_summary(record):
    return {name: dict(frame_count=len(rows),first=rows[0]['frame'],last=rows[-1]['frame'],
                       protected_edits=sum(f['protected_changed_pixels'] for f in rows),
                       outside_edits=sum(f['outside_changed_pixels'] for f in rows),
                       peak_frame_self_shadow_pixels=next(f['self_shadow_pixels'] for f in rows
                                                         if f['frame']==record['selected'][name]['frame']))
            for name,rows in record['frames'].items()}


def road_board(old, new, old_info, new_info, out):
    if (old_info['centre_world'],old_info['basis_columns'])!=(new_info['centre_world'],new_info['basis_columns']):
        raise ValueError('Different road placement')
    board = Image.new('RGB',(1464,852),'#f5f3ed')
    review.label(board,(20,18),new_info['asset']+'  /  PandaSet '+new_info['site'],31,True)
    for i,stem in enumerate(('front','back')):
        name = stem+'_camera'
        sel = new_info['selected'][name]
        for k in ('frame','original_sha256','bbox'):
            if sel[k]!=old_info['selected'][name][k]:
                raise ValueError('Different camera selection: '+name+' '+k)
        x0,y0,x1,y1 = sel['bbox']
        for j,(version,folder) in enumerate((('上一版',old),('本版',new))):
            im = Image.open(folder/(stem+'-after.webp'))
            x = 18+j*723
            top = 75+i*385
            label = ('前视' if i==0 else '后视')+'  '+str(sel['frame'])+'  /  '+version
            review.label(board,(x,top),label,23,True)
            box = (max(0,x0-28),max(0,y0-18),min(im.width,x1+28),min(im.height,y1+18))
            review.place(board,im.crop(box),(x,top+39,705,330))
    board.save(out/'road-comparison.jpg',quality=96,subsampling=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--asset',default='P190')
    p.add_argument('--roads',nargs='*',default=[],type=Path)
    p.add_argument('--baseline-road',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    result = dict(asset=a.asset, models=source_pair({'v2':a.baseline,'v3':a.bundle},a.asset,a.out), roads={})
    result['texture_visibility']=read(a.bundle/'calibration'/(a.asset+'_capture.json'))['texture_visibility']
    old = read(a.baseline_road/'placement.json') if a.baseline_road else None
    matched = False
    for road in a.roads:
        state = read(road/'placement.json')
        hashes = result['models']['v3']['native_checks']['sha256']
        if state['asset']!=a.asset or state['ply_sha256']!=hashes['asset.ply'] or state['glb_sha256']!=hashes['asset.glb']:
            raise ValueError('Road replay references a different asset')
        if state['reload_max_pixel_error'] or state['size_multiplier']!=1:
            raise ValueError('Checkpoint reload or scale check failed')
        result['roads'][state['site']] = dict(movie=check_movie(road/'drive.mp4',state),cameras=camera_summary(state),
            centre_world=state['centre_world'], checkpoint_reload_max_pixel_error=state['reload_max_pixel_error'],
            direct_sun=state['surface_lighting']['sun']['direct_light_used'])
        if old and old['site']==state['site']:
            if old['ply_sha256']!=result['models']['v2']['native_checks']['sha256']['asset.ply']:
                raise ValueError('Baseline drive references a different asset')
            road_board(a.baseline_road,road,old,state,a.out)
            matched=True
    if old and not matched:
        raise ValueError('No matching new replay for the baseline road')
    (a.out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(a.out/'audit.json')


if __name__=='__main__':
    main()
