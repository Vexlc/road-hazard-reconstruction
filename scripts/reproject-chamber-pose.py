"""Render the actual delivered GLB into a registered lid capture and compare image-space support and depth."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gen2hole/downstream/sensors'))
from road_geometry import Surface
sys.path.insert(0,str(ROOT/'gen2hole/recipes/series'))
from chamber_depth import registered_depth


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def make(run,key,picture):
    bundle=run/'series'
    refs=bundle/'references'
    spec=read(bundle/'calibration/survey.json')
    state=next((s for s in spec['states'] if s['id']==key),{})
    if not state.get('pose_from_capture'):raise ValueError('Select a state reconstructed from a pose capture')
    observation=spec['pose_capture']
    gauge=read(bundle/'calibration/observed_pose.json')
    image=Image.open(refs/observation['image']).convert('RGB')
    mask=np.asarray(Image.open(refs/observation['lid_mask']).convert('L'))>0
    optical=registered_depth(refs/observation['depth'],observation['metres_per_unit'])
    fx,fy,cx,cy=gauge['intrinsics_px']
    h,w=mask.shape
    if image.size!=(w,h) or optical.shape!=(h,w):raise ValueError('Registered input size changed')
    transform=np.asarray(gauge['frame_to_camera'])
    rotation,t=transform[:3,:3],transform[:3,3]
    glb=bundle/'assets/raised'/key/'asset.glb'
    mesh=Surface(glb,threads=4)
    hits=np.zeros(mask.shape,'u1')
    depth=np.full(mask.shape,np.nan,'f4')
    for start in range(0,h,96):
        stop=min(start+96,h)
        y,x=np.mgrid[start:stop,:w]
        camera_rays=np.column_stack(((x.ravel()-cx)/fx,(y.ravel()-cy)/fy,np.ones(x.size)))
        directions=camera_rays@rotation
        origins=np.broadcast_to(-t@rotation,directions.shape)
        distance,part,_=mesh.cast(origins,directions)
        hits[start:stop]=part.reshape(stop-start,w)
        depth[start:stop]=distance.reshape(stop-start,w)
    predicted=hits==9
    agreement=mask&predicted
    values=agreement&np.isfinite(optical)&(optical>0)
    if values.sum()<24:raise ValueError('Too few common visible tread depth samples')
    residual=np.abs(depth[values]-optical[values])
    report=dict(source_kind=observation['source_kind'],asset=key,glb_sha256=hashlib.sha256(glb.read_bytes()).hexdigest(),
        annotated_pixels=int(mask.sum()),glb_visible_tread_pixels=int(predicted.sum()),common_valid_depth_pixels=int(values.sum()),
        mask_iou=float(agreement.sum()/(mask|predicted).sum()),mask_recall=float(agreement.sum()/mask.sum()),
        mask_precision=float(agreement.sum()/predicted.sum()),absolute_optical_z_error_median_m=float(np.median(residual)),
        absolute_optical_z_error_p90_m=float(np.percentile(residual,90)),
        estimated_angle_deg=gauge['angle_degrees'],note='Reprojection of the exported textured relief into the pose-fitting view; same-view consistency.')
    if picture:
        paint=np.asarray(image).copy()
        def border(m):return m&~binary_erosion(m,iterations=2)
        paint[border(mask)]=[255,180,46]
        paint[border(predicted)]=[0,218,168]
        valid=np.isfinite(optical)&(optical>0)
        lo,hi=np.percentile(optical[valid],[2,98])
        u=np.clip((np.nan_to_num(optical)-lo)/(hi-lo),0,1)
        palette=np.array([[51,63,120],[40,134,149],[174,209,139],[255,225,128]])
        rgb=np.stack([np.interp(u,np.linspace(0,1,len(palette)),palette[:,i]) for i in range(3)],axis=-1).astype('u1')
        rgb[~valid]=[36,39,41]
        panel=Image.new('RGB',(1584,450),'#f4f3ed');draw=ImageDraw.Draw(panel)
        def word(x,y,s,size,bold=False):
            weight='Bold' if bold else 'Regular'
            font=ImageFont.truetype(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc',size,index=2)
            draw.text((x,y),s,font=font,fill='#233c30',anchor='lt')
        word(24,16,'井盖姿态的 RGB-D 回投核验',39,True)
        tag='合成标定样例' if observation['source_kind']=='synthetic_benchmark' else '实拍观测'
        word(24,75,f"{tag}　|　估计倾角 {report['estimated_angle_deg']:.2f}°　|　盖面轮廓 IoU {report['mask_iou']:.3f}　|　同像素深度差中位数 {np.median(residual)*1000:.1f} mm",22)
        tiles=[(image,'姿态观测 RGB'),(Image.fromarray(rgb),f'光轴深度  {lo:.2f}–{hi:.2f} m；深灰为缺测'),
               (Image.fromarray(paint),'原标注（橙）与网格投影（绿）'),
               (Image.open(bundle/'views'/(key+'_inspection.jpg')),'导出的 GLB')]
        for i,(im,text) in enumerate(tiles):
            x=24+i*390
            thumb=ImageOps.contain(im,(368,255),Image.Resampling.LANCZOS)
            panel.paste(thumb,(x+(368-thumb.width)//2,125+(255-thumb.height)//2))
            word(x,396,text,17)
        picture.parent.mkdir(parents=True,exist_ok=True)
        panel.save(picture,quality=94,subsampling=0)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--asset',required=True)
    p.add_argument('--figure',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args()
    result=json.dumps(make(a.run,a.asset,a.figure),ensure_ascii=False,indent=2)+'\n'
    if a.report:a.report.write_text(result,encoding='utf-8')
    print(result,end='')
