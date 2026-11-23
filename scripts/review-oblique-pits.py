"""Render source-pose checks from exported meshes and an orbit film from their native Gaussians."""
import argparse
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont, ImageOps

from atlas_gaussians import load, render, projection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles

INK, DIM, PAPER = '#202b2a', '#51605d', '#f3f2ec'


def read(path):
    return json.loads(path.read_text())


@lru_cache()
def font(size, bold=False):
    weight = 'Bold' if bold else 'Regular'
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-'+weight+'.ttc',size,index=2)


def label(image, xy, value, size, bold=False, color=INK):
    ImageDraw.Draw(image).text(xy,value,font=font(size,bold),fill=color,anchor='lt')


def source_check(bundle, row, out, context=True, detailed=False):
    key = row['id']
    cap = read(bundle/'calibration'/(key+'_capture.json'))
    spec = read(bundle/'calibration'/(key+'-outline.json'))
    K = np.array(cap['road_planes']['direct']['K'])
    axes = np.array(cap['local_axes_in_camera'])
    origin = np.array(cap['local_origin_in_camera'])
    rgb = np.array(ImageOps.exif_transpose(Image.open(bundle/row['source_image'])).convert('RGB'))
    y, x = np.indices(rgb.shape[:2],dtype='f4')
    ray = np.stack([(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],np.ones_like(x)],-1)@axes
    rays = np.concatenate([np.broadcast_to(-origin@axes,ray.shape),ray],axis=-1).astype('f4')
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    parts=[]
    for part,xyz,faces,uv,blob in glb_triangles(bundle/'assets'/row['glb'],names=True,appearance=True):
        scene.add_triangles(o3d.core.Tensor(xyz),o3d.core.Tensor(faces))
        parts.append((part,faces,uv,np.array(Image.open(io.BytesIO(blob)).convert('RGB'))))
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)
    primitive, geom, weights = [hit[k].numpy() for k in ('primitive_ids','geometry_ids','primitive_uvs')]
    recon = np.full_like(rgb, [220,222,216])
    footprint = np.zeros(x.shape,bool)
    for i,(part,faces,coords,tex) in enumerate(parts):
        use = geom==i
        if part=='Fracture lip and exposed bed':
            footprint=use
        w=weights[use]
        bary=np.column_stack([1-w.sum(1),w])
        uvs=np.einsum('ij,ijk->ik',bary,coords[faces[primitive[use]]])
        # Sample only hits. A full-size map also stays below OpenCV's 32767-pixel remap limit.
        lookup=np.zeros((*x.shape,2),'f4')
        lookup[use]=uvs*np.array([tex.shape[1]-1,tex.shape[0]-1])
        recon[use]=cv2.remap(tex,lookup,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[use]
    mask = np.zeros_like(x,'u1')
    rim = np.int32(spec['rim'])
    cv2.fillPoly(mask,[rim],1)
    outline=mask>0
    valid=footprint & outline
    for poly in spec.get('exclude_polygons',[]):
        cv2.fillPoly(mask,[np.int32(poly)],0)
    for a,b,c,d,t in spec.get('overlay_segments',[]):
        cv2.line(mask,(a,b),(c,d),0,t)
    valid &= mask>0
    if context and row.get('recipe_version') == 'oblique-photo-detail-v2':
        # Keep the traffic scene at its actual camera pose. Feather only the narrow
        # asphalt margin; the bed itself consists of GLB triangle hits and embedded UVs.
        blocked = np.zeros_like(mask)
        for poly in spec.get('exclude_polygons', []):
            cv2.fillPoly(blocked,[np.int32(poly)],1)
        apron = cv2.distanceTransform((~footprint).astype('u1'),cv2.DIST_L2,5)
        blend = np.clip(1-apron/5,0,1)*(geom<len(parts))*(blocked==0)
        marks = np.zeros_like(mask)
        detail=read(bundle/'calibration'/(key+'-detail.json'))
        for x0,y0,x1,y1,t in detail['context_detector_boxes']:
            cv2.rectangle(marks,(x0,y0),(x1,y1),1,t)
        clean = cv2.inpaint(rgb,marks,3,cv2.INPAINT_TELEA)
        scene = np.uint8(np.clip(clean*(1-blend[...,None])+recon*blend[...,None],0,255))
        x0,y0,x1,y1=spec['crop']
        (out/'P190-v2').mkdir(exist_ok=True)
        Image.fromarray(scene[y0:y1,x0:x1]).save(out/'P190-v2'/'street-view.jpg',quality=97,subsampling=0)
    iou=float(np.count_nonzero(outline & footprint)/np.count_nonzero(outline | footprint))
    error=float(np.abs(rgb[valid].astype('f4')-recon[valid]).mean())
    delta=rim.max(0)-rim.min(0)
    lo=np.maximum(0,np.int32(rim.min(0)-delta*[.075,.28]))
    hi=np.minimum(rgb.shape[1::-1],np.int32(rim.max(0)+delta*[.075,.28]))
    crop=tuple(np.r_[lo,hi])
    raw=Image.fromarray(rgb).crop(crop)
    view=Image.fromarray(recon).crop(crop)
    raw.save(out/(key+'-source.jpg'),quality=97)
    view.save(out/(key+'-source-pose.jpg'),quality=97)
    report=dict(outline_iou=iou,source_view_rgb_mae_255=error,visible_pix=int(valid.sum()),
                check='Ray intersection with delivered GLB from its fitting camera; same-view consistency')
    if detailed:
        delta=recon.astype('f4')-rgb
        weight=valid.astype('f4')
        filtered=cv2.GaussianBlur(delta*weight[...,None],(0,0),2.)
        support=cv2.GaussianBlur(weight,(0,0),2.)
        report['lowpass_rgb_mae_255']=float(abs(filtered[valid]/support[valid,None]).mean())
        report['pixel_mean_abs_error_p95_255']=float(np.percentile(abs(delta[valid]).mean(1),95))
    return report


def orbit_camera(points,theta,phi,w,h):
    xyz=points[:,[0,2,1]]*[1,1,-1]
    centre=(xyz.max(0)+xyz.min(0))*.5
    camera=dict(target=centre.tolist(),theta=theta,phi=phi,fov=30.,radius=1.)
    view,_=projection(camera,w,h)
    p=(points-centre[[0,2,1]]*[1,-1,1])@view[:3,:3].T
    slope=np.tan(np.deg2rad(15))*.83
    radius=np.maximum(abs(p[:,0])/(slope*w/h),abs(p[:,1])/slope)-p[:,2]
    camera['radius']=max(.1,float(radius.max()))
    return camera


def counts(bundle,row):
    with np.load(bundle/'assets'/row['surface']) as f:
        z=f['height']; pit=f['defect_mask']>.5
        pitch=float(f['width'])/(z.shape[1]-1)
        r,c=np.nonzero(pit)
        xy=np.float32(np.column_stack([c-(z.shape[1]-1)/2.,(z.shape[0]-1)/2.-r])*pitch)
        hull=cv2.convexHull(xy).reshape(-1,2)
        points=np.vstack([np.column_stack([hull,np.full(len(hull),level)]) for level in [0,z[pit].min()]])
        rect=sorted(map(float,cv2.minAreaRect(xy)[1]),reverse=True)
        return points,float(pit.sum()*pitch**2),rect


def fit(image,size):
    return ImageOps.contain(image,size,method=Image.Resampling.LANCZOS)


def place(canvas,image,box):
    x,y,w,h=box
    im=fit(image,(w,h))
    canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    items=[]
    for run in args.runs:
        bundle=run/'series'
        for row in read(bundle/'assets/asset_manifest.json'):
            items.append((row['id'],bundle,row))
    items.sort(key=lambda item: (0 if item[0] == 'P190' else 1 if item[0] == 'P197' else 2, item[0]))
    report={}
    snapshots={}
    fps=24
    tmp=out/'orbits.partial.mp4'
    with imageio.get_writer(str(tmp),fps=fps,codec='libx264',quality=8,macro_block_size=2,
                            output_params=['-threads','4','-movflags','+faststart']) as movie:
        for key,bundle,row in items:
            single=row['state']=='estimated'
            info=source_check(bundle,row,out) if single else read(bundle.parent/'evaluation/measurements.json')[key]['withheld_mesh']
            points,area,rect=counts(bundle,row)
            area=row.get('outline_area_m2',area)
            rect=row.get('oriented_extent_m',rect)
            report[key]=dict(label=row['label'],source=row.get('source','PothRGBD'),scale=row['scale_kind'],
                    footprint_m2=area,oriented_extent_m=rect,depth_p95_m=row['depth_p95_m'],check=info)
            source=Image.open(out/(key+'-source.jpg') if single else bundle/row['source_image']).convert('RGB')
            state=load(dict(row,ply=str((bundle/'assets'/row['ply']).resolve())))
            page=Image.new('RGB',(1440,810),PAPER)
            label(page,(36,29),key+'   '+row['label'],36,True)
            label(page,(38,96),'实拍参考',21,color=DIM)
            label(page,(515,96),'三维模型 · 环绕视角',21,color=DIM)
            place(page,source,(35,149,445,450))
            label(page,(38,652),'单张照片估计' if single else 'RGB-D 实拍重建',24,True)
            area_label = '可见范围约' if key == 'P197' else '坑口面积约'
            label(page,(38,699),f'{area_label} {area:.1f} m²',29,True)
            label(page,(515,675),f'坑口长 × 宽   {rect[0]:.1f} × {rect[1]:.1f} m',24,True)
            label(page,(515,725),f'可见表面凹深（P95）   {100*row["depth_p95_m"]:.0f} cm',23)
            if key=='P191':
                label(page,(38,760),'积水遮挡了水下坑底',17,color=DIM)
            elif key == 'P197':
                label(page,(38,760),'照片左缘截断了部分坑口',17,color=DIM)
            elif single:
                label(page,(38,760),'米制尺寸未经现场标定',17,color=DIM)
            else:
                label(page,(38,760),f'留出深度点误差中位数 {info["median_mm"]:.1f} mm',17,color=DIM)
            if key=='P190':
                scene=Image.new('RGB',(1440,810),PAPER)
                label(scene,(38,25),'护栏旁的路面塌坑',39,True)
                label(scene,(38,85),'用户提供的原始照片',23,color=DIM)
                place(scene,Image.open(bundle/row['source_image']),(26,141,1388,615))
                for _ in range(2*fps):
                    movie.append_data(np.asarray(scene))
            frames=144 if key=='P190' else 72
            _, axes = np.linalg.eigh(np.cov(points[:, :2], rowvar=False))
            long_axis = axes[:, -1]
            start = np.arctan2(long_axis[1], long_axis[0])
            for frame in range(frames):
                angle = start + 2*np.pi*frame/frames
                cam=orbit_camera(points,angle,np.deg2rad(48+12*np.sin(2*np.pi*frame/frames)),908,499)
                img=Image.fromarray(render(state,cam,908,499)[0])
                page.paste(img,(506,143))
                if frame==0:
                    snapshots[key]=(source.copy(),img.copy())
                    img.save(out/(key+'-orbit.jpg'),quality=97)
                movie.append_data(np.asarray(page))
            print(key,'area',round(area,2),'check',info,flush=True)
    tmp.replace(out/'orbits.mp4')
    (out/'measurements.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    hero=Image.new('RGB',(1500,950),PAPER)
    label(hero,(35,26),'护栏旁的路面塌坑',44,True)
    label(hero,(37,91),'P190   /   单张照片重建',22,color=DIM)
    label(hero,(40,166),'原始照片局部',23,True)
    label(hero,(40,473),'同一机位的模型回投',23,True)
    place(hero,snapshots['P190'][0],(35,216,660,220))
    place(hero,Image.open(out/'P190-source-pose.jpg'),(35,518,660,226))
    label(hero,(750,166),'模型旋转视角',23,True)
    place(hero,snapshots['P190'][1],(737,217,736,517))
    r=report['P190']
    label(hero,(40,815),f"估计坑口面积 {r['footprint_m2']:.1f} m²     /     可见表面凹深 P95 为 {100*r['depth_p95_m']:.0f} cm",29,True)
    label(hero,(40,875),'坑沿依照原照描绘。缺少现场尺度，遮挡与背向部分由相邻表面补全。',21,color=DIM)
    hero.save(out/'reference-comparison.jpg',quality=96,subsampling=0)
    cols=3
    rest=[i for i,_,_ in items if i!='P190']
    board=Image.new('RGB',(1560,122+415*((len(rest)+2)//3)),PAPER)
    label(board,(28,25),'其他实拍坑洞',44,True)
    for j,key in enumerate(rest):
        xx=22+j%cols*511; yy=121+(j//cols)*415
        s,im=snapshots[key]
        place(board,im,(xx,yy,495,226))
        place(board,s,(xx,yy+242,133,105))
        rr=report[key]
        label(board,(xx+146,yy+246),key+'  '+rr['label'],20,True)
        area_label = "可见范围约" if key == "P197" else "坑口约"
        label(board,(xx+146,yy+288),f"{area_label} {rr['footprint_m2']:.1f} m²",19)
        label(board,(xx+146,yy+325),'单图估计 · 水面可见' if key=='P191' else '单图估计' if 'monocular' in rr['scale'] else 'RGB-D 实拍',16,color=DIM)
    board.save(out/'additional-pits.jpg',quality=95,subsampling=0)
    archive = out/'models-glb.partial.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=3) as z:
        z.write(out/'measurements.json','measurements.json')
        for key,bundle,row in items:
            z.write(bundle/'assets'/row['glb'],key+'/asset.glb')
            z.write(bundle/'assets'/row['metadata'],key+'/metadata.json')
        z.writestr('sources.txt',
            'P190: user-provided photograph.\n'
            'P191: Pothole Big.jpg, Uncl3dad, public domain. https://commons.wikimedia.org/wiki/File:Pothole_Big.jpg\n'
            'P192-P196: PothRGBD, Mustafa Yurdakul and Sakir Tasdemir. https://github.com/ymyurdakul/PData\n'
            'P197: USGS, Sinkholes in West-central Florida, Freeze Event of 2010, public domain. '
            'https://www.usgs.gov/media/images/sinkholes-west-central-florida-freeze-event-2010-2\n'
            'P190 enhanced grain: Poly Haven Gravelly Sand and Stony Dirt Path, ambientCG Asphalt031, CC0.\n'
            'Single-image model scales are estimates. P197 covers the visible part of the collapse.\n')
    archive.replace(out/'models-glb.zip')


if __name__=='__main__':
    main()
