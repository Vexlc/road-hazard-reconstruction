"""Audit a photographed pool against its visible shoreline and the refilled destination meshes."""
import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gen2hole/downstream/sensors'))
from road_geometry import Surface, glb_triangles, read_json, save_json, sha256


def verify_mesh(folder):
    mesh = folder/'asset.glb'
    meta = read_json(folder/'metadata.json')
    water = meta['water']
    signature = hashlib.sha256()
    for name,vertices,faces,uv,texture in glb_triangles(mesh,names=True,appearance=True):
        if name=='Water surface':
            triangles = vertices[faces].astype('f8')
        else:
            for value in (name.encode(),vertices,faces,uv,texture):
                signature.update(value)
    normal = np.asarray(water['up_local'])
    flatness = float(abs(triangles@normal-water['free_surface_m']).max())
    start = triangles.mean(1)
    start[:,2] += .3
    hit,_,_ = Surface(mesh,exclude=(4,)).cast(start,np.broadcast_to([0,0,-1],start.shape))
    depths = hit.astype('f8')-.3
    area = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])*.5
    liters = float(area[:,2]@depths*1000)
    if (flatness>1e-6 or not np.isfinite(depths).all() or depths.min() < -1e-6 or
            not np.isclose(liters,water['retained_liters'],rtol=1e-5,atol=1e-5) or
            abs(liters+water['spilled_liters']-water['input_liters'])>.001):
        raise ValueError('Water surface, solid bed or volume budget does not agree: '+str(mesh))
    return dict(glb_sha256=sha256(mesh),bed_geometry_uv_texture_sha256=signature.hexdigest(),
                world_level_max_error_m=flatness,integrated_liters=liters,area_m2=float(np.linalg.norm(area,axis=1).sum()),
                spilled_liters=water['spilled_liters'],input_liters=water['input_liters'],
                grade_percent=float(100*np.linalg.norm(normal[:2])/normal[2]))


def source_view(bundle, key):
    cal = bundle/'calibration'
    capture = read_json(cal/(key+'_capture.json'))
    outline = np.int32(read_json(cal/(key+'-outline.json'))['rim'])
    water = np.int32(read_json(cal/(key+'-water.json'))['waterline'])
    left,top = outline.min(0)-[55,46]; right,bottom = outline.max(0)+[55,46]
    frame = np.array(Image.open(bundle/'references'/(key+'.jpg')).convert('RGB').crop((left,top,right,bottom)))
    rows,cols = np.indices(frame.shape[:2])
    camera = np.array(capture['road_planes']['direct']['K'])
    axes = np.array(capture['local_axes_in_camera'])
    origin = -np.array(capture['local_origin_in_camera'])@axes
    rays = np.stack(((cols+left-camera[0,2])/camera[0,0],(rows+top-camera[1,2])/camera[1,1],np.ones_like(cols)),-1).reshape(-1,3)@axes
    mesh = Surface(bundle/'assets/pothole'/key/'asset.glb')
    _,parts,_ = mesh.cast(np.broadcast_to(origin,rays.shape),rays)
    wet = parts.reshape(rows.shape)==4
    truth = cv2.fillPoly(np.zeros(rows.shape,'u1'),[water-[left,top]],1).astype(bool)
    overlap = (wet&truth).sum()/(wet|truth).sum()
    if overlap<.90:
        raise ValueError(f'Source water silhouette mismatch: {overlap:.3f}')
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),.65),28,58,L2gradient=True)>0
    distances = cv2.distanceTransform((~edges).astype('u1'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    boundary = wet & ~cv2.erode(wet.astype('u1'),np.ones((3,3),'u1')).astype(bool)
    edge_stats = dict(mean_px=float(distances[boundary].mean()),p90_px=float(np.quantile(distances[boundary],.90)),
                      within_1px_fraction=float((distances[boundary]<=1).mean()))
    marked = frame.copy()
    cv2.polylines(marked,[outline-[left,top]],True,(255,184,91),1)
    contours,_ = cv2.findContours(wet.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked,contours,-1,(60,240,231),1)
    return Image.fromarray(frame),Image.fromarray(marked),float(overlap),edge_stats


def depth_sensitivity(bundle, key, water):
    sys.path.insert(0,str(ROOT/'gen2hole/recipes/series'))
    from pit_water import solve
    capture = read_json(bundle/'calibration'/(key+'_capture.json'))
    with np.load(bundle/'assets/pothole'/key/'surface.npz') as field:
        height,shape = field['height'],field['normalized_submerged_bed']
        width,defect = float(field['width']),field['defect_mask']
    n = len(height)
    axis = np.linspace(-width/2,width/2,n)
    xx,yy = np.meshgrid(axis,axis[::-1])
    ids = np.arange(n*n,dtype='i4').reshape(n,n)
    left,right,lower,diag = ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
    faces = np.stack((left,lower,right,lower,diag,right),axis=-1).reshape(-1,3)
    depths = capture['underwater_depth_hypotheses_m']
    cases = []
    for depth in depths:
        candidate = height+(depths[1]-depth)*shape
        points = np.stack((xx,yy,candidate),axis=-1).reshape(-1,3)
        seed = dict(water,input_liters=float(water['input_liters']*depth/depths[1]))
        try:
            *_,stats = solve(points,faces,dict(height=candidate,defect_mask=defect),seed)
        except ValueError as exc:
            if 'drains' not in str(exc):
                raise
            stats = dict(retained_liters=0.,surface_area_m2=0.,max_water_depth_m=0.)
        cases.append(dict(depth_prior_m=depth,**{k:stats[k] for k in ('retained_liters','surface_area_m2','max_water_depth_m')}))
    if not np.isclose(cases[1]['surface_area_m2'],water['surface_area_m2'],atol=.0001):
        raise ValueError('Uncertainty sweep did not reproduce nominal delivery')
    return cases


def comparison_reel(out, baseline, sites, report):
    fontfile = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    large,medium,small = [ImageFont.truetype(fontfile,size,index=2) for size in (31,25,18)]
    reel = out/'revision.stage.mp4'
    with imageio.get_writer(str(reel),fps=10,codec='libx264',quality=8,macro_block_size=2,
                            output_params=['-threads','3','-movflags','+faststart']) as writer:
        for site in sorted(sites,key=lambda name:(name!='109',name!='110',name)):
            old,new = [root/('scene-'+site) for root in (baseline,out)]
            records = [read_json(folder/'placement.json') for folder in (old,new)]
            picks = [record['selected']['front_camera'] for record in records]
            if picks[0]['original_sha256']!=picks[1]['original_sha256']:
                raise ValueError('Comparison requires exactly the same original frame')
            tracks = [{row['frame']:(index,row) for index,row in enumerate(record['frames']['front_camera'])}
                      for record in records]
            near = picks[1]['frame']
            data = report['sites'][site]
            before_area = read_json(baseline/'quality.json')['sites'][site]['area_m2']
            def plate(images, bbox, frame):
                board = Image.new('RGB',(1280,536),'#182124')
                draw = ImageDraw.Draw(board)
                draw.text((25,13),f"P204  /  路段 {site}    ·    坡度 {data['grade_percent']:.1f}%",font=large,fill='#f5f3e9')
                draw.text((988,24),f'同机位  /  前视帧 {frame:02}',font=small,fill='#aec1be')
                left,top,right,bottom = bbox
                width = min(1920,max(550,(right-left)*1.55))
                height = width*336/600
                start = np.clip([(left+right-width)/2,(top+bottom-height)/2],[0,0],[1920-width,1080-height])
                box = np.r_[start,start+[width,height]]
                for index,(image,title) in enumerate(zip(images,('旧版','优化后'))):
                    pos = 26+640*index
                    draw.text((pos,65),title,font=medium,fill='#ded9cb' if not index else '#98e2d2')
                    scaled = np.rint(box*np.array(image.size*2)/[1920,1080,1920,1080]).astype(int)
                    board.paste(image.crop(tuple(scaled)).resize((600,336),Image.Resampling.LANCZOS),(pos,109))
                caption = f"模型水面面积   {before_area:.2f} → {data['area_m2']:.2f} m²"
                if not data['depth_resolved_above_host_plane_rms']:
                    caption += '    ·    剩余水深小于路面拟合误差，不宜解读为可辨水池'
                draw.text((27,464),caption,font=medium,fill='#f1f3e9')
                draw.text((28,504),'两版均为模拟；裁切区域和播放速度相同。',font=small,fill='#afc2c0')
                return board
            readers = [imageio.get_reader(folder/'drive.mp4') for folder in (old,new)]
            try:
                for frame in sorted(set(tracks[0])&set(tracks[1])):
                    if not near-14 <= frame < near:
                        continue
                    pair = [Image.fromarray(reader.get_data(track[frame][0])) for reader,track in zip(readers,tracks)]
                    writer.append_data(np.asarray(plate(pair,tracks[1][frame][1]['bbox'],frame)))
            finally:
                for reader in readers:
                    reader.close()
            images = [Image.open(folder/'front-after.webp').convert('RGB') for folder in (old,new)]
            final = plate(images,picks[1]['bbox'],near)
            final.save(out/f'revision-{site}.jpg',quality=95,subsampling=0)
            for repeat in range(23):
                writer.append_data(np.asarray(final))
    reel.replace(out/'revision.mp4')


def review(args):
    source = args.run/'source/series'
    key = args.asset
    capture = verify_mesh(source/'assets/pothole'/key)
    reference, marked, iou, edge = source_view(source,key)
    args.out.mkdir(exist_ok=True,parents=True)
    marked.save(args.out/'source-outline.png')
    report = dict(asset=key,source=capture,source_water_silhouette_iou=iou,source_image_edge_distance=edge,sites={})
    calibration = read_json(source/'calibration'/(key+'_capture.json'))
    if args.baseline_run:
        prior = args.baseline_run/'source/series'
        if sha256(prior/'references'/(key+'.jpg')) != sha256(source/'references'/(key+'.jpg')):
            raise ValueError('Different source photo in the baseline')
        report['prior_source_image_edge_distance'] = source_view(prior,key)[3]
    if 'projected_area_sensitivity_m2' in calibration:
        report['source_area_sensitivity_m2'] = calibration['projected_area_sensitivity_m2']
    board = Image.new('RGB',(1200,805),'#f3f2ea'); draw = ImageDraw.Draw(board)
    def label(xy,text,size):
        draw.text(xy,text,fill='#193136',font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size,index=2))
    def panel(image,box):
        x,y,w,h = box
        ratio = min(w/image.width,h/image.height)
        image = image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.LANCZOS)
        board.paste(image,(x+(w-image.width)//2,y+(h-image.height)//2))
    label((24,10),key+'  /  同一个坑床，不同坡度的积水',36)
    for index,(picture,title) in enumerate(((reference,'原始照片 · 右侧箭头前方'),(marked,'原机位 · 橙线为坑沿标注，青线为网格水面回投'))):
        label((25+590*index,74),title,20); panel(picture,(24+590*index,110,560,228))
    width=1152//len(args.sites)
    for index,site in enumerate(args.sites):
        bundle=args.run/site/'series'; scene=bundle/'scenes'/f'{site}_{int(key[1:])}'; dest=args.out/('scene-'+site)
        folder=bundle/'assets/pothole'/key; check=verify_mesh(folder); info=read_json(scene/'scene.json')
        placement,=info['placements']; views=read_json(dest/'placement.json'); replay,=read_json(dest/'checkpoint-replay.json')
        if not (info['checkpoint_reloaded'] and info['actors_sky_unchanged'] and placement['size_multiplier']==1
                and replay['case']==scene.name and check['bed_geometry_uv_texture_sha256']==capture['bed_geometry_uv_texture_sha256']
                and views['glb_sha256']==check['glb_sha256'] and placement['asset_ply_sha256']==sha256(folder/'asset.ply')
                and np.allclose(np.array(placement['basis_columns'])@read_json(folder/'metadata.json')['water']['up_local'],[0,0,1],atol=1e-6)):
            raise ValueError('Placement, gravity, bed or source mismatch: '+site)
        frames=[row for shots in views['frames'].values() for row in shots]
        if any(row['outside_changed_pixels'] or row['protected_changed_pixels'] for row in frames):
            raise ValueError('Pixels outside material or in a protected region changed: '+site)
        reader=imageio.get_reader(dest/'drive.mp4')
        try:
            decoded=sum(1 for _ in reader)
        finally:
            reader.close()
        if decoded!=len(frames)+33 or not any(row['visible_water_pixels'] for row in frames):
            raise ValueError('Missing water or incomplete drive: '+site)
        shot=views['selected']['front_camera']; check.update(checked_camera_frames=len(frames),decoded_video_frames=decoded,
            selected_front_frame=shot['frame'],selected_water_pixels=shot['visible_water_pixels'],
            protected_and_outside_edits=0,checkpoint_replay=replay)
        report['sites'][site]=check; pos=24+index*width
        if 'underwater_depth_hypotheses_m' in calibration:
            water = read_json(folder/'metadata.json')['water']
            check['depth_sensitivity'] = depth_sensitivity(bundle,key,water)
            check['host_plane_rms_m'] = placement['rms_m']
            check['depth_resolved_above_host_plane_rms'] = water['max_water_depth_m'] > placement['rms_m']
            optics = {name:sum(row['reflection'][name] for row in frames) for name in frames[0]['reflection']}
            if optics['refracted_bed_samples'] < .995*optics['water_samples']:
                raise ValueError('Water rays did not reach the solid bed: '+site)
            check['optical_samples'] = optics
        label((pos,354),f"{site}  /  坡度 {check['grade_percent']:.1f}%",22)
        label((pos,390),f"模型留水 {check['integrated_liters']:.2f} L  ·  前视帧 {shot['frame']}",17)
        left,top,right,bottom=shot['bbox']; pad=max(24,(right-left)//8)
        for row,suffix in enumerate(('before','after')):
            image=Image.open(dest/('front-'+suffix+'.webp'))
            region=(max(0,left-pad),max(0,top-pad),min(image.width,right+pad),min(image.height,bottom+pad))
            panel(image.crop(region),(pos,433+row*165,width-14,153))
    label((24,760),'上排原路面，下排迁入后。留水量由假设坑床计算，不是现场测量。',18)
    board.save(args.out/'overview.jpg',quality=95,subsampling=0)
    if args.baseline_views:
        comparison_reel(args.out,args.baseline_views,args.sites,report)
    save_json(args.out/'quality.json',report)
    print(f'{key}: source water IoU {iou:.4f}; checked {len(args.sites)} refills')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--asset',default='P204')
    parser.add_argument('--baseline-run',type=Path)
    parser.add_argument('--baseline-views',type=Path)
    parser.add_argument('--sites',nargs='+',default=['110','109','034'])
    review(parser.parse_args())
