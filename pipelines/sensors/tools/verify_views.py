"""Audit paired image pixels, occlusion, legacy front silhouettes and cross-sensor reprojection."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from road_geometry import CLASSES,load_inventory,read_json,save_json


def main(parent,out,raw):
    _,scenes,_=load_inventory(parent)
    cameras=Counter(); protected=Counter(); shadow_views=Counter();shadow_pixels=Counter();colour=[];ious=[];ratios=[];pixelcounts=[];skipped=[]
    residuals=[];projection_counts=Counter();view_count=0;seen=set()
    profile=read_json(out/'tools/ego_back_profile.json')
    for case,(origin,scene) in sorted(scenes.items()):
        s=scene['site'];folder=out/'camera'/case;views=read_json(folder/'views.json')
        assert len(views)==2*len(scene['placements'])
        with np.load(out/'cases'/(case+'.npz')) as f: delta={k:f[k] for k in ('frame_offsets','action','instance','part','xyz')}
        legacypath=origin.parents[2]/'sensor_views'/case/'classes.npz'
        with np.load(legacypath) as f:legacy=f['class_id'];frames=f['source_frames'].tolist()
        for i in views:
            if not i['available']:skipped.append((case,i['asset_id'],i['camera']));continue
            aid,cam,frame=i['asset_id'],i['camera'],i['frame'];view_count+=1;cameras[cam]+=1
            ident=(case,aid,cam);assert ident not in seen;seen.add(ident)
            l,t,r,b=i['crop_xyxy'];rows=slice(t,b);cols=slice(l,r)
            before=np.asarray(Image.open(folder/i['before_stem']).convert('RGB'))
            after=np.asarray(Image.open(folder/i['after_stem']).convert('RGB'))
            with Image.open(raw/s/'camera'/cam/f'{frame:02d}.jpg') as source:
                assert np.array_equal(before,np.asarray(source.convert('RGB').crop((l,t,r,b))))
            with np.load(folder/i['geometry']) as g:
                dep,prt,instance,pro,alpha,norm,sun=(g[k] for k in ('depth_z_m','part','instance','protected','composite_alpha','world_normal','sun_shadow'))
            assert np.array_equal(np.isfinite(dep),prt>1) and np.array_equal(instance>0,prt>1)
            assert np.array_equal(after[pro>0],before[pro>0]) and not alpha[pro>0].any()
            assert not np.any((pro>0)&(prt>0))
            assert not np.any((sun>0)&((pro>0)|(prt>0)))
            assert (sun.max()==0) if s=='034' else True
            shade=int((sun>51).sum())
            assert abs(shade-i['cast_shadow_pixels'])<=int((sun==51).sum())
            if shade:shadow_views[s]+=1;shadow_pixels[s]+=shade
            if cam=='back_camera':
                pf=profile[s];ego=np.arange(t,b)[:,None]>=np.interp(np.arange(l,r),pf['x'],pf['first_protected_y'])[None,:]
                assert np.array_equal(ego,(pro&4)>0)
            assert abs(np.linalg.norm(norm[prt>1].astype('f4'),axis=1)-1).max()<.002
            for bit in (1,2,4):protected[str(bit)]+=int(((pro&bit)>0).sum())
            target=(instance==i['instance']);assert target.sum()==i['visible_target_pixels']
            assert (alpha[target]>0).any() and np.isfinite(dep[target]).all() and (dep[target]>0).all()
            pixelcounts.append(int(target.sum()));ratios.append(float((alpha[target]>122).mean()))
            colour.append(float(np.abs(before.astype('f4')-after).mean(axis=2)[target].mean()))
            if cam=='front_camera' and frame in frames:
                cl=CLASSES[scene['placements'][i['instance']-1]['category']]
                original=legacy[frames.index(frame),rows,cols]==cl
                inter=np.count_nonzero(original&target);union=np.count_nonzero(original|target)
                ious.append((inter/union,case,aid))
            lo,hi=delta['frame_offsets'][frame:frame+2]
            use=(delta['instance'][lo:hi]==i['instance'])&(delta['action'][lo:hi]==1)
            xyz=delta['xyz'][lo:hi][use];hitpart=delta['part'][lo:hi][use]
            P=np.asarray(i['camera_to_world']);K=np.asarray(i['K_full']);v=(xyz-P[:3,3])@P[:3,:3]
            xy=np.floor(v[:,:2]/np.where(v[:,2:3]!=0,v[:,2:3],1)*[K[0,0],K[1,1]]+[K[0,2]-l,K[1,2]-t]).astype('i4')
            valid=(v[:,2]>.05)&(xy[:,0]>=0)&(xy[:,0]<r-l)&(xy[:,1]>=0)&(xy[:,1]<b-t)
            xx,yy=xy[valid].T;vz=v[valid,2]
            same=(instance[yy,xx]==i['instance'])&(prt[yy,xx]==hitpart[valid])
            res=vz[same]-dep[yy[same],xx[same]]
            residuals.extend(res.tolist());projection_counts['target_lidar_returns_at_capture']+=len(v)
            projection_counts['inside_crop_and_in_front']+=len(vz);projection_counts['same_visible_mesh_part']+=len(res)
    vals=np.array([a[0] for a in ious]);res=np.array(residuals)
    def stats(v):return {k:round(float(np.percentile(v,p)),6) for k,p in [('min',0),('p10',10),('median',50),('p90',90),('max',100)]}
    report=dict(views=view_count,camera_counts=dict(cameras),skipped=skipped,protected_pixels=dict(protected),direct_shadow_views=dict(shadow_views),direct_shadow_pixels=dict(shadow_pixels),
        target_pixels=stats(pixelcounts),gaussian_alpha_above_48pct_coverage=stats(ratios),target_rgb_difference_8bit=stats(colour),
        shared_front_frame_count=len(ious),legacy_gaussian_silhouette_vs_mesh_iou=stats(vals),smallest_front_iou=sorted(ious)[:4],
        same_part_reprojection_counts=dict(projection_counts),signed_lidar_minus_camera_depth_m=stats(res),
        absolute_depth_error_under_5cm=int((abs(res)<.05).sum()),
        note='Different physical emitters can see different parts or different faces of the same part. These are pixel-depth comparisons, not calibration ground truth.')
    save_json(out/'checks/camera_audit.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent','out','raw']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.parent,a.out,a.raw)
