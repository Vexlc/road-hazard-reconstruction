"""Read serialized assets/checkpoints and decode every native and final review video."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio_ffmpeg
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import yaml

from quality import validate
from run import PROJECT, atomic_json, fingerprint, sha, valid


def subset(actual, source):
    if torch.is_tensor(actual):return torch.is_tensor(source) and torch.equal(actual,source)
    if isinstance(actual,dict):return isinstance(source,dict) and all(k in source and subset(v,source[k]) for k,v in actual.items())
    if isinstance(actual,(tuple,list)):return len(actual)==len(source) and all(subset(a,b) for a,b in zip(actual,source))
    return actual==source


def decode(path,width,height,fps=None):
    params=['-vf',f'scale={width}:{height}'+(f',fps={fps}' if fps else '')]
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24',output_params=params)
    meta=next(reader)
    frames=[np.frombuffer(frame,dtype='u1').reshape(height,width,3).copy() for frame in reader]
    return meta,frames


def guard_height_check(background,count,data,records):
    """Compare sampled guard surfels against LiDAR frames not used by the fitter."""
    xyz=background['xyz'][-count:-count//2].detach().numpy().copy()
    q=background['rotation'][-count:-count//2].detach().numpy()
    normals=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:,2]
    xyz+=normals*.002  # Undo the deliberate undercoat offset.
    ids=np.linspace(0,len(xyz)-1,min(10000,len(xyz)),dtype=int)
    samples=xyz[ids]
    low,high=xyz[:,:2].min(0)-.2,xyz[:,:2].max(0)+.2
    cloud=[]
    centers=np.array([r['center_world'] for r in records])
    normals=np.array([np.array(r['basis_columns'])[:,2] for r in records])
    for frame in range(1,80,3):
        a=PlyData.read(data/'lidar_forward/background'/f'{frame:03d}.ply',mmap='r')['vertex'].data
        points=np.stack([a[k] for k in ('x','y','z')],axis=1)
        points=points[np.all((points[:,:2]>low)&(points[:,:2]<high),axis=1)]
        nearest=np.argmin(np.linalg.norm(points[:,None,:2]-centers[None,:,:2],axis=2),axis=1)
        reference=centers[nearest,2]-np.sum((points[:,:2]-centers[nearest,:2])*normals[nearest,:2],axis=1)/normals[nearest,2]
        cloud.append(points[abs(points[:,2]-reference)<.30])
    ground=np.concatenate(cloud)
    distance,nearest=cKDTree(ground[:,:2]).query(samples[:,:2],k=3,workers=2)
    close=(distance[:,0]<.08)&(distance[:,2]<.16)
    if close.sum()<1000:raise ValueError('insufficient independent held-out LiDAR terrain comparisons')
    error=abs(samples[close,2]-np.median(ground[nearest[close],2],axis=1))
    q95=float(np.quantile(error,.95))
    if q95>.05:raise ValueError(f'road undercoat differs from held-out LiDAR: p95={q95:.4f}m')
    return {'held_out_lidar_frames':list(range(1,80,3)),'sampled_surfels':len(samples),
            'matched_ground_samples':int(close.sum()),'median_absolute_height_error_m':float(np.median(error)),
            'p95_absolute_height_error_m':q95,'max_absolute_height_error_m':float(error.max()),
            'unmatched_samples_including_blind_zone':int((~close).sum()),
            'scope':'Height check only where independent held-out LiDAR exists; blind-zone continuation remains inferred.'}


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'))
    scenes=Path(job['output']);root=scenes.parent;work=Path(job['work'])
    if args.presentation and not (root/'五类动态展示/播放索引.json').is_file():
        raise RuntimeError('Presentation is not published yet; finish present.py before verification.')
    library=Path(job['libraries'][-1])
    manifest=json.loads((library/'asset_manifest.json').read_text(encoding='utf-8'))
    current=fingerprint()['sha256'];asset_checks=[]
    for m in manifest:
        directory=library/m['category']/m['id']
        recipe=json.loads((directory/'recipe.json').read_text(encoding='utf-8'))
        assert valid(directory,recipe,current),m['id']
        qa=validate(PROJECT,directory,Path(job.get('mesh_cache',str(work/'meshes'))),recipe)
        assert qa==m['quality']
        asset_checks.append({'id':m['id'],'category':m['category'],'quality':qa})
    all_models={}
    for folder in map(Path,job['libraries']):
        for m in json.loads((folder/'asset_manifest.json').read_text(encoding='utf-8')):
            all_models[m['id']]=dict(m,ply=str(folder/m['ply']))
    report={'assets':asset_checks,'scenes':[],'cases':[],'videos':[]}
    pairs=set();placements=0;category_ids={c:set() for c in ['stereo','rgbd','pothole','missing','raised']}
    plan=json.loads(Path(job['plan']).read_text(encoding='utf-8'))
    groups=min(sum(m['category']==c for m in manifest) for c in ('pothole','missing','raised'))
    for site,config in job['configs'].items():
        source_config=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        source_path=Path(source_config['model_path'])/'trained_model/iteration_7000.pth'
        source=torch.load(source_path,map_location='cpu',weights_only=False)
        tree=cKDTree(source['background']['xyz'].detach().numpy())
        cases=sorted(scenes.glob(f'{site}_*/scene.json'))
        assert len(cases)==groups
        guard_hash=None;guard_qa=None;guard_checks=[];guard_cache={}
        for path in cases:
            case=json.loads(path.read_text(encoding='utf-8'));folder=path.parent
            done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            assert all(sha(folder/name)==h for name,h in done['files'].items())
            state=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
            assert set(state)==set(source)
            for key in state:
                if key not in ('background','iter'):assert subset(state[key],source[key]),key
            background=state['background']
            assert all(torch.isfinite(v).all() for v in background.values() if torch.is_tensor(v))
            guard_count=case.get('host_road_guard',{}).get('points',0)
            kept=len(background['xyz'])-sum(d['asset_points']+d['apron_points'] for d in case['placements'])-guard_count
            if 'retained_background_points' in case:assert kept==case['retained_background_points']
            if guard_count:
                guard=case['host_road_guard']
                assert guard['layers']==2 and guard_count%2==0
                assert guard['terrain_fit_rms_max_m']<=.025
                assert guard['maximum_support_distance_m']<3.
                data=Path(source_config['source_path'])
                assert all(sha(data/name)==digest for name,digest in guard['source_geometry_sha256'].items())
                h=hashlib.sha256()
                for key in ('xyz','feature_dc','feature_rest','scaling','rotation','opacity'):
                    h.update(background[key][-guard_count:].detach().numpy().tobytes())
                digest=h.hexdigest()
                if digest not in guard_cache:
                    guard_cache[digest]=guard_height_check(background,guard_count,data,plan['slots_by_site'][site])
                if guard_hash is None:guard_hash=digest;guard_qa=guard_cache[digest]
                guard_checks.append({'case':case['id'],'sha256':digest,'independent_height':guard_cache[digest]})
            distance,old_indices=tree.query(background['xyz'][:kept].detach().numpy(),workers=2)
            assert distance.max()<1e-7
            for key in ('xyz','feature_dc','feature_rest','scaling','rotation','opacity'):
                assert torch.equal(background[key][:kept],source['background'][key][old_indices]),key
            checks=[]
            assert {d['category'] for d in case['placements']}==set(category_ids)
            for d in case['placements']:
                if d['category'] not in ('stereo','rgbd'):assert (d['asset_id'],site) not in pairs
                placements+=1
                pairs.add((d['asset_id'],site));category_ids[d['category']].add(d['asset_id'])
                m=all_models[d['asset_id']]
                ply=PlyData.read(m['ply'],mmap='r')['vertex'].data
                xyz=np.stack([ply[k] for k in ['x','y','z']],axis=1)
                scale=np.stack([ply[f'scale_{i}'] for i in range(3)],axis=1)
                quat=np.stack([ply[f'rot_{i}'] for i in range(4)],axis=1)
                basis=np.array(d['basis_columns']);center=np.array(d['center_world'])
                start=d['merged_asset_offset'];stop=start+d['asset_points']
                actual=background['xyz'][start:stop].detach().numpy()
                world_error=float(np.max(abs(actual-(xyz@basis.T+center))))
                assert world_error<2e-5 and np.array_equal(scale,background['scaling'][start:stop].detach().numpy())
                expected_rot=basis[None]@Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
                actual_quat=background['rotation'][start:stop].detach().numpy()
                rotation_error=float(np.max(abs(expected_rot-Rotation.from_quat(actual_quat[:,[1,2,3,0]]).as_matrix())))
                assert rotation_error<2e-5
                assert d['size_multiplier']==1 and d['reload_max_pixel_error']<=1
                assert d['actual_footprint_color_coverage']>=.80
                if d['category']=='missing':assert d['cavity_depth_check']['below_road_7cm_fraction']>=.55
                metal_check=None
                if d.get('material_contact'):
                    local_normals=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()[:,:,2]
                    # A downward-facing metal underside cannot be a transferred road.
                    down=(local_normals[:,2]<-.5)&(xyz[:,2]>-.02)&(xyz[:,2]<.022)
                    if down.any():
                        source_rgb=np.stack([ply[f'f_dc_{i}'] for i in range(3)],axis=1)[down]*.28209479177387814+.5
                        source_rgb=np.clip(source_rgb,0,1)
                        linear=np.where(source_rgb<=.04045,source_rgb/12.92,((source_rgb+.055)/1.055)**2.4)
                        linear=np.clip(linear*d['exposure_gain'],0,1)
                        expected=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)
                        rendered=background['feature_dc'][start:stop,0].detach().numpy()[down]*.28209479177387814+.5
                        error=float(np.max(abs(rendered-expected)))
                        assert error<1e-5,'metal underside was recolored as road'
                        metal_check={'downward_surface_points':int(down.sum()),'maximum_rgb_error_from_exposure_corrected_canonical':error}
                checks.append({'asset_id':d['asset_id'],'world_error_m':world_error,
                               'rotation_matrix_error':rotation_error,'physical_scales_unchanged':True,
                               'footprint_color_coverage':d['actual_footprint_color_coverage'],
                               'cavity':d['cavity_depth_check'],'reload_max_pixel_error':d['reload_max_pixel_error']})
                checks[-1]['metal_material_preservation']=metal_check
                _,frames=decode(folder/f'{d["category"]}_local_before_after.mp4',320,90)
                assert len(frames)==24
            _,frames=decode(folder/'global_before_after.mp4',640,180)
            assert len(frames)==40
            motion=[]
            for half in (0,1):
                crops=[f[20:140,half*320+20:half*320+300] for f in frames]
                unique=len({hashlib.sha256(f.tobytes()).digest() for f in crops})
                change=float(np.mean(abs(crops[0].astype(float)-crops[20].astype(float))))
                assert unique==40 and change>1.
                motion.append({'unique_scene_frames':unique,'scene_pixel_change':change})
            report['cases'].append({'case':case['id'],'checks':checks,'actor_sky_equal':True,
                                    'retained_background_parameters_equal':True,'global_motion':motion})
            del state,background
        report['scenes'].append({'site':site,'cases':len(cases),'source_checkpoint_sha256':sha(source_path),
                                 'road_guard_sha256':guard_hash,'independent_road_height':guard_qa,
                                 'case_road_checks':guard_checks})
        del source,tree
    assert placements==groups*len(job['configs'])*5
    for c,ids in category_ids.items():
        expected=min(groups,len(plan['selected_assets'][c])) if c in ('stereo','rgbd') else groups
        assert len(ids)==expected
    assert {m['id'] for m in manifest}.issubset({name for name,_ in pairs})
    if args.presentation:
        videos=root/'五类动态展示'
        index=json.loads((videos/'播放索引.json').read_text(encoding='utf-8'))
        for v in index['videos']:
            # Decode actual scene pixels, excluding headings, frame counters, and marker centers.
            reader=imageio_ffmpeg.read_frames(str(videos/v['file']),pix_fmt='rgb24',output_params=[
                '-vf','crop=1352:378:560:160,scale=676:189,fps=10'])
            meta=next(reader);frames=[]
            for raw in reader:frames.append(np.frombuffer(raw,dtype='u1').reshape(189,676,3).copy())
            expected_seconds=len(v['segments'])*4
            assert len(frames)==expected_seconds*10 and abs(meta['duration']-expected_seconds)<.1
            segments=[]
            for first in range(0,len(frames),40):
                counts=[];changes=[]
                for x in (0,340):
                    patches=[f[40:150,x+20:x+315] for f in frames[first:first+40]]
                    counts.append(len({hashlib.sha256(p.tobytes()).digest() for p in patches}))
                    changes.append(float(np.mean(abs(patches[0].astype(float)-patches[20].astype(float)))))
                assert min(counts)>=38 and min(changes)>1.
                segments.append({'unique_scene_frames':counts,'actual_scene_pixel_change':changes})
            report['videos'].append(dict(v,motion_checked_segments=segments))
    report.update(technical_checks_passed=True,new_asset_count=len(manifest),native_scene_versions=len(report['cases']),
                  actual_placements=placements,unique_asset_scene_pairs=len(pairs),all_physical_scale_1=True,
                  category_unique_assets={k:len(v) for k,v in category_ids.items()},
                  new_asset_materials=sorted({m['material'] for m in manifest}),
                  caveat='File/geometry/motion checks do not constitute a photorealism rating.')
    atomic_json(root/'技术验证.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets','cases','videos')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--presentation',action='store_true');main(p.parse_args())
