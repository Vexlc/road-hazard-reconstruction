"""Verify coverage, native geometry, actor preservation and complete video decoding."""
from collections import Counter
import gc
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import re
import subprocess

import imageio_ffmpeg
import numpy as np
from plyfile import PlyData
import torch

WORK=Path(__file__).parent
OUT=PROJECT/'outputs/道路病害_多场景迁入'
LIB=PROJECT/'outputs/road_hazard_expansion'

def equal_subset(a,b):
    if torch.is_tensor(b):return torch.equal(a,b)
    if isinstance(b,dict):return all(k in a and equal_subset(a[k],v) for k,v in b.items())
    if isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal_subset(x,y) for x,y in zip(a,b))
    if isinstance(b,np.ndarray):return np.array_equal(a,b)
    return a==b

def main():
    torch.set_num_threads(4)
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((LIB/'asset_manifest.json').read_text(encoding='utf-8'))}
    selected=[n for xs in plan['selected_assets'].values() for n in xs]
    assert len(set(selected))==50 and all(len(x)==10 for x in plan['selected_assets'].values())
    original={}
    for name in selected:
        a=PlyData.read(LIB/models[name]['ply'],mmap='r')['vertex'].data
        original[name]=np.stack([a[n] for n in ['x','y','z']],axis=1)
    placements=[];reports=[]
    for site in plan['scene_titles']:
        source=torch.load(OUT/'重建场景'/site/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
        seen=[];max_error=0
        for batch in [b for b in plan['scenes'] if b['site']==site]:
            folder=OUT/'迁入场景'/batch['id']
            record=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
            assert record.get('insertion_revision')==2
            assert record['assets']==batch['assets']
            saved=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
            assert set(saved)==set(source),(site,'checkpoint components changed')
            for key in saved:
                if key not in ('background','iter'):assert equal_subset(source[key],saved[key]),(batch['id'],key)
            bg=saved['background']
            for key in ['xyz','feature_dc','feature_rest','scaling','rotation','opacity']:
                assert bool(torch.isfinite(bg[key]).all()),(batch['id'],key)
            for slot,d in enumerate(record['placements']):
                assert d['center_world']==plan['slots_by_site'][site][slot]['center_world']
                name=d['asset_id'];start=d['merged_asset_offset'];n=d['asset_points']
                assert d['size_multiplier']==1.0
                expected=original[name]@np.array(d['basis_columns']).T+np.array(d['center_world'])
                actual=bg['xyz'][start:start+n].detach().numpy()
                error=float(np.max(np.abs(actual-expected)))
                assert error<2e-5,(batch['id'],name,error)
                assert d['rms_m']<.03 and d['neighborhood_coverage']>=.72
                assert d['changed_pixels']>100,(batch['id'],name,'asset invisible')
                seen.append(name);max_error=max(error,max_error)
                placements.append({'scene':site,'version':batch['id'],'asset_id':name,'category':d['category'],
                                   'closeup_file':str(folder/f'{d["category"]}_close.mp4'),
                                   'checkpoint':record['checkpoint'],'center_world':d['center_world'],
                                   'basis_columns':d['basis_columns'],'size_multiplier':1.0,
                                   'category_video_start_seconds':(int(batch['id'][-2:])-1)*3})
            del saved;gc.collect()
        assert Counter(seen)==Counter(selected)
        reports.append({'scene':site,'assets':len(seen),'scene_versions':10,'max_world_position_error_m':max_error,
                        'original_actors_sky_identical':True,'all_background_parameters_finite':True,
                        'metrics':json.loads((OUT/'重建场景'/site/'重建指标.json').read_text(encoding='utf-8'))})
        del source;gc.collect()
        print('NATIVE SCENES VERIFIED',site,flush=True)
    assert len(placements)==150
    videos=json.loads((OUT/'可视化视频索引.json').read_text(encoding='utf-8'))
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    video_reports=[]
    for v in videos:
        p=OUT/v['file']
        result=subprocess.run([ffmpeg,'-hide_banner','-i',str(p),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        count=int(re.findall(r'frame=\s*(\d+)',result.stderr)[-1])
        assert count==v['frames']
        resolution='x'.join(map(str,v['resolution']))
        assert resolution in result.stderr
        video_reports.append({**v,'decoded_frames':count,'bytes':p.stat().st_size,'complete_decode':True})
        print('VIDEO VERIFIED',p.name,count,flush=True)
    validation={'new_reconstructed_scenes':3,'previous_reconstructed_scenes':2,'total_base_scenes':5,
                'random_seed':plan['seed'],'unique_assets':50,'assets_per_category':10,'total_placements':150,
                'scene_versions':30,'source_asset_scale_preserved':True,'scenes':reports,'videos':video_reports}
    (OUT/'全量验证.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'迁入资产检索.json').write_text(json.dumps(placements,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ALL VALIDATION COMPLETE',flush=True)

if __name__=='__main__':main()
