"""Transfer the calibrated checkpoint's visible asset contribution onto original sensor RGB."""
import json
import hashlib
import subprocess
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

ROOT=Path(os.environ['GEN2HOLE_BUNDLE'])
sys.path.insert(0,str(Path(os.environ['ROAD_PROJECT'])/'scripts/hazard_factory'))
import migrate as merger
rb=merger.rb


@torch.no_grad()
def probes(g,case,rows):
    original={k:getattr(g.background,v) for k,v in rb.ATTRS.items()}
    dark=dict(original)
    dc=original['feature_dc'].clone()
    rest=original['feature_rest'].clone()
    labels=[]
    for i,p in enumerate(case['placements']):
        a=rows[p['asset_id']]
        path=(ROOT/'assets'/a['ply']).with_name('splat_material.npz')
        with np.load(path) as file:
            ao=file['ambient_visibility'].astype('f4')/255.
            # The auxiliary photographic render retains a small contact-shadow contribution.
            w=rb.tensor(np.where(file['road'],.68*(1-ao),1.).astype('f4'))[:,None,None]
        start=p['merged_asset_offset']; part=slice(start,start+p['asset_points'])
        if len(w)!=p['asset_points']:raise ValueError('Material labels and checkpoint differ')
        dc[part]=(dc[part]+.5/rb.C0)*(1-w)-.5/rb.C0
        rest[part]*=(1-w)
        labels.append((['pothole','missing','raised'].index(p['category']),part,w))
    dark['feature_dc']=torch.nn.Parameter(dc,requires_grad=False)
    dark['feature_rest']=torch.nn.Parameter(rest,requires_grad=False)
    coded=dict(dark);codes=dc.clone()
    for i,part,w in labels:codes[part,:,i]+=.70/rb.C0*w[:,:,0]
    coded['feature_dc']=torch.nn.Parameter(codes,requires_grad=False)
    return original,dark,coded


def box(mask):
    ys,xs=np.where(mask)
    if not len(xs):return [],0
    return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],len(xs)


def current_geometry(job,site,rows):
    numbers=job.get('case_groups',[])
    paths=[ROOT/'scenes'/f'{site}_{i:02d}'/'scene.json' for i in numbers]
    if not paths:return True
    for file in paths:
        if not file.is_file():return False
        for placement in json.loads(file.read_text())['placements']:
            a=rows[placement['asset_id']]
            if placement.get('reference_rgb_sha256')!=a['source_sha256']:return False
            with (ROOT/'assets'/a['ply']).open('rb') as f:
                h=hashlib.sha256()
                for block in iter(lambda:f.read(4<<20),b''):h.update(block)
            if h.hexdigest()!=placement.get('asset_ply_sha256'):return False
    return True


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    job=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf8'))
    plan=json.loads(Path(job['plan']).read_text(encoding='utf8'))
    rows={a['id']:a for a in json.loads((ROOT/'assets/asset_manifest.json').read_text(encoding='utf8'))}
    site=Path(rb.cfg.source_path).name;start=plan['drive_start_by_site'][site]
    if not current_geometry(job,site,rows):
        adapter=Path(__file__).with_name('native_entry.py')
        subprocess.run([sys.executable,'-u',str(adapter),*sys.argv[1:]],check=True)
        if not current_geometry(job,site,rows):raise ValueError('Native source digest mismatch after rebuild')
    data=Path(rb.cfg.source_path); raw=data/'images'
    with Image.open(raw/'000_0.jpg') as f: W,H=f.size
    scene=rb.create_scene();g=scene.gaussians;renderer=rb.StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda cam:cam.meta['frame'])
    rb.WIDTH,rb.HEIGHT=W,H;rb.resize_cameras(cameras)
    out=ROOT/'sensor_views';out.mkdir(exist_ok=True)
    for fn in sorted((ROOT/'scenes').glob(site+'_*/scene.json')):
        case=json.loads(fn.read_text(encoding='utf8'))
        if job.get('case_groups') and int(case['id'].split('_')[1]) not in job['case_groups']:continue
        dst=out/case['id'];dst.mkdir(exist_ok=True)
        signature=json.loads((fn.parent/'complete.json').read_text())['signature']
        if all((dst/n).is_file() for n in ('frames.json','front_after.mp4','classes.npz')) and json.loads((dst/'frames.json').read_text()).get('native_signature')==signature:continue
        state=torch.load(fn.parent/'trained_model/iteration_7000.pth',map_location='cuda',weights_only=False)
        g.load_state_dict(state)
        color_correction=bool(g.use_color_correction)
        g.use_color_correction=False
        original,dark,coded=probes(g,case,rows)
        masks=np.zeros((40,H,W),'u1')
        records=[]
        best=[-1,-1,-1]
        with imageio.get_writer(str(dst/'front_after.stage.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2,
                                output_params=['-threads','3','-movflags','+faststart']) as video:
            for j,frame in enumerate(range(start,start+40)):
                cam=cameras[frame]
                def render(state):
                    rb.set_background(g.background,state)
                    return renderer.render(cam,g)['rgb'].detach().clone()
                native=render(original);black=render(dark);classes=(render(coded)-black).clamp_min(0.)/.70
                matte=classes.sum(0).clamp(0,1)
                dyn=np.asarray(Image.open(data/'dynamic_masks/all'/f'{frame:03d}_0.png').convert('L').resize((W,H),Image.Resampling.NEAREST))>0
                visible=(matte>1/255.)&~torch.as_tensor(dyn,device='cuda')
                matte=matte*visible
                fore=torch.minimum((native-black).clamp_min(0),matte[None])
                photo=np.asarray(Image.open(raw/f'{frame:03d}_0.jpg').convert('RGB')).copy()
                src=rb.tensor(photo/255.).permute(2,0,1)
                img=torch.round(255*((1-matte)*src+fore)).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                label=torch.where((matte>.48)&visible,classes.argmax(0)+1,0).byte().cpu().numpy()
                masks[j]=label
                counts=[];boxes=[]
                for i in range(3):
                    bounds,count=box(label==i+1)
                    boxes.append(bounds);counts.append(count)
                    if count:
                        x0,y0,x1,y1=bounds
                        full=(x0>=8 and x1<=W-8 and y0>=8 and y1<=H-8)
                        score=count*(1 if full else .02)
                        if score>best[i]:
                            best[i]=score
                            evidence=Path(job['work'])/'sensor_examples'/case['id']
                            evidence.mkdir(exist_ok=True,parents=True)
                            key=['pothole','missing','raised'][i]
                            Image.fromarray(img).save(evidence/(key+'_after.jpg'),quality=97,subsampling=0)
                            (evidence/(key+'.json')).write_text(json.dumps({'frame':frame,'bbox':bounds,'count':count}))
                video.append_data(img)
                records.append({'frame':frame,'class_pixels':counts,'class_bboxes_xyxy':boxes})
        rb.set_background(g.background,original)
        g.use_color_correction=color_correction
        np.savez_compressed(dst/'classes.stage.npz',class_id=masks,source_frames=np.arange(start,start+40))
        (dst/'front_after.stage.mp4').replace(dst/'front_after.mp4')
        (dst/'classes.stage.npz').replace(dst/'classes.npz')
        info={'native_signature':signature,'case':case['id'],'camera':'front','size':[W,H],'fps':10,'source_frame_range_inclusive':[start,start+39],
              'method':'Original calibrated sensor RGB with native SH colour/opacity probes. The native depth sorting is retained. Dynamic-object image masks take precedence.',
              'mask_values':{'0':'background','1':'pothole','2':'missing','3':'raised'},
              'assets':case['assets'],'colour_correction_disabled_for_mattes':color_correction,
              'frames':records}
        (dst/'frames.stage.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
        (dst/'frames.stage.json').replace(dst/'frames.json')
        print('SENSOR',case['id'],[max(r['class_pixels'][i] for r in records) for i in range(3)],flush=True)
        del state,dark,coded,original
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
