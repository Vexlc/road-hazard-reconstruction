"""Render a camera-only inspection comparison without modifying native scenes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

import migrate as m
from camera_fit import structural_support,fit,projection
from run import atomic_json,sha


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    job=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf-8'))
    root=Path(os.environ['FRAMING_REVIEW_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
    case_id=os.environ.get('FRAMING_REVIEW_CASE','109_10');site=case_id.split('_')[0]
    models={x['id']:x for library in map(Path,job['libraries'])
            for x in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8'))}
    cache={};checks=[]
    for path in sorted(Path(job['output']).glob('*/scene.json')):
        case=json.loads(path.read_text(encoding='utf-8'))
        for placement in case['placements']:
            if placement['category'] not in ('missing','raised'):continue
            name=placement['asset_id']
            if name not in cache:cache[name]=structural_support(Path(job['mesh_cache'])/(name+'.npz'))
            points,corners=cache[name];center=np.array(placement['center_world']);basis=np.array(placement['basis_columns'])
            world=points@basis.T+center;minimum=1e9
            for angle in np.linspace(-.4,.4,24):
                eye,target=fit(center,basis,corners,float(angle));depth,margin=projection(world,eye,target)
                assert depth.min()>.04 and margin.min()>27.9,(case['id'],name,margin.min())
                minimum=min(minimum,float(margin.min()))
            checks.append({'case':case['id'],'asset_id':name,'angles':24,'minimum_structural_vertex_margin_px':minimum})
    scene=m.rb.create_scene();renderer=m.rb.StreetGaussianRenderer()
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    folder=Path(job['output'])/case_id;checkpoint=folder/'trained_model/iteration_7000.pth'
    sealed=sha(checkpoint)
    scene.gaussians.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False))
    case=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
    d=next(p for p in case['placements'] if p['category']=='raised');model=models[d['asset_id']]
    anchor=SimpleNamespace(center=np.array(d['center_world']),basis=np.array(d['basis_columns']),
                           reference=cameras[d['source_reference_frame']])
    points,corners=cache[d['asset_id']]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',29)
    output=root/'竖起井盖_局部取景修复对照.mp4'
    old_clipped=[]
    with imageio.get_writer(str(output),fps=12,codec='libx264',quality=8,macro_block_size=2,
                           output_params=['-threads','3','-movflags','+faststart']) as writer:
        for i,angle in enumerate(np.linspace(-.4,.4,24)):
            old=m.camera(anchor,model,float(angle))
            eye,target=fit(anchor.center,anchor.basis,corners,float(angle))
            new=m.rb.camera_at(anchor.reference,eye,target,(960,540),960*.91)
            first=m.rb.numpy_image(renderer.render(old,scene.gaussians))
            second=m.rb.numpy_image(renderer.render(new,scene.gaussians))
            w=old.get_extrinsic(world2cam=True).cpu().numpy();k=old.K.cpu().numpy()
            q=(points@anchor.basis.T+anchor.center)@w[:3,:3].T+w[:3,3]
            uv=q[:,:2]/q[:,2:3]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
            old_clipped.append(int(np.count_nonzero((q[:,2]<=0)|(uv[:,0]<0)|(uv[:,0]>=960)|(uv[:,1]<0)|(uv[:,1]>=540))))
            canvas=Image.new('RGB',(1920,640),(18,23,29));draw=ImageDraw.Draw(canvas)
            draw.text((20,9),'原近景：固定观察距离',font=font,fill='white')
            draw.text((980,9),'新版取景：完整井框与井盖范围',font=font,fill='#9ae2cc')
            canvas.paste(Image.fromarray(first),(0,56));canvas.paste(Image.fromarray(second),(960,56))
            draw.text((20,602),'同一原生场景；仅改变观察相机；几何、尺度、颜色和光照保持一致。',font=font,fill='#bac8d6')
            writer.append_data(np.asarray(canvas))
            if i==12:canvas.save(root/'竖起井盖_局部取景修复对照.jpg',quality=94)
    assert max(old_clipped)>0
    assert sha(checkpoint)==sealed
    atomic_json(root/'局部取景验证.json',{'passed':True,'checked_native_well_placements':len(checks),
       'checked_camera_angles':sum(c['angles'] for c in checks),'framing_checks':checks,
       'example_case':case_id,'example_original_clipped_vertices_per_frame':old_clipped,
       'example_new_clipped_structural_vertices':0,'native_checkpoint_byte_unchanged':True,
       'checkpoint_sha256':sealed,'method':'Fit the full frame/lid bounding box with 15mm geometric padding and 28px image margin. No asset scaling.',
       'scope':'Camera-fit preview. Existing v3 native checkpoints and five review videos are unchanged; this does not improve background sharpness or lighting.'})
    print('FRAMING REVIEW COMPLETE',len(checks),'placements',flush=True)


if __name__=='__main__':main()
