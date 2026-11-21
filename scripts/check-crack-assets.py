"""Compare exported crack vertices and scene checkpoints to their registered source masks."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image

mesh_parts=runpy.run_path(str(Path(__file__).with_name('check-chamber-family.py')))['mesh_parts']


def check(root):
    bundle=root/'series'
    manifest=json.loads((bundle/'assets/asset_manifest.json').read_text())
    scenes={}
    for path in (bundle/'scenes').glob('*/scene.json'):
        scene=json.loads(path.read_text())
        for p in scene['placements']:
            if not scene['checkpoint_reloaded'] or p['reload_max_pixel_error']!=0:
                raise ValueError('Checkpoint reload mismatch: '+scene['id'])
            scenes.setdefault(p['asset_id'],[]).append(dict(case=scene['id'],ply_sha256=p['asset_ply_sha256'],
                                                            world_error_m=p['world_error_m']))
    result={}
    for meta in manifest:
        key=meta['id']
        mask=np.asarray(Image.open(bundle/'calibration'/(key+'_outline.png')).convert('L'))>0
        with np.load(bundle/'assets'/meta['surface']) as field:
            if not np.array_equal(field['defect_mask']>.5,mask):
                raise ValueError(key+' mask raster mismatch')
            height=field['height']
        span=meta['metric_source_span_m']
        decoded=np.full(mask.shape,np.nan)
        parts=mesh_parts(bundle/'assets'/meta['glb'])
        for xyz,_ in parts.values():
            col=np.rint((xyz[:,0]/span+.5)*(mask.shape[1]-1)).astype(int)
            row=np.rint((.5-xyz[:,1]/span)*(mask.shape[0]-1)).astype(int)
            decoded[row,col]=xyz[:,2]
        if not np.isfinite(decoded).all():raise ValueError(key+' missing GLB vertices')
        error=float(np.abs(decoded-height).max())
        if error>1e-6 or np.max(decoded)>1e-6:raise ValueError(key+' decoded GLB vs signed metric field')
        # Outermost vertices return smoothly to road level where the photo crop ends.
        visible=mask.copy()
        visible[[0,-1],:]=False;visible[:,[0,-1]]=False
        depressed=decoded < -1e-9
        overlap=float((depressed&visible).sum()/(depressed|visible).sum())
        if overlap<.998:raise ValueError(key+' mask footprint overlap')
        asset_hash=hashlib.sha256((bundle/'assets'/meta['ply']).read_bytes()).hexdigest()
        for scene in scenes.get(key,[]):
            if scene['ply_sha256']!=asset_hash:raise ValueError(key+' source PLY mismatch')
        result[key]=dict(source_mask_pixels=int(mask.sum()),components=meta['components'],triangles=sum(meta['geometry_parts'].values()),
                         gaussians=meta['gaussians'],source_mask_vs_glb_interior_iou=overlap,mesh_vs_field_max_m=error,
                         deepest_vertex_m=float(-decoded.min()),design_limit_m=meta['depth_design_m'],scenes=scenes.get(key,[]))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--require-scenes',action='store_true')
    args=p.parse_args()
    audit=check(args.run)
    if args.require_scenes and not all(a['scenes'] for a in audit.values()):raise ValueError('Some assets lack a scene checkpoint')
    print(json.dumps(audit,indent=2,allow_nan=False))
