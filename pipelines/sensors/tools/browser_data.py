"""Pack local, millimetre-quantized drawing data for the offline viewer. Metric products keep full precision."""
import argparse
import base64
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from replay import load_scan,export
from road_geometry import load_inventory,read_json,save_json


def enc(arr):
    return base64.b64encode(np.ascontiguousarray(arr,dtype='<i2').tobytes()).decode('ascii')


def script(path,table,key,payload):
    path.write_text('window.'+table+'=window.'+table+'||{};window.'+table+'['+json.dumps(key)+']='+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf8')


@lru_cache(maxsize=36)
def base_points(out,site,frame):
    with np.load(Path(out)/'base'/site/f'{frame:03d}.npz') as f:return f['xyz'],f['sensor']


def run(parent,out):
    assets,scenes,_=load_inventory(parent)
    view=out/'viewer';cloud=view/'cloud';grid=view/'height'
    for d in [view,cloud,grid]:d.mkdir(exist_ok=True)
    result={};serial=0
    for aid,(asset_dir,meta) in sorted(assets.items()):
        package=asset_dir.parents[2].name
        result[aid]=dict(category=meta['category'],label=meta['label'],
            photo='../'+package+'/'+meta['source_image'],gaussians='../'+str((asset_dir/'asset.ply').relative_to(parent)),
            glb='../'+str((asset_dir/'asset.glb').relative_to(parent)),sites={})
        with np.load(out/'geometry'/(aid+'.npz')) as h:
            xs,ys=h['x_m'][::2],h['y_m'][::2]
            top=h['top_z_m'][::2,::2];bottom=h['substrate_z_m'][::2,::2];part=h['top_part'][::2,::2]
        a=np.stack([np.where(np.isfinite(top),np.rint(top*1000),-32768),
                    np.where(np.isfinite(bottom),np.rint(bottom*1000),-32768),part],axis=-1)
        if (part>=9).any():row=int(np.unravel_index(np.nanargmax(top),top.shape)[0])
        else:row=int(np.abs(ys).argmin())
        script(grid/(aid+'.js'),'RDheight',aid,dict(data=enc(a),w=len(xs),h=len(ys),x0=float(xs[0]),
            y0=float(ys[0]),step=.02,row=row))
    for case,(scene_file,info) in sorted(scenes.items()):
        site=info['site'];case_dir=out/'camera'/case;views=read_json(case_dir/'views.json');payload={}
        caseinfo=read_json(out/'cases'/(case+'.json'));inst={a['asset_id']:a for a in caseinfo['assets']}
        with np.load(out/'cases'/(case+'.npz')) as f:z={k:f[k] for k in ('row','frame_offsets','action','instance','xyz','part')}
        for row in views:
            if not row['available']:continue
            aid=row['asset_id'];local=inst[aid];frame=row['frame'];camera=row['camera'].split('_')[0]
            B=np.array(local['basis_columns']);c=np.array(local['center_world'])
            full,ds=base_points(str(out),site,frame)
            lo,hi=z['frame_offsets'][frame:frame+2]
            affected=(z['action'][lo:hi]==1)&(z['instance'][lo:hi]==local['instance'])
            new=z['xyz'][lo:hi][affected];ids=z['row'][lo:hi][affected];parts=z['part'][lo:hi][affected]
            # Use a cheap world AABB before converting the complete source scan to local coordinates.
            subset=np.flatnonzero(np.max(abs(full-c),axis=1)<2.8)
            xyz=(full[subset]-c)@B
            good=(np.max(abs(xyz[:,:2]),axis=1)<1.63)&(abs(xyz[:,2])<1.18)&(~np.isin(subset,z['row'][lo:hi]))
            context=subset[good]
            rng=np.random.default_rng(4421+serial);serial+=1
            if len(context)>500:context=np.sort(rng.choice(context,500,replace=False))
            orig=(full[ids]-c)@B;end=(new-c)@B
            within=(np.max(abs(orig[:,:2]),axis=1)<1.75)&(abs(orig[:,2])<1.3)
            bg=(full[context]-c)@B
            keep=np.arange(len(ids))
            if len(keep)>1500:keep=np.sort(rng.choice(keep,1500,replace=False))
            def packed(before,after,pr,ds,good):
                return np.column_stack((np.rint(before*1000).clip(-30000,30000),
                    np.rint(after*1000).clip(-30000,30000),pr,ds,good)).astype('<i2')
            drawing=np.concatenate([packed(bg,bg,np.zeros(len(bg)),ds[context],np.ones(len(bg))),
                                    packed(orig[keep],end[keep],parts[keep],ds[ids[keep]],within[keep])])
            key=aid+'_'+camera;payload[key]=enc(drawing)
            fname='camera/'+case+'/'
            item=dict(frame=frame,before=fname+row['before_stem'],after=fname+row['after_stem'],metric=fname+row['geometry'],
                pixels=row['visible_target_pixels'],returns=len(ids),by_sensor=np.bincount(ds[ids],minlength=2).tolist(),
                draw_context=len(bg),draw_changed=len(keep),shadow=row.get('cast_shadow_pixels',0),crop=row['crop_xyxy'])
            entry=result[aid]['sites'].setdefault(site,dict(case=case,by_sensor=local['by_sensor'],z=local['z_range_m'],parts=local['part_counts']))
            entry[camera]=item
        script(cloud/(case+'.js'),'RDcloud',case,payload)
    (view/'index.js').write_text('window.ROAD='+json.dumps(result,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf8')
    return result


def examples(out):
    directory=out/'examples';directory.mkdir(exist_ok=True)
    rows=read_json(out/'camera/034_13/views.json')
    for aid in ('P13','M13','A13'):
        record=next(r for r in rows if r['asset_id']==aid and r['camera']=='front_camera')
        frame=record['frame'];key=f'034_13_{aid}_f{frame:03d}'
        for mode in ('before','after'):
            scan=load_scan(out,'034_13',frame,mode)
            export(directory/(key+'_'+mode+'.ply'),scan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.parent,a.out);examples(a.out)
