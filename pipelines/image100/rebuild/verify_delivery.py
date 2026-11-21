"""Check published GLB/PLY bytes, metric fields, native checkpoint receipts and visible driving masks."""
import argparse,collections,csv,hashlib,json,struct
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'checks'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=1,allow_nan=False)+'\n')
def glb(p):
    data=p.read_bytes();magic,version,length=struct.unpack_from('<4sII',data)
    assert (magic,version,length)==(b'glTF',2,len(data))
    n,typeid=struct.unpack_from('<II',data,12);assert typeid==0x4e4f534a
    scene=json.loads(data[20:20+n]);size,tag=struct.unpack_from('<II',data,20+n);begin=28+n
    assert tag==0x004e4942 and size+begin==len(data)
    def acc(i):
        a=scene['accessors'][i];v=scene['bufferViews'][a['bufferView']];dim={'VEC2':2,'VEC3':3,'SCALAR':1}[a['type']]
        typ={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        assert 'byteStride' not in v;off=v.get('byteOffset',0)+a.get('byteOffset',0)
        return np.frombuffer(data,typ,count=a['count']*dim,offset=begin+off).reshape(-1,dim)
    h=hashlib.sha256();faces=0;ymin=np.inf;ymax=-np.inf
    for mesh in scene['meshes']:
        for prim in mesh['primitives']:
            a=prim['attributes'];xyz=acc(a['POSITION']);normal=acc(a['NORMAL']);uv=acc(a['TEXCOORD_0']);idx=acc(prim['indices'])
            assert np.isfinite(xyz).all() and np.isfinite(uv).all() and np.isfinite(normal).all()
            assert len(xyz)==len(uv)==len(normal) and idx.max()<len(xyz) and len(idx)%3==0
            faces+=len(idx)//3;h.update(xyz.tobytes());h.update(idx.tobytes());ymin=min(ymin,xyz[:,1].min());ymax=max(ymax,xyz[:,1].max())
    for mat in scene['materials']:
        m=mat['pbrMetallicRoughness'];assert 'baseColorTexture' in m and 'metallicRoughnessTexture' in m and 'normalTexture' in mat
    for img in scene['images']:
        v=scene['bufferViews'][img['bufferView']];assert v['byteLength']>300 and 'uri' not in img
    return h.hexdigest(),faces,float(ymin),float(ymax)
def ply(p,meta):
    with p.open('rb') as f:
        line=f.readline();assert line.strip()==b'ply';names=[]
        while True:
            line=f.readline().decode().strip()
            if not line:raise ValueError(p)
            if line=='end_header':break
            if line.startswith('element vertex'):num=int(line.split()[-1])
            if line.startswith('property '):
                prop=line.split();assert prop[1]=='float';names.append(prop[-1])
        offset=f.tell()
    assert names[-1]=='rot_3' and sum(s.startswith('f_rest_') for s in names)==9 and num==meta['gaussians']
    assert p.stat().st_size-offset==4*num*len(names)
    values=np.memmap(p,mode='r',dtype='<f4',offset=offset,shape=(num,len(names)))
    for i in range(0,num,65536):
        a=values[i:i+65536];assert np.isfinite(a).all()
        assert a[:,2].min()>=meta['lowest_z_m']-2e-4 and a[:,2].max()<=meta['highest_z_m']+2e-4
    return num
def assets():
    catalog=ROOT/'assets';rows=read(catalog/'asset_manifest.json')
    assert len(rows)==len({r['id'] for r in rows})==100
    counts=collections.Counter(a['category'] for a in rows);assert counts=={'pothole':34,'missing':33,'raised':33}
    records=[];descs=[]
    for a in sorted(rows,key=lambda a:a['id']):
        key=a['id'];out=(catalog/a['metadata']).parent
        meta=read(out/'metadata.json');assert meta['source_sha256']==a['source_sha256'] and a['size_multiplier']==1.
        ref=ROOT/a['source_image'];im=Image.open(ref).convert('RGB');assert im.width>=1024 and im.height==im.width
        descs.append(np.array(im.resize((80,80),Image.Resampling.LANCZOS),'f4'))
        assert sha(ref)==a['source_sha256'];gh,faces,ymin,ymax=glb(out/'asset.glb')
        assert faces==sum(a['geometry_parts'].values()) and abs(ymin-a['lowest_z_m'])<1e-5 and abs(ymax-a['highest_z_m'])<1e-5
        num=ply(out/'asset.ply',meta)
        with np.load(out/'splat_material.npz') as d:
            assert len(d['part'])==len(d['road'])==len(d['ambient_visibility'])==num and d['road'].any() and (~d['road']).any()
        with np.load(out/'surface.npz') as f:
            h,alpha=f['height'],f['alpha'];mask=f['defect_mask']>.5;assert h.shape==mask.shape==alpha.shape and np.isfinite(h).all() and np.isfinite(alpha).all()
            assert h[mask].min()<-.015 and h.max()<.025 and alpha.min()>=0 and alpha.max()<=1.00001 and mask.any()
            if a['category'] in ('raised','missing'):assert f['opening_mask'].sum()>20 and h[mask].min()<=-.4
        for view in ('inspection','reverse'):
            with Image.open(ROOT/'views'/(key+'_'+view+'.jpg')) as im:im.load();assert im.size==(840,568)
        records.append(dict(id=key,label=a['label'],category=a['category'],sha256_rgb=a['source_sha256'],sha256_mesh_geometry=gh,sha256_glb=sha(out/'asset.glb'),sha256_ply=sha(out/'asset.ply'),triangles=faces,gaussians=num,size_m=[a['width_m'],a['length_m'],a['depth_design_m']]))
    assert len({r['sha256_rgb'] for r in records})==len({r['sha256_mesh_geometry'] for r in records})==100
    nearest=min((float(abs(descs[i]-descs[j]).mean()),records[i]['id'],records[j]['id']) for i in range(len(records)) for j in range(i))
    report=dict(counts=counts,source_images=100,unique_geometry=100,total_triangles=sum(r['triangles'] for r in records),total_gaussians=sum(r['gaussians'] for r in records),nearest_source_pair_mae_80px=nearest,records=records)
    save(OUT/'asset_integrity.json',report);print('verified 100 GLB, PLY, sources, fields and 200 inspection renders',flush=True)
def scenes():
    asset_rows=read(OUT/'asset_integrity.json')['records'];known={a['id']:a for a in asset_rows}
    cases=sorted((ROOT/'scenes').glob('*/scene.json'));assert len(cases)==102
    stats=[];pairs=set();layers=['pothole','missing','raised'];ranges=[];checks=[]
    for file in cases:
        desc=read(file);case=desc['id'];site=desc['site'];receipt=read(file.parent/'complete.json')
        for path,h in receipt['files'].items():
            assert sha(file.parent/path)==h,(case,path)
        assert desc['actors_sky_unchanged'] and desc['checkpoint_reloaded']
        ds=ROOT/'sensor_views'/case;frame=read(ds/'frames.json')
        assert frame['native_signature']==receipt['signature'] and list(frame['assets'])==desc['assets'] and frame['size']==[1920,1080]
        labels={layers.index(p['category'])+1 for p in desc['placements']}
        with np.load(ds/'classes.npz') as f:
            mask=f['class_id'];indices=f['source_frames'];assert mask.shape==(40,1080,1920) and np.array_equal(indices,list(range(*[frame['source_frame_range_inclusive'][0],frame['source_frame_range_inclusive'][1]+1])))
            peaks=[0]*3
            for item,row,index in zip(mask,frame['frames'],indices):
                counts=np.bincount(item.ravel(),minlength=4)
                assert len(counts)==4 and row['class_pixels']==counts[1:].tolist() and index==row['frame']
                for j in range(3):
                    peaks[j]=max(peaks[j],int(counts[j+1]));box=row['class_bboxes_xyxy'][j]
                    if counts[j+1]==0:
                        assert box==[];continue
                    l,t,r,b=box;support=item[t:b,l:r]==j+1
                    assert support.sum()==counts[j+1] and all(v.any() for v in (support[0],support[-1],support[:,0],support[:,-1]))
            assert {i+1 for i,n in enumerate(peaks) if n>0}==labels
        assert (ds/'front_after.mp4').stat().st_size>4096
        for p in desc['placements']:
            k=p['asset_id'];a=known[k]
            assert p['asset_ply_sha256']==a['sha256_ply'] and p['reference_rgb_sha256']==a['sha256_rgb']
            assert not (site,k) in pairs;pairs.add((site,k))
            assert p['reload_max_pixel_error']<=1 and p['world_error_m']<.00002 and p['log_scale_error']<.000001 and p['size_multiplier']==1.
            category=p['category'];idx=layers.index(category)
            v=[n['class_pixels'][idx] for n in frame['frames']]
            useful=[j for j,n in enumerate(frame['frames']) if v[j]>0 and all(x>8 for x in n['class_bboxes_xyxy'][idx][:2]) and n['class_bboxes_xyxy'][idx][2]<1912 and n['class_bboxes_xyxy'][idx][3]<1072]
            best=max(useful or list(range(40)),key=lambda j:v[j])
            stats.append(dict(site=site,case=case,id=k,category=category,best_frame=int(indices[best]),best_box=frame['frames'][best]['class_bboxes_xyxy'][idx],max_visible_pixels=max(v),visible_frames=sum(n>0 for n in v),reload_max_channel_error=p['reload_max_pixel_error'],position_max_error_m=p['world_error_m'],depth_check=p['cavity_depth_check']))
        checks.append(dict(case=case,checkpoint_sha256=receipt['files']['trained_model/iteration_7000.pth'],checkpoint_signature=receipt['signature'],front_frames=40,asset_count=len(desc['assets'])))
    assert pairs=={(s,k) for s in ['034','109','110'] for k in known}
    save(OUT/'scene_integrity.json',dict(checkpoints=len(cases),placement_count=len(pairs),front_frames=len(cases)*40,visible_in_all_placements=all(r['visible_frames']>0 for r in stats),max_position_error_m=max(r['position_max_error_m'] for r in stats),max_reload_channel_error=max(r['reload_max_channel_error'] for r in stats),cases=checks,placements=stats))
    print('verified 102 checkpoints, 300 unique placements and 4080 label frames',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['assets','scenes','all']);args=p.parse_args();OUT.mkdir(exist_ok=True)
    if args.mode in ('assets','all'):assets()
    if args.mode in ('scenes','all'):scenes()
