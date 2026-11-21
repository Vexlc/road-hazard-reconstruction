"""Independently decode delivered GLB geometry, PLYs, source registration and checkpoint hashes."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    sha=sha256()
    with path.open('rb') as f:
        for s in iter(lambda:f.read(4*1024*1024),b''):sha.update(s)
    return sha.hexdigest()


def decode(path):
    buf=path.read_bytes()
    assert buf[:4]==b'glTF' and struct.unpack_from('<II',buf,4)==(2,len(buf))
    n=struct.unpack_from('<I',buf,12)[0];g=json.loads(buf[20:20+n]);binary=memoryview(buf)[28+n:]
    def read(aid):
        a=g['accessors'][aid];b=g['bufferViews'][a['bufferView']]
        dim={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        dt=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']])
        return np.ndarray((a['count'],dim),dt,buffer=binary,
                          offset=b.get('byteOffset',0)+a.get('byteOffset',0),
                          strides=(b.get('byteStride',dim*dt.itemsize),dt.itemsize))
    texture_sizes=[]
    for im in g['images']:
        b=g['bufferViews'][im['bufferView']];s=b.get('byteOffset',0)
        with Image.open(BytesIO(binary[s:s+b['byteLength']])) as tex:
            tex.load();texture_sizes.append(list(tex.size))
    v,f,ids,names=[],[],[],[]
    total=0
    def visit(i,parent):
        nonlocal total
        node=g['nodes'][i]
        m=np.eye(4)
        if 'matrix' in node:m=np.array(node['matrix']).reshape(4,4).T
        else:
            m[:3,:3]=Rotation.from_quat(node.get('rotation',[0,0,0,1])).as_matrix()@np.diag(node.get('scale',[1]*3))
            m[:3,3]=node.get('translation',[0]*3)
        world=parent@m
        if 'mesh' in node:
            mesh=g['meshes'][node['mesh']]
            for prim in mesh['primitives']:
                assert prim.get('mode',4)==4
                att=prim['attributes']
                p=read(att['POSITION'])@world[:3,:3].T+world[:3,3]
                p=p[:,[0,2,1]]*[1,-1,1]
                uv=read(att['TEXCOORD_0']);normal=read(att['NORMAL'])
                assert np.isfinite(p).all() and np.isfinite(uv).all() and np.isfinite(normal).all()
                tris=read(prim['indices']).reshape(-1,3).astype('i4')
                assert tris.max()<len(p) and tris.min()>=0
                mat=g['materials'][prim['material']];pbr=mat['pbrMetallicRoughness']
                for tex in [pbr['baseColorTexture'],pbr['metallicRoughnessTexture'],mat['normalTexture']]:
                    assert g['textures'][tex['index']]['source'] <len(texture_sizes)
                v.append(p.astype('f4'));f.append(tris+total);total+=len(p)
                ids.append(np.full(len(tris),len(names),'i2'));names.append(node.get('name',mesh.get('name')))
        for k in node.get('children',[]):visit(k,world)
    for n in g['scenes'][g.get('scene',0)]['nodes']:visit(n,np.eye(4))
    return np.concatenate(v),np.concatenate(f),np.concatenate(ids),np.array(names),texture_sizes


def iou(a,b):
    return float(np.sum(a&b)/max(np.sum(a|b),1))


def check_asset(row, key, cfg):
    root=ROOT/'assets';d=(root/row['glb']).parent
    v,f,labels,names,textures=decode(root/row['glb'])
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*.5
    if (area<=1e-13).any():raise ValueError('Zero-area triangles '+key)
    scene=o3d.t.geometry.RaycastingScene(nthreads=5)
    scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f.astype('u4')))
    verts=PlyData.read(root/row['ply'],mmap='r')['vertex'].data
    assert len(verts)==row['gaussians']
    rng=np.random.default_rng(41387)
    chosen=verts[rng.choice(len(verts),8192,replace=False)]
    splats=np.stack([chosen[k] for k in 'xyz'],1)
    nearest=scene.compute_closest_points(o3d.core.Tensor(splats.copy()),nthreads=5)['points'].numpy()
    errors=np.linalg.norm(nearest-splats,axis=1)
    if errors.max()>.0002:raise ValueError('Gaussian off exported GLB: '+key)
    n=481
    cols,rows=np.meshgrid(np.linspace(0,479,n),np.linspace(0,479,n))
    cx,cy,px,py,_=row['road_projection']
    x,y=(cols-cx)/px,(cy-rows)/py
    rays=np.stack([x,y,np.ones_like(x)*1.8,np.zeros_like(x),np.zeros_like(x),np.full_like(x,-1)],-1).astype('f4')
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=5)
    z=1.8-hit['t_hit'].numpy()
    idx=hit['primitive_ids'].numpy()
    valid=np.isfinite(z)
    parts=np.full(idx.shape,-1,dtype='i2');parts[valid]=labels[idx[valid]]
    reg={}
    if cfg['category']=='pothole':
        mask=np.asarray(Image.open(ROOT/'calibration'/(key+'_outline.png')).resize((n,n),Image.Resampling.NEAREST))>127
        body=np.isin(parts,np.flatnonzero([name.startswith(('Water','Fracture')) for name in names]))
        reg.update(observed_planform_mask_iou=iou(mask,body),depression_over_5mm_iou=iou(mask,valid&(z<-.005)),
                   deepest_vertical_ray_m=float(-z[valid].min()))
        assert reg['observed_planform_mask_iou']>.97
        if key=='wet':
            wet=parts==np.flatnonzero(names=='Water surface')[0]
            reg['water_height_p05_p95_m']=np.percentile(z[wet],[5,95]).tolist()
            assert np.ptp(reg['water_height_p05_p95_m'])<.002
    else:
        cx,cy,ax,ay=cfg['inner']
        if cfg['shape']=='round': mask=((cols-cx)/ax)**2+((rows-cy)/ay)**2<.81
        else:mask=np.maximum(abs(cols-cx)/ax,abs(rows-cy)/ay)<.90
        reg['opening_rays_below_10cm_fraction']=float((z[mask]<-.10).mean())
        if cfg['category']=='missing':
            assert reg['opening_rays_below_10cm_fraction']>.99
        if 'lid' in cfg:
            lid=cfg['lid']
            top=parts==np.flatnonzero(names=='Lid face')[0]
            if 'ellipse' in lid:
                lx,ly,rx,ry=lid['ellipse']
                vnorm=(ly-rows)/ry
                ref=((cols-lx-lid.get('bow_x',0)*vnorm**2)/rx)**2+vnorm**2<1
            else:
                from PIL import ImageDraw
                poly=Image.new('1',(n,n))
                ImageDraw.Draw(poly).polygon([tuple(np.array(p)*(n-1)/479) for p in lid['quad']],fill=1)
                ref=np.asarray(poly,dtype=bool)
            reg['cover_silhouette_iou_vs_reference_fit']=iou(top,ref)
            p=v[np.unique(f[labels==np.flatnonzero(names=='Lid face')[0]])]
            eigvec=np.linalg.eigh(np.cov(p.T))[1][:,0]
            tilt=float(np.rad2deg(np.arccos(abs(eigvec[2]))))
            reg['plane_fit_lid_degrees']=tilt
            assert abs(tilt-row['lid_angle_degrees'])<1.2
            assert reg['cover_silhouette_iou_vs_reference_fit']>.77
            assert reg['opening_rays_below_10cm_fraction']>.09
    expected=sha256((ROOT/row['source_image']).read_bytes()).hexdigest()
    assert expected==row['source_sha256']
    return dict(id=row['id'],source=key,triangles=len(f),glb_parts=len(names),material_texture_images=len(textures),
                gaussian_points=len(verts),gaussian_surface_error_max_m_8192_samples=float(errors.max()),
                lowest_z_m=float(v[:,2].min()),highest_z_m=float(v[:,2].max()),registration=reg,
                source_sha256=expected,glb_sha256=digest(root/row['glb']),ply_sha256=digest(root/row['ply']))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--with-scenes',action='store_true')
    args=ap.parse_args()
    library=json.loads((ROOT/'assets/asset_manifest.json').read_text(encoding='utf8'))
    layout=json.loads((ROOT/'calibration/layout.json').read_text(encoding='utf8'))
    by_id={r['id']:r for r in library}
    out=ROOT/'checks';out.mkdir(exist_ok=True)
    assets=[check_asset(by_id[spec['id']],key,spec) for key,spec in layout.items()]
    report={'assets':assets,'registration_grid':[481,481],'reference_note':'Image-space consistency against source-image contours. Metric scale and hidden geometry are model assumptions.'}
    if args.with_scenes:
        cases=[]
        for f in sorted((ROOT/'scenes').glob('*/complete.json')):
            complete=json.loads(f.read_text(encoding='utf8'))
            assert all(digest(f.parent/k)==h for k,h in complete['files'].items())
            case=json.loads((f.parent/'scene.json').read_text(encoding='utf8'))
            assert case['checkpoint_reloaded'] and case['actors_sky_unchanged']
            group=[]
            for p in case['placements']:
                assert p['asset_points']==by_id[p['asset_id']]['gaussians'] and p['size_multiplier']==1.
                assert p['reload_max_pixel_error']<=1 and p['world_error_m']<2e-5 and p['log_scale_error']==0
                group.append({k:p[k] for k in ['asset_id','world_error_m','log_scale_error','reload_max_pixel_error','actual_footprint_color_coverage','cavity_depth_check','rms_m']})
            cases.append({'id':case['id'],'checkpoint_sha256':complete['files']['trained_model/iteration_7000.pth'],'placements':group})
        assert len(cases)==9 and sum(len(c['placements']) for c in cases)==27
        report['native_scenes']=cases
    (out/'geometry_and_scenes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for a in assets:print(a['id'],a['registration'])
    print('Native cases',len(report.get('native_scenes',[])),flush=True)


if __name__=='__main__':main()
