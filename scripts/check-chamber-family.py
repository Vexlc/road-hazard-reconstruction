"""Check the identity of the delivered glTF parts, materials, poses and road placements."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mesh_parts(path):
    data = path.read_bytes()
    size, = struct.unpack_from('<I',data,12)
    gltf = json.loads(data[20:20+size])
    binary = memoryview(data)[28+size:]
    def view(i):
        descriptor = gltf['bufferViews'][i]
        offset = descriptor.get('byteOffset',0)
        return binary[offset:offset+descriptor['byteLength']]
    def accessor(i):
        a = gltf['accessors'][i]
        return np.frombuffer(view(a['bufferView']),{5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']],
                             offset=a.get('byteOffset',0),count=a['count']*{'VEC3':3,'VEC2':2,'SCALAR':1}[a['type']])
    result = {}
    for node in gltf['nodes']:
        if 'mesh' not in node:
            continue
        require(set(node) <= {'name','mesh','extras'},'Unexpected glTF node transform')
        part = gltf['meshes'][node['mesh']]
        primitive, = part['primitives']
        data = primitive['attributes']
        pos = accessor(data['POSITION']).reshape(-1,3)
        xyz = pos[:,[0,2,1]]*[1,-1,1]
        mat = gltf['materials'][primitive['material']]
        pbr = mat['pbrMetallicRoughness']
        digest = hashlib.sha256(accessor(primitive['indices']).tobytes()+accessor(data['TEXCOORD_0']).tobytes())
        for texture in (pbr['baseColorTexture'],pbr['metallicRoughnessTexture'],mat['normalTexture']):
            image = gltf['textures'][texture['index']]['source']
            digest.update(view(gltf['images'][image]['bufferView']))
        result[part['name']] = (xyz,digest.hexdigest())
    return result


def check(root):
    bundle = root/'series'
    spec = read(bundle/'calibration/survey.json')
    roles = ('frame','lid') if any(not s.get('absent') for s in spec['states']) else ('frame',)
    source_hashes = {s:hashlib.sha256((bundle/'references'/(s+'.png')).read_bytes()).hexdigest() for s in roles}
    checks = read(bundle/'checks.json')
    lid_diameter=spec.get('lid_diameter_m')
    if 'pose_capture' in spec:
        lid_diameter=read(bundle/'calibration/observed_pose.json')['lid_diameter_m']
    template = {}
    assets = {}
    all_scenes = {}
    for f in (bundle/'scenes').glob('*/scene.json'):
        scene = read(f)
        for p in scene['placements']:
            all_scenes.setdefault(p['asset_id'],{})[scene['site']] = (scene,p)
    anchors = {}
    for state in spec['states']:
        key = state['id']
        absent = state.get('absent',False)
        folder = bundle/'assets'/('missing' if absent else 'raised')/key
        meta = read(folder/'metadata.json')
        require(key in checks and meta['component_source_sha256']==source_hashes,key+' source hashes')
        require(hashlib.sha256((folder/'asset.glb').read_bytes()).hexdigest()==checks[key]['sha256']['asset.glb'], key+' GLB hash')
        parts = mesh_parts(folder/'asset.glb')
        lid_parts = {s for s in parts if s.startswith('Lid ')}
        require(lid_parts == (set() if absent else {'Lid face','Lid rolled edge','Lid underside with casting ribs'}),key+' lid parts')
        fixed = set(parts)-lid_parts
        if assets:
            require(fixed=={s for s in template if not s.startswith('Lid ')},key+' fixed parts')
        max_error = 0.
        if not absent:
            t = np.asarray(meta['lid_transform'])
            if state.get('pose_from_capture'):
                observed=read(bundle/'calibration/observed_pose.json')
                require(np.allclose(t,observed['transform'],atol=1e-12),key+' measured pose')
                require(np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-8) and
                        np.linalg.det(t[:3,:3])>.999999,key+' observed rigid transform')
            else:
                angle = np.radians(state['angle_deg'])
                rotation = np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
                pivot = np.array([-lid_diameter/2,0,0])
                if 'capture_record' in spec and not any(state['offset_m']):
                    pivot[2]=spec['lid_thickness_m']+.010
                require(np.allclose(t[:3,:3],rotation,atol=1e-12),key+' commanded angle')
                require(np.allclose((rotation@pivot+t[:3,3]-pivot)[:2],state['offset_m'],atol=1e-12),key+' commanded translation')
        for name,(vertices,signature) in parts.items():
            xyz = (vertices-t[:3,3])@t[:3,:3] if name in lid_parts else vertices
            if name not in template:
                template[name]=(xyz,signature)
            expected,old_signature=template[name]
            require(signature==old_signature,key+' changed UV, topology or PBR textures: '+name)
            error = float(np.abs(xyz-expected).max())
            require(error<1e-6 if name in lid_parts else error==0,key+' geometry drift: '+name)
            max_error=max(error,max_error)
        sites = {}
        for site,(scene,p) in all_scenes.get(key,{}).items():
            pose = [p['center_world'],p['basis_columns'],p['source_reference_frame'],p['slot']]
            require(anchors.setdefault(site,pose)==pose,key+' different world anchor on '+site)
            require(p['asset_ply_sha256']==checks[key]['sha256']['asset.ply'] and p['size_multiplier']==1 and
                    scene['checkpoint_reloaded'] and p['reload_max_pixel_error']==0,key+' placed checkpoint')
            sites[site]=dict(case=scene['id'],slot=p['slot'],reload_max_pixel_error=p['reload_max_pixel_error'])
        assets[key]=dict(state=state['label'],triangles=checks[key]['triangles'],gaussians=checks[key].get('gaussians'),
                         highest_z_m=meta['highest_z_m'],max_vertex_residual_m=max_error,scenes=sites)
    return dict(family=spec['family'],component_sha256=source_hashes,assets=assets,common_anchors=anchors,
                verified_parts=list(template),metric_scale=meta['scale_kind'],
                source_image_count=len(set(source_hashes.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--require-site',action='append',default=[])
    a=p.parse_args()
    report=check(a.run)
    for site in a.require_site:
        require(all(site in r['scenes'] for r in report['assets'].values()),site+' missing placements')
    content=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.out:a.out.write_text(content,encoding='utf-8')
    else:print(content,end='')
