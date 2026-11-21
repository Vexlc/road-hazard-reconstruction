"""Binary format checks from the original 100-asset release."""
import hashlib
import json
import struct
import numpy as np

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
        assert f.readline().strip()==b'format binary_little_endian 1.0'
        num=0
        while True:
            line=f.readline().decode().strip()
            if not line:raise ValueError(p)
            if line=='end_header':break
            if line.startswith('element vertex'):num=int(line.split()[-1])
            if line.startswith('property '):
                prop=line.split();assert prop[1]=='float';names.append(prop[-1])
        offset=f.tell()
    expected=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(9)]
    expected+=['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    assert names==expected and num==meta['gaussians']
    assert p.stat().st_size-offset==4*num*len(names)
    values=np.memmap(p,mode='r',dtype='<f4',offset=offset,shape=(num,len(names)))
    for i in range(0,num,65536):
        a=values[i:i+65536];assert np.isfinite(a).all()
        assert a[:,2].min()>=meta['lowest_z_m']-2e-4 and a[:,2].max()<=meta['highest_z_m']+2e-4
    return num
