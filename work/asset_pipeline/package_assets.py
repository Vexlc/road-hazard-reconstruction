"""Set glTF Y-up root transforms without recompressing the embedded PBR images."""
from pathlib import Path
import struct,json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'outputs/road_assets').glob('*/*_*.glb'):
    b=p.read_bytes();jl,jt=struct.unpack_from('<II',b,12);j=json.loads(b[20:20+jl])
    if j.get('extras',{}).get('authoring_z_up_to_gltf_y_up'):continue
    R=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)
    for scene in j.get('scenes',[]):
        nodes=scene.get('nodes',[]);idx=len(j['nodes']);j['nodes'].append({'name':'metres_Y_up','matrix':R.flatten(order='F').tolist(),'children':nodes});scene['nodes']=[idx]
    j.setdefault('extras',{})['authoring_z_up_to_gltf_y_up']=True
    s=json.dumps(j,separators=(',',':')).encode();s+=b' '*((-len(s))%4);rest=b[20+jl:]
    p.write_bytes(struct.pack('<4sII',b'glTF',2,20+len(s)+len(rest))+struct.pack('<II',len(s),jt)+s+rest)
    print(p.name,flush=True)
