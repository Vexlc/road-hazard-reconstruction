"""Ray-cast top surfaces and exposed substrates from the atlas GLBs into local metric grids."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
from pathlib import Path
import struct
import tempfile

import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parents[1]
ATLAS=ROOT/'tools/asset_atlas'
PREFIX='window.roadAssets='


def cast_grid(scene, xx, yy, start):
    rays=np.column_stack([xx.ravel(),np.full(xx.size,start),-yy.ravel(),
                          np.zeros(xx.size),-np.ones(xx.size),np.zeros(xx.size)]).astype('f4')
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=2)['t_hit'].numpy()
    return (start-hit).reshape(xx.shape)


def read_mesh(path,separate):
    full=o3d.t.geometry.RaycastingScene(nthreads=2)
    underlying=o3d.t.geometry.RaycastingScene(nthreads=2) if separate else None
    bounds=[];saw_layer=False;fixed_parts=0
    with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as blob:
        magic,version,total,js_size,tag=struct.unpack_from('<4sIIII',blob)
        if (magic,version,total,tag)!=(b'glTF',2,len(blob),0x4e4f534a):
            raise ValueError('Expected self-contained GLB 2: '+str(path))
        header=json.loads(blob[20:20+js_size])
        binary=28+js_size
        def accessor(index):
            a=header['accessors'][index];v=header['bufferViews'][a['bufferView']]
            dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
            dim={'VEC3':3,'SCALAR':1}[a['type']]
            if v.get('byteStride',dim*dtype.itemsize)!=dim*dtype.itemsize or 'sparse' in a:
                raise ValueError('Unexpected strided/sparse mesh accessor')
            return np.frombuffer(blob,dtype=dtype,count=a['count']*dim,
                        offset=binary+v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,dim).copy()
        for idx in header['scenes'][header.get('scene',0)]['nodes']:
            node=header['nodes'][idx]
            if set(node)-{'name','mesh','extras'}:raise ValueError('Expected world-aligned atlas GLB '+str(path))
            mesh=header['meshes'][node['mesh']]
            layer=mesh.get('name','').startswith(('Lid ','Hinge ')) or mesh.get('name')=='Water surface'
            saw_layer |= layer
            for primitive in mesh['primitives']:
                if primitive.get('mode',4)!=4:raise ValueError('Only triangles can be queried')
                xyz=accessor(primitive['attributes']['POSITION'])
                faces=accessor(primitive['indices']).reshape(-1,3).astype('uint32')
                bounds.append((xyz.min(axis=0),xyz.max(axis=0)))
                pos=o3d.core.Tensor(xyz);tri=o3d.core.Tensor(faces)
                if underlying is not None and not layer:
                    underlying.add_triangles(pos,tri);fixed_parts+=1
                else:
                    full.add_triangles(pos,tri)
        digest=hashlib.sha256(blob).hexdigest()
    lo=np.min([b[0] for b in bounds],axis=0);hi=np.max([b[1] for b in bounds],axis=0)
    if underlying is not None and not saw_layer:
        full,underlying=underlying,None
    return full,underlying if saw_layer and fixed_parts else None,lo,hi,digest


def encode(grid,low,step):
    valid=np.isfinite(grid);code=np.zeros(grid.shape,dtype='<u2')
    code[valid]=1+np.rint((grid[valid]-low)/step).clip(0,65534).astype('<u2')
    return base64.b64encode(code.tobytes()).decode('ascii')


def completion_profile(row, x, y, start, bounds):
    from pit_texture_eval import read_texture

    paths = {kind: ATLAS / path for kind, path in row['completionModels'].items()}
    ref, _, lo, hi, digest = read_mesh(paths['rangeOnly'], False)
    axes = [0, 2]
    if not all(np.allclose(a[axes], b[axes], atol=1e-7, rtol=0) for a, b in zip(bounds, (lo, hi))):
        raise ValueError('Comparison surfaces have different footprints for ' + row['id'])
    height = cast_grid(ref, x, y, max(start, hi[1] + 0.25))
    width = float(hi[0] - lo[0])
    image = read_texture(paths['support'], width)
    u = np.rint((0.5 + x / width) * (image.shape[1] - 1)).astype(int)
    v = np.rint((0.5 - y / width) * (image.shape[0] - 1)).astype(int)
    sample = image[v, u]
    classes = np.full(x.shape, 255, dtype='u1')
    for i, color in enumerate([(230, 145, 68), (67, 91, 112), (59, 193, 188), (220, 220, 210)]):
        classes[(sample == color).all(axis=-1)] = i
    if (classes == 255).any():
        raise ValueError('Unrecognized height-source color for ' + row['id'])
    return height, dict(
        range_only_glb_sha256=digest,
        support_glb_sha256=hashlib.sha256(paths['support'].read_bytes()).hexdigest(),
        support=base64.b64encode(classes.tobytes()).decode('ascii'),
    )


def build(row,dest,resume):
    key=row['id'];path=ATLAS/row['glb']
    output=dest/(key+'.js')
    if resume and output.is_file() and not row.get('completionModels'):
        # Merely matching an identifier is insufficient when a user has rebuilt the GLB.
        old=output.read_text(encoding='utf-8')
        if old.startswith('window.roadRelief(') and old.endswith(');\n'):
            name,grid=json.loads('['+old[len('window.roadRelief('):-3]+']')
            with path.open('rb') as stream:
                sha=hashlib.sha256()
                for piece in iter(lambda:stream.read(4<<20),b''):sha.update(piece)
            if (name == key and grid.get('glb_sha256') == sha.hexdigest() and 'rangeOnly' not in grid
                    and (not row.get('water') or 'under' in grid)):
                code=np.frombuffer(base64.b64decode(grid['top'],validate=True),'<u2').reshape(grid['n'],grid['n'])
                return dict(id=key,samples=int(np.count_nonzero(code)),substrate='under' in grid)
    full,under,lo,hi,digest=read_mesh(path,row['category'] in ('异常井盖','落座参照') or bool(row.get('throughHoles') or row.get('water')))
    n=193 if row['category']=='裂缝' or row.get('panelSize') or row.get('water') or row.get('completionModels') else 97
    # The grid covers cell centres, avoiding ambiguous triangle intersections on exterior edges.
    xpitch=(hi[0]-lo[0])/n;ypitch=(hi[2]-lo[2])/n
    xs=np.linspace(lo[0]+xpitch/2,hi[0]-xpitch/2,n)
    ys=np.linspace(-lo[2]-ypitch/2,-hi[2]+ypitch/2,n)
    x,y=np.meshgrid(xs,ys)
    start=float(hi[1])+.25
    top=cast_grid(full,x,y,start)
    bottom=cast_grid(under,x,y,start) if under is not None else None
    if bottom is not None:
        # Query the two disjoint triangle sets once. Their pointwise maximum is
        # the first return of the complete GLB for these descending rays.
        top=np.maximum(top,bottom)
    valid=top[np.isfinite(top)]
    if valid.size==0:raise ValueError('No GLB intersections for '+key)
    low=float(valid.min());high=float(valid.max())
    if bottom is not None:
        shared=np.isfinite(top)&np.isfinite(bottom)
        if np.any(bottom[shared]>top[shared]+.001):
            raise ValueError('Substrate above top in '+key)
        low=min(low,float(bottom[shared].min()))
    baseline = None
    if row.get('completionModels'):
        baseline, comparison = completion_profile(row, x, y, start, (lo, hi))
        if not np.isfinite(baseline).all():
            raise ValueError('Incomplete range-only profile for ' + key)
        low = min(low, float(baseline.min()))
        high = max(high, float(baseline.max()))
    step=max(.00005,(high-low)/65532)
    obj=dict(n=n,x=[float(xs[0]),float(xs[-1])],y=[float(ys[0]),float(ys[-1])],
             low=low,step=step,top=encode(top,low,step),glb_sha256=digest)
    if bottom is not None:obj['under']=encode(bottom,low,step)
    if baseline is not None:
        obj.update(comparison, rangeOnly=encode(baseline, low, step))
    with tempfile.NamedTemporaryFile(dir=dest,mode='w',delete=False,encoding='utf-8') as f:
        f.write('window.roadRelief('+json.dumps(key)+','+json.dumps(obj,separators=(',',':'))+');\n')
    Path(f.name).replace(dest/(key+'.js'))
    return dict(id=key,samples=int(valid.size),substrate=bottom is not None,pitch_m=float(max(xpitch,ypitch)),
                min_visible_m=float(valid.min()),max_visible_m=high)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids',nargs='+');p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--resume',action='store_true',help='Reuse completed grids whose source GLB digest still matches')
    args=p.parse_args()
    catalog=(ATLAS/'catalog.js').read_text(encoding='utf-8')
    if not catalog.startswith(PREFIX):raise ValueError('Build the asset atlas first')
    rows=json.loads(catalog[len(PREFIX):].rstrip(';\n'))
    if args.ids:
        rows=[r for r in rows if r['id'] in set(args.ids)]
        if len(rows)!=len(set(args.ids)):raise ValueError('Unknown atlas ID')
    dest=ATLAS/'relief';dest.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records=[]
        for record in pool.map(lambda r:build(r,dest,args.resume),rows):
            records.append(record)
            if len(records)%100==0:print(len(records),'GLBs',flush=True)
    print(json.dumps(dict(assets=len(records),valid_hits=sum(r['samples'] for r in records),
                          substrates=sum(r['substrate'] for r in records)),indent=2))


if __name__=='__main__':main()
