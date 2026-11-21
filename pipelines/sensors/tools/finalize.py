from pathlib import Path
import json,hashlib,zipfile,struct
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
    a=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):a.update(data)
    return a.hexdigest()
checks=ROOT/'checks';scan=read(checks/'scan_audit.json');camera=read(checks/'camera_audit.json')
assert scan['asset_count']==109 and scan['placements']==327 and scan['placements_with_both_lidars']==327 and scan['counts']['case_frames']==8880
assert camera['views']==654 and len(camera['skipped'])==0 and read(checks/'film_audit.json')['frames_decoded']==78 and not read(checks/'browser_audit.json')['errors']
geometry=read(ROOT/'geometry/inventory.json');index=[]
for aid,g in sorted(geometry.items()):
    source=ROOT.parent/g['source_mesh'];meta=read(source.with_name('metadata.json'))
    assert sha(source)==g['mesh_sha256'] and sha(ROOT/g['file'])==g['grid_sha256']
    matches=[]
    for p in sorted((ROOT/'cases').glob('*.json')):
        info=read(p)
        for a in info['assets']:
            if a['asset_id']==aid:matches.append({'case':info['case'],'site':info['site'],'instance':a['instance']})
    assert len(matches)==3
    index.append(dict(asset=aid,category=g['category'],label=g['label'],source_reference_sha256=meta['source_sha256'],
        source_gaussian_sha256=sha(source.with_name('asset.ply')),source_glb_sha256=g['mesh_sha256'],
        geometry=g['file'],case_placements=matches))
for f in sorted((ROOT/'examples').glob('*.ply')):
    with f.open('rb') as h:
        header=''
        while not header.endswith('end_header\n'):
            header+=h.readline().decode()
        n=int(next(v.split()[-1] for v in header.splitlines() if v.startswith('element vertex')))
        types={'double':8,'float':4,'int':4,'uchar':1}
        stride=sum(types[l.split()[1]] for l in header.splitlines() if l.startswith('property'))
        assert f.stat().st_size==h.tell()+n*stride
assert len(list((ROOT/'examples').glob('*.ply')))==6
content=dict(schema=1,units='metres',assets=index,original_sites=['034','109','110'],cases=len(list((ROOT/'cases').glob('*.npz'))),
    raw_scans=240,case_frames=scan['counts']['case_frames'],camera_pairs=camera['views'],
    gaussian_and_mesh_source_packages=['道路病害图生三维','道路病害图生三维_新增100'],
    open_first='传感器对照.html',data_guide='使用说明.txt',demonstration='双向车载与雷达.mp4',examples='examples',
    audit_files=[str(p.relative_to(ROOT)) for p in sorted(checks.glob('*.json')) if p.name!='independent_samples.json'])
(ROOT/'index.json').write_text(json.dumps(content,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='files.sha256'
       and '__pycache__' not in p.parts and not p.name.endswith('.tmp')]
(ROOT/'files.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
files.append(ROOT/'files.sha256')
archive=ROOT.parent/'路面病害_传感器与几何增强.zip'
tmp=archive.with_suffix('.writing')
packed={'.npz','.webp','.mp4','.jpg','.png'}
with zipfile.ZipFile(tmp,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=3) as zipf:
    for p in files:
        zipf.write(p,str(p.relative_to(ROOT.parent)),compress_type=zipfile.ZIP_STORED if p.suffix in packed else zipfile.ZIP_DEFLATED,compresslevel=3)
with zipfile.ZipFile(tmp) as z:
    assert len(z.infolist())==len(files) and z.testzip() is None
    assert {v.filename:v.file_size for v in z.infolist()}=={str(p.relative_to(ROOT.parent)):p.stat().st_size for p in files}
tmp.replace(archive)
print('validated',len(files),'files;',archive.stat().st_size,'bytes',flush=True)
