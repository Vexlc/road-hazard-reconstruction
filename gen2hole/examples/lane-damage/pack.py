"""Package native assets with their field evidence and verifiable per-file digests."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
GROUPS={'point-pits':['P546','P547','P548','P549','P550'],
        'cover-and-water':['P551','M552','A553'],'large-road-failures':['P554','P555','P556']}


def archive(outfile, entries):
    hashes={}
    with ZipFile(outfile,'w',ZIP_DEFLATED,compresslevel=4) as z:
        for name,path in sorted(entries.items()):
            data=path if isinstance(path,bytes) else path.read_bytes()
            hashes[name]=hashlib.sha256(data).hexdigest()
            z.writestr(name,data)
        z.writestr('SHA256SUMS',''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    with ZipFile(outfile) as z:
        for name,digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest


def pack(models,views,materials,out):
    out.mkdir(parents=True,exist_ok=True)
    for group,keys in GROUPS.items():
        entries={}
        for key in keys:
            (folder,)=models.glob(f'*/series/assets/*/{key}')
            for p in folder.iterdir():
                if p.is_file():entries[f'{key}/{p.name}']=p
            for p in views.glob(key+'-*.jpg'):entries[f'{key}/preview/{p.name}']=p
            ref=folder.parents[2]/'references'/(key+'.jpg')
            entries[f'references/{key}.jpg']=ref
        entries['quality.json']=(json.dumps({k:json.loads((views/'quality.json').read_text())[k] for k in keys},indent=2)+'\n').encode()
        entries['SOURCES.md']=HERE/'SOURCES.md'
        archive(out/(group+'.zip'),entries)
    output=views.parent
    films={'SOURCES.md': HERE/'SOURCES.md','drive-check.json':output/'drive-check.json'}
    for p in (output/'109').glob('*/*/*'):
        if p.name in ('drive.mp4','placement.json','part.png','rays.npz'):
            films[str(p.relative_to(output))]=p
    for p in (output/'approach').glob('*'):films['approach/'+p.name]=p
    archive(out/'driving-views.zip',films)
    survey=json.loads((HERE/'points/survey.json').read_text())
    paths=[Path(d)/(d+'_diff_2k.jpg') for d in survey['materials']]+[Path(survey['road_material']['path'])]
    archive(out/'scan-materials.zip',{str(p):materials/p for p in paths})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','views','materials','out'):p.add_argument(key,type=Path)
    pack(**vars(p.parse_args()))
