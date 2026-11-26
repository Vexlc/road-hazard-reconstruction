"""Package the six native assets, their evidence and the verified road-camera passes."""
import argparse
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lane-damage'))
# The previous release's pack module has the same filename.
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('lane_archive',HERE.parent/'lane-damage/pack.py')
module=module_from_spec(spec);spec.loader.exec_module(module)


def pack(models,output,out):
    out.mkdir(parents=True,exist_ok=True)
    for name,keys in [('scan-pairs',['P557','P558','P559']),('paint-and-breaks',['A560','M561','P562'])]:
        entries={'SOURCES.md':HERE/'SOURCES.md','README.md':HERE/'README.md','quality.json':output/'views/quality.json'}
        for key in keys:
            (folder,)=models.glob(f'*/series/assets/*/{key}')
            for f in folder.iterdir():entries[f'{key}/{f.name}']=f
            ref=HERE/'references'/(key+'.jpg')
            if ref.exists():entries[f'references/{key}.jpg']=ref
            for preview in (output/'views').glob(key+'-*.jpg'):entries[f'{key}/{preview.name}']=preview
        module.archive(out/(name+'.zip'),entries)
    films={'SOURCES.md':HERE/'SOURCES.md','camera-check.json':output/'camera-check.json'}
    for f in (output/'034').glob('*/*/*'):
        if f.name in ('drive.mp4','placement.json','rays.npz','part.png'):
            films[str(f.relative_to(output))]=f
    module.archive(out/'road-camera.zip',films)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','output','out'):p.add_argument(key,type=Path)
    pack(**vars(p.parse_args()))
