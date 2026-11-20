"""Keep dataset-annotated road returns, preserving raw LiDAR/label index order."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run import atomic_json,sha

ROAD_NAMES={'Road','Lane Line Marking','Stop Line Marking','Other Road Marking'}


def prepare(raw,output,sites):
    output.mkdir(parents=True,exist_ok=True)
    report={'source':'PandaSet original per-point semantic annotations',
            'mapping_reference':'https://github.com/scaleapi/pandaset-devkit#semantic-segmentation',
            'road_names':sorted(ROAD_NAMES),'scenes':{}}
    for site in sites:
        source=raw/site;dest=output/site;dest.mkdir(exist_ok=True)
        classes=json.loads((source/'annotations/semseg/classes.json').read_text(encoding='utf-8'))
        road_ids=[int(i) for i,name in classes.items() if name in ROAD_NAMES]
        assert len(road_ids)==4
        records=[]
        for frame in range(80):
            lidar_file=source/'lidar'/f'{frame:02d}.pkl.gz'
            label_file=source/'annotations/semseg'/f'{frame:02d}.pkl.gz'
            identity={'lidar_sha256':sha(lidar_file),'semseg_sha256':sha(label_file),
                      'class_map_sha256':sha(source/'annotations/semseg/classes.json')}
            path=dest/f'{frame:03d}.npz';stamp=path.with_suffix('.json')
            if path.exists() and stamp.exists():
                old=json.loads(stamp.read_text(encoding='utf-8'))
                if old.get('inputs')==identity and old.get('output_sha256')==sha(path):
                    records.append(old);continue
            p=pd.read_pickle(lidar_file);labels=pd.read_pickle(label_file)
            if not p.index.equals(labels.index) or not p.index.is_unique:
                raise ValueError(f'{site}/{frame}: raw LiDAR and semantic indices do not align')
            keep=np.isin(labels['class'].values,road_ids)
            xyz=p.loc[keep,['x','y','z']].to_numpy(dtype='f4')
            sensor=p.loc[keep,'d'].to_numpy(dtype='u1')
            np.savez_compressed(path,xyz=xyz,sensor=sensor,
                    intensity=p.loc[keep,'i'].to_numpy(dtype='f4'),
                    label=labels.loc[keep,'class'].to_numpy(dtype='u1'),
                    raw_index=p.index.values[keep].astype('i8'))
            record={'frame':frame,'inputs':identity,'output_sha256':sha(path),
                    'road_points':len(xyz),'forward_road_points':int((sensor==1).sum()),
                    'mechanical_road_points':int((sensor==0).sum())}
            atomic_json(stamp,record);records.append(record)
        report['scenes'][site]={'road_ids':road_ids,'frames':records}
        print('ANNOTATED ROAD READY',site,sum(r['road_points'] for r in records),flush=True)
    atomic_json(output/'provenance.json',report)


def read(cache,site,frame,sensor=1):
    with np.load(Path(cache)/site/f'{frame:03d}.npz') as data:
        return data['xyz'][data['sensor']==sensor]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',nargs='+',required=True)
    a=p.parse_args();prepare(a.raw,a.output,a.scenes)
