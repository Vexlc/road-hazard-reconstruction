"""Prepare a sequence on CPU while the GPU reconstructs another sequence."""
import json
import sys
from prepare_train import WORK,REPO,run

scene=sys.argv[1]
for stage,keys in [('prepare',['timestamps','images','calib','pose','dynamic_masks','objects']),('lidar',['lidar_forward'])]:
    status=WORK/f'{stage}_{scene}_status.json'
    if status.exists() and json.loads(status.read_text())['returncode']==0:continue
    run(scene,stage,[REPO/'data_processor/pandaset_processor/pandaset_convertor.py',
        '--data_root',WORK/'data/raw','--target_dir',WORK/'data/processed','--scene_ids',int(scene),
        '--workers',1,'--process_keys',*keys],REPO/'data_processor')
