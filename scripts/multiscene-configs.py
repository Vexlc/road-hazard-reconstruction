#!/usr/bin/env python3
"""Generate machine-local config paths after restoring a new-road runtime package."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'tools/multiscene'
OUT = ROOT / 'outputs/道路病害_多场景迁入'
template = yaml.safe_load((ROOT / 'streetcrafter/overlay/configs/pandaset_056_warmup.yaml').read_text())
for site in ['034', '109', '110']:
    config = dict(template)
    config.update(parent_cfg=str(ROOT / 'work/street_crafter/configs/waymo_val_121.yaml'),
                  source_path=str(WORK / 'data/processed' / site),
                  model_path=str(OUT / '重建场景' / site),
                  record_dir=str(WORK / 'training_records' / site),
                  exp_name='road_diversity_' + site, resume=False)
    path = WORK / f'pandaset_{site}.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    folder = OUT / '迁入场景'
    for dest in folder.glob(site + '_*'):
        (dest / 'render_config.yaml').write_text(yaml.safe_dump({
            'parent_cfg': str(path), 'model_path': str(dest), 'mode': 'trajectory',
            'loaded_iter': 7000, 'resolution': 1280}))
    print('Configured', site)
