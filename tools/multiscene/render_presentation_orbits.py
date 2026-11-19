"""Render a full 360-degree turntable for every one of the fifty chosen assets."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import sys

import imageio.v2 as imageio
from PIL import Image
import torch

WORK=Path(__file__).parent
sys.path.insert(0,str(WORK.parent/'visualization'))
import render_catalog as c

c.IW,c.IH=760,570
c.FRAMES=96

def main():
    torch.set_num_threads(4)
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((c.LIB/'asset_manifest.json').read_text(encoding='utf-8'))}
    dest=WORK/'composite3/orbits';dest.mkdir(parents=True,exist_ok=True)
    for cat,ids in plan['selected_assets'].items():
        for asset in ids:
            path=dest/(asset+'.mp4')
            if path.exists():continue
            frames=c.render_asset(models[asset])
            with imageio.get_writer(str(path),fps=24,codec='libx264',quality=8,macro_block_size=2,
                  output_params=['-threads','2','-movflags','+faststart']) as video:
                for frame in frames:video.append_data(frame)
            print('ORBIT',cat,asset,flush=True)
    (dest/'complete.json').write_text(json.dumps({'assets':50,'frames_per_asset':96,'fps':24,
          'seconds_per_asset':4,'azimuth_span_degrees':360,'resolution':[760,570]},indent=2),encoding='utf-8')

if __name__=='__main__':main()
