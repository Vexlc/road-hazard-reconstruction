"""Frame the aperture more tightly during the existing real-geometry turntable."""
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import sys
import numpy as np
import torch
import imageio.v2 as imageio
WORK=Path(__file__).parent
sys.path.insert(0,str(WORK.parent/'visualization'))
import render_catalog as c
c.IW,c.IH,c.FRAMES=760,570,96

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    plan=json.loads((WORK/'plan.json').read_text());models={m['id']:m for m in json.loads((c.LIB/'asset_manifest.json').read_text())}
    out=WORK/'composite4/orbits';out.mkdir(parents=True,exist_ok=True)
    for name in plan['selected_assets']['stereo']:
        path=out/(name+'.mp4')
        if path.exists():continue
        m=models[name];state,_,_=c.load_asset(m);xyz=state['means'].cpu().numpy()
        support=xyz[(abs(xyz[:,0])<m['width_m']*.78)&(abs(xyz[:,1])<m['length_m']*.78)]
        low,high=np.quantile(support[:,2],[.005,.995]);center=np.array([0.,0.,(low+high)*.5])
        support=support[::max(1,len(support)//4000)];views,k=c.cameras(center,support,'stereo')
        with imageio.get_writer(str(path),fps=24,codec='libx264',quality=8,macro_block_size=2,
              output_params=['-threads','2','-movflags','+faststart']) as video:
            for start in range(0,96,8):
                rgb,_,_=c.rasterization(**state,viewmats=views[start:start+8],Ks=k.expand(8,-1,-1),width=760,height=570,
                  sh_degree=1,packed=False,backgrounds=c.tensor([[.23,.27,.30]]).expand(8,-1),rasterize_mode='antialiased')
                assert bool(torch.isfinite(rgb).all())
                for frame in (rgb.clamp(0,1)*255+.5).to(torch.uint8).cpu().numpy():video.append_data(frame)
        print('TIGHT ORBIT',name,flush=True)

if __name__=='__main__':main()
