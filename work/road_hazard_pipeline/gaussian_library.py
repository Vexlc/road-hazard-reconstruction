"""Create standalone Gaussian assets from ray-lit full 3D meshes."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/road_hazards'
sys.path.insert(0,str(ROOT/'work/asset_pipeline'))
from gaussian_assets import make_gaussians,save_ply,cpu_state,tensor
from sample_hazards import load_surface,CACHE

def camera():
    p=np.array([1.10,-1.50,1.82]);f=np.array([0,0,-.05])-p;f/=np.linalg.norm(f);r=np.cross(f,[0,0,1]);r/=np.linalg.norm(r);down=np.cross(f,r)
    m=np.eye(4);m[:3,:3]=np.stack([r,down,f],1);m[:3,3]=p
    return tensor(np.linalg.inv(m)),tensor([[1666.667,0,600],[0,1666.667,450],[0,0,1]])

@torch.no_grad()
def render(state,path):
    from gsplat.rendering import rasterization
    W,K=camera();rgb,alpha,_=rasterization(state['xyz'],state['rotation'],state['scaling'].exp(),state['opacity'].sigmoid()[:,0],torch.cat([state['feature_dc'],state['feature_rest']],1),W[None],K[None],1200,900,sh_degree=1,packed=False,backgrounds=tensor([[.52,.54,.56]]),rasterize_mode='antialiased')
    Image.fromarray(np.uint8(rgb[0].clamp(0,1).cpu().numpy()*255+.5)).save(path)

def main():
    catalog=json.loads((OUT/'catalog.json').read_text())
    for cfg in catalog:
        name=cfg['id'];d=OUT/name/'gaussians';d.mkdir(exist_ok=True)
        if (d/'metadata.json').exists():continue
        for _ in range(900):
            if (OUT/name/'sampling.json').exists() and (CACHE/(name+'.npz')).exists():break
            time.sleep(1)
        else:raise RuntimeError('sampling not ready: '+name)
        start=time.time();s=load_surface(name);state,error=make_gaussians(s)
        save_ply(state,d/'asset.ply');torch.save(cpu_state(state),d/'asset.pth');render(state,d/'preview.png')
        info={'id':name,'gaussians':len(state['xyz']),'representation':'anisotropic 3D Gaussian Splatting','sh_degree':1,'units':'metres','axes':'Z-up','sampling':json.loads((OUT/name/'sampling.json').read_text()),'appearance_sh_fit_mean_absolute_error':error,'geometry':cfg,'ray_baked_ambient_occlusion':True,'gradient_training':False,'license':'CC0-1.0','seconds':time.time()-start}
        (d/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2));print('GAUSSIANS',name,len(state['xyz']),time.time()-start,flush=True)
        del s,state;torch.cuda.empty_cache()

if __name__=='__main__':main()
