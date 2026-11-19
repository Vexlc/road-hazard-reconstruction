from create_scene import create_scene
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.config import cfg
from gsplat.rendering import rasterization
import torch,json
from pathlib import Path

@torch.no_grad()
def main():
    scene=create_scene();pc=scene.gaussians;renderer=StreetGaussianRenderer();cam=next(c for c in scene.getTestCameras() if c.meta['frame']==40)
    renderer.render(cam,pc)
    root=Path(cfg.model_path);s=json.loads((root/'summary.json').read_text());start=s['original_background_gaussians_retained'];results={}
    xyz=pc.get_xyz;rot=pc.get_rotation;scale=pc.get_scaling;opacity=pc.get_opacity[:,0]
    for a in s['assets']:
        stop=start+a['gaussians'];colors=torch.zeros((len(xyz),3),device='cuda');colors[:s['original_background_gaussians_retained'],2]=1
        asset_end=s['original_background_gaussians_retained']+sum(x['gaussians'] for x in s['assets'])
        colors[asset_end:len(pc.background._xyz),1]=1
        colors[start:stop,0]=1
        rgb,_,_=rasterization(xyz,rot,scale,opacity,colors,cam.get_extrinsic(world2cam=True)[None],cam.K[None],cam.image_width,cam.image_height,packed=False,rasterize_mode='antialiased')
        _,alpha,_=rasterization(xyz[start:stop],rot[start:stop],scale[start:stop],opacity[start:stop],colors[start:stop],cam.get_extrinsic(world2cam=True)[None],cam.K[None],cam.image_width,cam.image_height,packed=False,rasterize_mode='antialiased')
        mask=alpha[0,...,0]>.98
        ratio=rgb[0,...,0][mask]/alpha[0,...,0][mask]
        results[a['name']]={'opaque_pixels':int(mask.sum()),'joint_visible_fraction_mean':float(ratio.mean()),'joint_visible_fraction_min':float(ratio.min()),'original_road_weight_mean':float(rgb[0,...,2][mask].mean()),'resampled_road_weight_mean':float(rgb[0,...,1][mask].mean())}
        start=stop
    (root/'occlusion_check.json').write_text(json.dumps(results,indent=2));print(results)

if __name__=='__main__':main()
