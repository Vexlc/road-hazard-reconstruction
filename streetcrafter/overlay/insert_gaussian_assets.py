"""Insert actual Gaussian assets into the dynamic Gaussian model, one joint pass.

No mesh rasterizer, depth-based image compositing or decal overlay is used here.
"""
from pathlib import Path
import sys,json,time,os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates
import imageio.v2 as imageio
from create_scene import create_scene
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from export_reconstruction import export_splat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/asset_pipeline'))
from gaussian_assets import sample_surface,make_gaussians,save_ply,cpu_state,tensor
from gaussian_road import road_cut_mask,make_road_apron,surface_path

def look_at(position,target):
    p=np.array(position,float);f=np.array(target,float)-p;f/=np.linalg.norm(f)
    r=np.cross(f,[0,0,1]);r/=np.linalg.norm(r);down=np.cross(f,r)
    c=np.eye(4);c[:3,:3]=np.stack([r,down,f],1);c[:3,3]=p
    return tensor(np.linalg.inv(c))

@torch.no_grad()
def preview_asset(state,path):
    from gsplat.rendering import rasterization
    w2c=look_at([1.05,-1.4,1.45],[0,0,-.025]);K=tensor([[1075,0,352],[0,1075,276],[0,0,1]])
    rgb,alpha,_=rasterization(state['xyz'],state['rotation'],state['scaling'].exp(),state['opacity'].sigmoid()[:,0],torch.cat([state['feature_dc'],state['feature_rest']],1),w2c[None],K[None],704,552,sh_degree=1,packed=False,backgrounds=tensor([[.52,.54,.56]]),rasterize_mode='antialiased')
    Image.fromarray(np.uint8(rgb[0].clamp(0,1).cpu().numpy()*255)).save(path)

@torch.no_grad()
def main():
    start=time.time();scene=create_scene();renderer=StreetGaussianRenderer()
    sid=Path(cfg.source_path).name;label='A' if sid=='056' else 'B'
    out=ROOT/'outputs'/f'gs_inserted_{label}_{sid}';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'outputs'/f'inserted_{label}_{sid}'/'placement.json').read_text())
    hazard_id=os.environ.get('ROAD_HAZARD_ID')
    if hazard_id:
        import copy
        sys.path.insert(0,str(ROOT/'work/road_hazard_pipeline'))
        from sample_hazards import load_surface
        placement=copy.deepcopy(manifest['placements'][1 if 'pothole' in hazard_id else 0]);placement['asset']=hazard_id
        manifest['placements']=[placement]
        out=ROOT/'outputs/road_hazard_insertions'/f'{label}_{sid}_{hazard_id}';out.mkdir(parents=True,exist_ok=True)
    g=scene.gaussians.background;world=g._xyz.detach().cpu().numpy();cut=np.zeros(len(world),bool)
    additions=[];conversion=[]
    for placement in manifest['placements']:
        name=placement['asset'];d=ROOT/'outputs/gaussian_assets'/name
        if not hazard_id:d.mkdir(parents=True,exist_ok=True)
        if hazard_id:
            d=ROOT/'outputs/road_hazards'/name/'gaussians';d.mkdir(exist_ok=True)
            s=load_surface(name,np.array(placement['basis_columns']))
            selection=(s['road_mask']>.5)&(s['alpha']>.9)
            source=np.median(s['color'][selection,:3],axis=0);target=np.array(placement['calibration']['target_srgb'])
            def lin(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
            sl=float(lin(source)@np.array([.2126,.7152,.0722]));tl=float(lin(target)@np.array([.2126,.7152,.0722]))
            placement['calibration']={'source_srgb':source.tolist(),'target_srgb':target.tolist(),'linear_gain':[float(np.clip(tl/max(sl,.01),.08,3))]*3,'neutral_color_fraction':.75,'target_tint':np.clip(lin(target)/max(tl,.01),.8,1.2).tolist()}
        else:s=sample_surface(name)
        if not (d/'asset.ply').exists():
            canonical,error=make_gaussians(s)
            save_ply(canonical,d/'asset.ply');torch.save(cpu_state(canonical),d/'asset.pth');preview_asset(canonical,d/'preview.png')
            (d/'metadata.json').write_text(json.dumps({'name':name,'representation':'anisotropic 3D Gaussian Splatting','gaussians':len(canonical['xyz']),'units':'metres','axes':'Z-up','sh_degree':1,'source':'scan height field and PBR maps','conversion':'surface sampling, tangent covariance, PBR-to-SH least squares','gradient_training':False,'appearance_sh_fit_mean_absolute_error':error,'normal_sigma_m':.0003},indent=2))
            del canonical
        c=np.array(placement['center_world']);b=np.array(placement['basis_columns'])
        gs,error=make_gaussians(s,c,b,placement['calibration']);additions.append(gs)
        save_ply(gs,out/f'{name}_world.ply')
        surface=np.load(surface_path(name));local=(world-c)@b
        row=(.5-local[:,1]/float(surface['length']))*(surface['alpha'].shape[0]-1);col=(local[:,0]/float(surface['width'])+.5)*(surface['alpha'].shape[1]-1)
        alpha=map_coordinates(surface['alpha'],[row,col],order=1,mode='constant',cval=0)
        cut|=(alpha>.015)&(abs(local[:,2])<.35)
        conversion.append({'name':name,'gaussians':len(gs['xyz']),'appearance_sh_fit_mean_absolute_error':error,'placement':placement})
        print('converted',sid,name,len(gs['xyz']),error,flush=True)
        del s
    cameras=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
    cut=road_cut_mask(g,manifest['placements'])
    apron,apron_info=make_road_apron(scene,renderer,cameras,manifest['placements'])
    save_ply(apron,out/'resampled_road.ply');additions.append(apron)
    print('road resampled',sid,int(cut.sum()),apron_info,flush=True)
    keep=tensor(~cut).bool();base_count=int(keep.sum())
    for attr,key in [('_xyz','xyz'),('_features_dc','feature_dc'),('_features_rest','feature_rest'),('_scaling','scaling'),('_rotation','rotation'),('_opacity','opacity'),('_semantic','semantic')]:
        old=getattr(g,attr);value=torch.cat([old[keep]]+[a[key] for a in additions],0)
        setattr(g,attr,torch.nn.Parameter(value,requires_grad=True))
    del additions
    summary={'scene':sid,'representation':'3DGS assets appended to background Gaussian parameters','rendering':'assets, road and dynamic objects in one joint gsplat pass; original sky cubemap retained; no asset mesh/image overlay','base_training_iterations':7000,'asset_gradient_training':False,'removed_road_gaussians':int(cut.sum()),'original_background_gaussians_retained':base_count,'inserted_gaussians':len(g._xyz)-base_count,'road_apron':apron_info,'assets':conversion,'validation':{}}
    image_dir=out/'frames';image_dir.mkdir(exist_ok=True)
    video=imageio.get_writer(str(out/'pure_3dgs.mp4'),fps=10,codec='libx264',quality=9,macro_block_size=2)
    compare=imageio.get_writer(str(out/('before_after.mp4' if hazard_id else 'hybrid_vs_3dgs.mp4')),fps=10,codec='libx264',quality=8,macro_block_size=2)
    for camera in cameras:
        frame=camera.meta['frame'];rgb=renderer.render(camera,scene.gaussians)['rgb'].clamp(0,1)
        assert torch.isfinite(rgb).all()
        im=np.uint8(rgb.permute(1,2,0).cpu().numpy()*255+.5)
        video.append_data(im)
        comparison_folder=f'scene_{label}_{sid}' if hazard_id else f'inserted_{label}_{sid}'
        hybrid=np.array(Image.open(ROOT/'outputs'/comparison_folder/'frames'/f'{frame:03d}.jpg'))
        compare.append_data(np.concatenate([hybrid,im],1));Image.fromarray(im).save(image_dir/f'{frame:03d}.jpg',quality=96)
        if frame in [35,40,42,43]:Image.fromarray(np.concatenate([hybrid,im],1)).save(out/f'comparison_{frame:03d}.jpg',quality=96)
        if frame%20==0:print('render',sid,frame,flush=True)
    video.close();compare.close()
    count=export_splat(scene.gaussians,cameras[40],out/'scene_with_assets_frame040.ply');summary['gaussians_frame040']=count
    # This checkpoint keeps the actor trajectories and camera-time model intact.
    state=scene.gaussians.save_state_dict(is_final=True);state['iter']=7001
    (out/'trained_model').mkdir(exist_ok=True);torch.save(state,out/'trained_model/iteration_7000.pth');del state
    # Reload the saved parameters and require a matching render.
    reference=renderer.render(cameras[40],scene.gaussians)['rgb'].clone()
    scene.gaussians.load_state_dict(torch.load(out/'trained_model/iteration_7000.pth',weights_only=False))
    reloaded=renderer.render(cameras[40],scene.gaussians)['rgb']
    error=float((reference-reloaded).abs().max());assert error<1e-6
    summary['validation']['checkpoint_reload_max_rgb_error']=error
    original_pose=cameras[40].get_extrinsic().clone();camera=cameras[40]
    sweep=imageio.get_writer(str(out/'parallax_0.6m.mp4'),fps=12,codec='libx264',quality=9,macro_block_size=2)
    for offset in .6*np.sin(np.linspace(-np.pi/2,3*np.pi/2,48)):
        pose=original_pose.clone();pose[:3,3]+=original_pose[:3,0]*float(offset);camera.set_extrinsic(pose)
        rgb=renderer.render(camera,scene.gaussians)['rgb'].clamp(0,1)
        sweep.append_data(np.uint8(rgb.permute(1,2,0).cpu().numpy()*255+.5))
    sweep.close();camera.set_extrinsic(original_pose)
    summary['seconds']=time.time()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'render_config.yaml').write_text(f'parent_cfg: {ROOT}/work/street_crafter/configs/pandaset_{sid}_warmup.yaml\nmodel_path: {out}\nmode: trajectory\nloaded_iter: 7000\n')
    print('DONE',sid,count,summary['seconds'],flush=True)

if __name__=='__main__':main()
