"""Native five-category migration with covariance-aware cuts and checkpoint reload QA.

Run with StreetCrafter's --config, and HAZARD_FACTORY_JOB pointing to a JSON job.
The job names calibrated source scenes and LiDAR-fitted slots explicitly.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import yaml

from run import atomic_json, lock, sha

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'outputs/road_hazard_joint_scenes/reproduction'))
import render_batches as rb


def state_hash(state):
    digest = hashlib.sha256()
    def walk(value):
        if torch.is_tensor(value):
            a = value.detach().cpu().contiguous().numpy()
            digest.update(str((a.shape, str(a.dtype))).encode())
            digest.update(a.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(key.encode()); walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: walk(item)
        else:
            digest.update(repr(value).encode())
    walk({k:v for k,v in state.items() if k not in ('background', 'iter')})
    return digest.hexdigest()


def camera(anchor, model, angle=0):
    if model['category'] in ('missing', 'raised'):
        return rb.near_camera(anchor, model, 960, 540, angle)
    e = max(model['width_m'], model['length_m'])
    distance, rise = max(.075, e*1.48), max(.095, e*1.35)
    target = anchor.center + anchor.basis[:, 2]*(-min(.012, model.get('depth_p95_m', .04)*.25))
    eye = anchor.center + anchor.basis[:, 1]*(-distance*np.cos(angle))
    eye += anchor.basis[:, 0]*(distance*np.sin(angle)) + anchor.basis[:, 2]*rise
    return rb.camera_at(anchor.reference, eye, target, (960, 540), 960*.91)


def volume_cut(anchor, source, model):
    """Keep oriented splat support out of the replacement volume, including below road."""
    _, surface = rb.read_asset(model)
    world = source['xyz'].detach().cpu().numpy()
    local = (world-anchor.center)@anchor.basis
    scales = source['scaling'].detach().exp().cpu().numpy()
    quats = source['rotation'].detach().cpu().numpy()
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    rot = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    axes = np.einsum('ij,njk->nik', anchor.basis.T, rot)*scales[:, None, :]
    covariance = axes@axes.transpose(0, 2, 1)
    sigma = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    depth = model.get('depth_measured_m', model.get('depth_design_m', model.get('depth_p95_m', .1)))
    low = -max(depth+.15, .25)
    half = np.array([float(surface['width']), float(surface['length'])])*.5
    eligible = np.all(abs(local[:, :2]) < half+3*sigma[:, :2]+.025, axis=1)
    eligible &= (local[:, 2]+3*sigma[:, 2] > low)
    eligible &= (local[:, 2]-3*sigma[:, 2] < .16)
    ids = np.flatnonzero(eligible)
    points = local[ids]
    mask = surface['alpha'] > .03
    rr, cc = np.where(mask & ~binary_erosion(mask))
    edge = np.c_[(cc/(mask.shape[1]-1)-.5)*float(surface['width']),
                 (.5-rr/(mask.shape[0]-1))*float(surface['length'])]
    distance, nearest = cKDTree(edge).query(points[:, :2], workers=2)
    direction = (points[:, :2]-edge[nearest])/np.maximum(distance[:, None], 1e-12)
    radial_sigma = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', direction,
                                               covariance[ids, :2, :2], direction), 0))
    inside = rb.sample_field(surface, 'alpha', points) > .03
    remove = inside | (distance < 3*radial_sigma+.025)
    return ids[remove], {'volume_bottom_local_m': low, 'support_sigma': 3.,
                        'candidate_count': len(ids), 'removed_count': int(remove.sum()),
                        'method': 'oriented covariance projected along footprint boundary; depth-aware volume'}


def cavity_fraction(result, cam, anchor, model):
    """Ray-depth check in the center of a missing cover's actual opening."""
    if model['category'] != 'missing':
        return None
    values = np.linspace(-.32, .32, 23)
    x, y = np.meshgrid(values*model['width_m'], values*model['length_m'])
    exponent = model['exponent']
    mask = abs(x/(.5*model['width_m']))**exponent + abs(y/(.5*model['length_m']))**exponent < .62
    local = np.c_[x[mask], y[mask], np.zeros(mask.sum())]
    points = local@anchor.basis.T+anchor.center
    w = cam.get_extrinsic(world2cam=True).cpu().numpy()
    k = cam.K.cpu().numpy()
    q = points@w[:3, :3].T+w[:3, 3]
    uv = q[:,:2]/q[:,2:3]*[k[0,0], k[1,1]]+[k[0,2], k[1,2]]
    px = np.rint(uv).astype(int)
    good = ((px[:,0]>=0)&(px[:,0]<cam.image_width)&(px[:,1]>=0)&(px[:,1]<cam.image_height))
    px = px[good]
    depth = result['depth'][0].cpu().numpy()[px[:,1], px[:,0]]
    rays = np.c_[(px[:,0]-k[0,2])/k[0,0], (px[:,1]-k[1,2])/k[1,1], np.ones(len(px))]
    pose = np.linalg.inv(w)
    hits = (rays*depth[:,None])@pose[:3,:3].T+pose[:3,3]
    height = (hits-anchor.center)@anchor.basis[:,2]
    fraction = float(np.mean(height < -.07))
    if fraction < .55:
        raise ValueError(f"{model['id']}: cavity obscured; below-road ray fraction={fraction}")
    return {'rays': len(px), 'below_road_7cm_fraction': fraction,
            'median_hit_local_height_m': float(np.median(height))}


def writer(path, fps):
    return imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8,
                             macro_block_size=2, output_params=['-threads','3','-movflags','+faststart'])


@torch.no_grad()
def road_bake_color(points, teachers, return_observed=False, allow_image_border=False):
    """Project observed RGB only onto pixels classified as road and free of actors.

    The fitted LiDAR plane supplies target geometry. Monocular rendered depth is
    used as a soft score, so its underground drift cannot reject a visible road.
    """
    xyz=rb.tensor(points)
    best=torch.full((len(xyz),),-1.,device='cuda')
    colors=torch.zeros(len(xyz),3,device='cuda')
    for w,k,im,depth,road in teachers:
        q=xyz@w[:3,:3].T+w[:3,3];z=q[:,2]
        u=q[:,0]*k[0,0]/z+k[0,2];v=q[:,1]*k[1,1]/z+k[1,2]
        h,width=depth.shape[-2:]
        grid=torch.stack([2*(u+.5)/width-1,2*(v+.5)/h-1],1)[None,None]
        observed=F.grid_sample(depth,grid,align_corners=False,padding_mode='border')[0,0,0]
        sampled=F.grid_sample(im,grid,align_corners=False,padding_mode='border')[0,:,0].T
        support=F.grid_sample(road,grid,mode='nearest',align_corners=False,padding_mode='border')[0,0,0]
        if allow_image_border:
            valid=(z>2)&(u>-.5)&(u<width-.5)&(v>-.5)&(v<h-.5)&(support>.5)
        else:
            valid=(z>2)&(u>5)&(u<width-5)&(v>6)&(v<h-6)&(support>.5)
        score=torch.where(valid,1/z-.005*(observed-z).abs().clamp(max=3),torch.full_like(z,-1))
        take=score>best;best[take]=score[take];colors[take]=sampled[take]
    missing=(best<0).cpu().numpy();rgb=colors.cpu().numpy()
    if missing.any():
        if (~missing).sum()<50:raise ValueError('insufficient observed road pixels')
        nearest=cKDTree(points[~missing,:2]).query(points[missing,:2],workers=2)[1]
        rgb[missing]=rgb[~missing][nearest]
    coverage=float(1-missing.mean())
    return (rgb,coverage,~missing) if return_observed else (rgb,coverage)


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    job = json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf-8'))
    job.setdefault('road_occlusion_guard',True)
    out, work = Path(job['output']), Path(job['work'])
    out.mkdir(parents=True, exist_ok=True)
    site = Path(rb.cfg.source_path).name
    rb.bake_color=road_bake_color
    plan = json.loads(Path(job['plan']).read_text(encoding='utf-8'))
    models = {}
    factory = []
    for library in job['libraries']:
        library = Path(library)
        for m in json.loads((library/'asset_manifest.json').read_text(encoding='utf-8')):
            m = dict(m)
            for key in ('ply', 'surface', 'glb'):
                m[key] = str(library/m[key])
            models[m['id']] = m
            if m.get('recipe_version') == 'physical-v1': factory.append(m)
    categories = plan['category_order']
    import batch_plan
    groups=batch_plan.build(job,plan,models,factory)
    if not groups:raise ValueError('no complete synthetic triplet')
    cases=[{'id':f'{site}_{group["number"]:02d}','site':site,'assets':group['assets']} for group in groups]
    if job.get('case_groups'):
        cases=[c for c in cases if int(c['id'].split('_')[1]) in job['case_groups']]
    if job.get('road_occlusion_guard'):
        import road_guard
    if job.get('material_contact'):
        import material_contact
    base_signature = {'job':job, 'code':sha(Path(__file__)),
                             'batch_plan_code':sha(Path(batch_plan.__file__)),
                             'rb_code':sha(Path(rb.__file__)), 'plan':sha(Path(job['plan'])),
                             'source_checkpoint':sha(Path(rb.cfg.model_path)/'trained_model/iteration_7000.pth'),
                             'road_mask_provenance':sha(work/'road_masks/provenance.json')}
    if job.get('road_occlusion_guard'):
        base_signature['guard_code']=sha(Path(road_guard.__file__))
        base_signature['road_geometry_inputs']=road_guard.input_provenance(Path(rb.cfg.source_path))
        base_signature['scene_config']=sha(Path(job['configs'][site]))
    if job.get('annotated_road_cache'):
        import annotated_road
        base_signature['annotated_road_source']=sha(Path(job['annotated_road_cache'])/'provenance.json')
        base_signature['annotated_road_code']=sha(Path(annotated_road.__file__))
    if job.get('material_contact'):
        base_signature['material_contact_code']=sha(Path(material_contact.__file__))
    signatures = {c['id']:hashlib.sha256(json.dumps(dict(base_signature,case=c,
                             assets={n:[sha(Path(models[n]['ply'])),sha(Path(models[n]['surface'])),
                                        sha(Path(job['mesh_cache'])/(n+'.npz'))
                                        if job.get('material_contact') and models[n]['category'] in ('missing','raised') else None]
                                     for n in c['assets']}),sort_keys=True).encode()).hexdigest() for c in cases}
    def current(dest):
        try:
            done = json.loads((dest/'complete.json').read_text(encoding='utf-8'))
            return done['signature']==signatures[dest.name] and all(sha(dest/name)==h for name,h in done['files'].items())
        except (OSError, ValueError, KeyError): return False
    with lock(work/'locks'/f'migrate_{site}.lock'):
        if all(current(out/c['id']) for c in cases):
            print('VERIFIED MIGRATION CACHE',site,flush=True); return
        scene = rb.create_scene(); g = scene.gaussians; renderer = rb.StreetGaussianRenderer()
        cameras = sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.meta['frame'])
        rb.WIDTH, rb.HEIGHT = 960,540
        rb.resize_cameras(cameras)
        source = {k:getattr(g.background,v) for k,v in rb.ATTRS.items()}
        invariant = state_hash(g.save_state_dict(is_final=True))
        teachers = []
        for frame in range(0,80,3):
            cam = cameras[frame]; result = renderer.render(cam,g)
            rgb = F.interpolate(cam.original_image[None].to('cuda'),size=(540,960),mode='bilinear',align_corners=False)
            path=work/'road_masks'/site/f'{frame:03d}_0.png'
            road=rb.tensor(np.array(Image.open(path),dtype='f4')/255)[None,None]
            teachers.append((cam.get_extrinsic(world2cam=True).clone(),cam.K.clone(),rgb,result['depth'][None].clone(),road))
        records = plan['slots_by_site'][site]
        centers = np.array([x['center_world'][:2] for x in records])
        world = source['xyz'].cpu().numpy()
        spread = source['scaling'].exp().max(1).values.cpu().numpy()
        anchors = [rb.Anchor(record,centers,cameras,world,spread,teachers) for record in records]
        for anchor,record in zip(anchors,records):
            anchor.reference = cameras[record['source_reference_frame']]
            if np.max(abs(anchor.basis.T@anchor.basis-np.eye(3))) > 1e-5 or np.linalg.det(anchor.basis)<.999:
                raise ValueError('invalid placement basis')
            if record['rms_m']>.03:
                raise ValueError('unsupported host road plane fit')
        # A seven-metre apron may extend behind the observing car. Gate the actual
        # replacement footprint independently; keep the wide extrapolation metric visible.
        support_coverage={}
        for case in cases:
            for slot,name in enumerate(case['assets']):
                model,anchor=models[name],anchors[slot]
                _,surface=rb.read_asset(model)
                a=surface['alpha'][::4,::4]
                x,y=np.meshgrid(np.linspace(-.5,.5,a.shape[1])*float(surface['width']),
                                np.linspace(.5,-.5,a.shape[0])*float(surface['length']))
                points=np.c_[x[a>.10],y[a>.10],np.zeros((a>.10).sum())]
                _,coverage=rb.bake_color(points@anchor.basis.T+anchor.center,teachers)
                support_coverage[name]=coverage
                if coverage<.80:
                    raise ValueError(f'{site}/{name}: actual footprint color coverage {coverage:.3f} < .80')
        del result
        for case in cases:
            dest = out/case['id']
            if current(dest):continue
            guard,guard_info=None,{'points':0}
            if job.get('road_occlusion_guard'):
                # Each checkpoint depends on its own assets, so future appended
                # variants cannot invalidate an unrelated old road surface.
                guard,guard_info=road_guard.build(job,site,anchors,[case],models,cameras,teachers,road_bake_color,rb)
            stage = out/'.staging'/uuid.uuid4().hex
            stage.mkdir(parents=True)
            rb.set_background(g.background,source)
            keep = torch.ones(len(world),dtype=torch.bool,device='cuda')
            additions,details,before = [],[],[]
            for slot,name in enumerate(case['assets']):
                model,anchor = models[name],anchors[slot]
                asset,apron,_,exposure = anchor.fit_asset(model)
                contact_info=None
                if job.get('material_contact') and model['category'] in ('missing','raised'):
                    asset,contact_info=material_contact.apply(anchor,model,asset,exposure,job,rb)
                cut,cut_info = volume_cut(anchor,source,model)
                keep[torch.as_tensor(cut,device='cuda')] = False
                before.append(rb.numpy_image(renderer.render(camera(anchor,model),g)))
                additions.extend([asset,apron])
                details.append({'asset_id':name,'category':model['category'],'size_multiplier':1.,
                                'asset_points':len(asset['xyz']),'apron_points':len(apron['xyz']),
                                'exposure_gain':exposure,'color_coverage':anchor.coverage,
                                'actual_footprint_color_coverage':support_coverage[name],
                                'material_contact':contact_info,
                                'cut':cut_info, **anchor.record})
            all_additions=additions+([guard] if guard is not None else [])
            merged = {k:torch.nn.Parameter(torch.cat([source[k][keep]]+[x[k] for x in all_additions]),
                                           requires_grad=False) for k in rb.ATTRS}
            rb.set_background(g.background,merged)
            offset = int(keep.sum())
            for slot,(name,d) in enumerate(zip(case['assets'],details)):
                model,anchor = models[name],anchors[slot]
                vertices,_ = rb.read_asset(model)
                n = len(vertices)
                actual = merged['xyz'][offset:offset+n].cpu().numpy()
                error = float(np.max(abs(actual-(vertices[:,:3]@anchor.basis.T+anchor.center))))
                scale_error = float(np.max(abs(merged['scaling'][offset:offset+n].cpu().numpy()-vertices[:,16:19])))
                if error>2e-5 or scale_error>1e-6:raise ValueError('physical geometry changed')
                d.update(merged_asset_offset=offset,world_error_m=error,log_scale_error=scale_error)
                offset += d['asset_points']+d['apron_points']
                cam = camera(anchor,model)
                result = renderer.render(cam,g)
                picture = rb.numpy_image(result)
                d['cavity_depth_check'] = cavity_fraction(result,cam,anchor,model)
                Image.fromarray(before[slot]).save(stage/f'{d["category"]}_before.png')
                Image.fromarray(picture).save(stage/f'{d["category"]}_after.png')
                with writer(stage/f'{d["category"]}_local_before_after.mp4',12) as video:
                    for angle in np.linspace(-.40,.40,24):
                        view = camera(anchor,model,float(angle))
                        after = rb.numpy_image(renderer.render(view,g))
                        rb.set_background(g.background,source)
                        first = rb.numpy_image(renderer.render(view,g))
                        rb.set_background(g.background,merged)
                        video.append_data(np.concatenate([first,after],axis=1))
            start = plan['drive_start_by_site'][site]
            with writer(stage/'global_before_after.mp4',10) as video:
                for frame in range(start,start+40):
                    view = cameras[frame]
                    after = rb.numpy_image(renderer.render(view,g))
                    rb.set_background(g.background,source)
                    first = rb.numpy_image(renderer.render(view,g))
                    rb.set_background(g.background,merged)
                    video.append_data(np.concatenate([first,after],axis=1))
            state = g.save_state_dict(is_final=True)
            if state_hash(state)!=invariant:raise ValueError('actor or sky changed')
            state['iter'] = 7001
            model_dir = stage/'trained_model'; model_dir.mkdir()
            torch.save(state,model_dir/'iteration_7000.pth')
            # Reload serialized state, rerender every category, compare lossless reference pixels.
            reloaded = torch.load(model_dir/'iteration_7000.pth',map_location='cuda',weights_only=False)
            g.load_state_dict(reloaded)
            for slot,(name,d) in enumerate(zip(case['assets'],details)):
                picture = rb.numpy_image(renderer.render(camera(anchors[slot],models[name]),g))
                saved = np.array(Image.open(stage/f'{d["category"]}_after.png'))
                d['reload_max_pixel_error'] = int(np.max(abs(picture.astype(int)-saved.astype(int))))
                if d['reload_max_pixel_error']>1:raise ValueError('serialized scene did not reproduce')
            if state_hash(reloaded)!=invariant:raise ValueError('serialized actors/sky differ')
            atomic_json(stage/'scene.json', dict(case, placements=details, actors_sky_sha256=invariant,
                        actors_sky_unchanged=True, checkpoint_reloaded=True,
                        host_road_guard=guard_info,retained_background_points=int(keep.sum()),
                        source_checkpoint=str(Path(rb.cfg.model_path)/'trained_model/iteration_7000.pth'),
                        limitations=['Forward-camera background remains imperfect at nearby virtual viewpoints.',
                                     'Photographed assets retain captured illumination.']))
            config = {'parent_cfg':job['configs'][site], 'model_path':str(dest),
                      'mode':'trajectory', 'loaded_iter':7000, 'resolution':960}
            (stage/'render_config.yaml').write_text(yaml.safe_dump(config,allow_unicode=False),encoding='utf-8')
            files = {str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
            atomic_json(stage/'complete.json', {'signature':signatures[case['id']],'files':files})
            if dest.exists():
                quarantine=work/'retired_migrations';quarantine.mkdir(exist_ok=True)
                os.replace(dest,quarantine/(dest.name+'_'+uuid.uuid4().hex))
            os.replace(stage,dest)
            rb.set_background(g.background,source)
            del state,reloaded,merged,additions
            torch.cuda.empty_cache()
            print('MIGRATED',case['id'],'5 categories; native reload passed',flush=True)


if __name__ == '__main__':main()
