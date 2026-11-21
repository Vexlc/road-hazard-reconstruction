"""LiDAR-supported road surfels that occlude shafts in low driving views.

The original road can drift below LiDAR. A small circular apron is insufficient:
a ray toward a one-metre-deep shaft may cross the road many metres in front of it.
This builds actual native Gaussian road geometry over those ray intersections.
"""
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.spatial.transform import Rotation
import torch
from run import sha


def input_provenance(data):
    paths=[data/'intrinsics/0.txt']
    paths += [data/'extrinsics'/f'{frame:03d}_0.txt' for frame in range(80)]
    paths += [data/'lidar_forward/background'/f'{frame:03d}.ply' for frame in range(0,80,3)]
    return {str(path.relative_to(data)):sha(path) for path in paths}


def plane_height(xy, anchors):
    estimates=[];weights=[]
    for a in anchors:
        n=a.basis[:,2]
        estimates.append(a.center[2]-((xy-a.center[:2])*n[:2]).sum(1)/n[2])
        weights.append(1/(np.sum((xy-a.center[:2])**2,axis=1)+.5))
    estimates,weights=np.array(estimates),np.array(weights)
    return np.sum(estimates*weights,axis=0)/weights.sum(0)


def support_hulls(anchors,cases,models,cameras):
    hulls=[]
    for slot,anchor in enumerate(anchors):
        selected=[models[c['assets'][slot]] for c in cases]
        if selected[0]['category'] not in ('missing','raised'):continue
        width=max(m['width_m'] for m in selected)
        length=max(m['length_m'] for m in selected)
        depth=max((m['depth_measured_m'] if 'depth_measured_m' in m else m['depth_design_m']) for m in selected)
        theta=np.linspace(0,2*np.pi,33)[:-1]
        shape=np.c_[np.cos(theta)*width*.70,np.sin(theta)*length*.70]
        samples=np.concatenate([np.c_[shape,np.full(len(shape),z)] for z in (-.07,-depth)])
        intersections=[]
        for cam in cameras:
            eye=(cam.camera_center.cpu().numpy()-anchor.center)@anchor.basis
            if eye[2]<.2:continue
            world=samples@anchor.basis.T+anchor.center
            w=cam.get_extrinsic(world2cam=True).cpu().numpy()
            k=cam.K.cpu().numpy()
            q=world@w[:3,:3].T+w[:3,3]
            uv=q[:,:2]/np.maximum(q[:,2:3],.01)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
            visible=(q[:,2]>.5)&(uv[:,0]>-cam.image_width*.2)&(uv[:,0]<cam.image_width*1.2)
            visible&=(uv[:,1]>0)&(uv[:,1]<cam.image_height*1.2)
            t=-eye[2]/(samples[:,2]-eye[2])
            cross=eye[:2]+t[:,None]*(samples[:,:2]-eye[:2])
            intersections.append(cross[visible])
        intersections=[x for x in intersections if len(x)]
        if not intersections:continue
        local=np.concatenate(intersections)
        # Include the whole apron perimeter, padding the projection hull beyond any shaft ray.
        local=np.r_[local,np.c_[2.8*np.cos(theta),2.8*np.sin(theta)]]
        xy=(np.c_[local,np.zeros(len(local))]@anchor.basis.T+anchor.center)[:,:2]
        hull=ConvexHull(xy)
        polygon=xy[hull.vertices]
        center=polygon.mean(0)
        delta=polygon-center
        polygon+=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),.01)*.5
        hulls.append(Delaunay(polygon))
    if not hulls:raise ValueError('no visible shaft support region')
    return hulls


def terrain(data,anchors,low,high,road_cache=None):
    cloud=[]
    for frame in range(0,80,3):
        path=data/'lidar_forward/background'/f'{frame:03d}.ply'
        if road_cache:
            from annotated_road import read
            xyz=read(road_cache,data.name,frame,sensor=1)
        else:
            a=PlyData.read(path,mmap='r')['vertex'].data
            xyz=np.stack([a[k] for k in ('x','y','z')],axis=1)
        take=np.all((xyz[:,:2]>low-1.4)&(xyz[:,:2]<high+1.4),axis=1)
        xyz=xyz[take]
        if len(xyz):
            prior=plane_height(xyz[:,:2],anchors)
            cloud.append(xyz[abs(xyz[:,2]-prior)<.25])
    points=np.concatenate(cloud)
    tree=cKDTree(points[:,:2])
    step=.35
    xs=np.arange(np.floor((low[0]-.7)/step)*step,high[0]+1.05,step)
    ys=np.arange(np.floor((low[1]-.7)/step)*step,high[1]+1.05,step)
    xx,yy=np.meshgrid(xs,ys)
    xy=np.c_[xx.ravel(),yy.ravel()]
    prior=plane_height(xy,anchors)
    heights=prior.copy();valid=np.zeros(len(xy),bool);rms=[]
    coefficients=np.zeros((len(xy),3))
    for i,origin in enumerate(xy):
        neighbors=tree.query_ball_point(origin,.85)
        if len(neighbors)<40:continue
        p=points[neighbors]
        a=np.c_[p[:,:2]-origin,np.ones(len(p))]
        keep=np.ones(len(p),bool)
        for _ in range(6):
            if keep.sum()<30:break
            coef=np.linalg.lstsq(a[keep],p[keep,2],rcond=None)[0]
            residual=p[:,2]-a@coef
            med=np.median(residual[keep])
            sigma=max(.004,np.median(abs(residual[keep]-med))*1.4826)
            keep=abs(residual-med)<min(.045,2.8*sigma)
        if keep.sum()<30:continue
        error=float(np.sqrt(np.mean(residual[keep]**2)))
        if error>.025 or np.linalg.norm(coef[:2])>.45 or abs(coef[2]-prior[i])>.20:continue
        heights[i]=coef[2];valid[i]=True;coefficients[i]=coef;rms.append(error)
    if valid.sum()<20:raise ValueError('insufficient LiDAR terrain support')
    # The initial front LiDAR has a near-ground blind zone. Continue nearby fitted
    # planes for at most three metres, only where source RGB still observes road.
    # Keep this inference separate from directly sampled terrain support.
    distances,nearest=cKDTree(xy[valid]).query(xy[~valid],k=4,workers=2)
    delta=xy[~valid,None,:]-xy[valid][nearest]
    coef=coefficients[valid][nearest]
    predictions=np.sum(delta*coef[:,:,:2],axis=2)+coef[:,:,2]
    weights=1/(distances+.15)**2
    heights[~valid]=np.sum(predictions*weights,axis=1)/weights.sum(1)
    disagreement=np.zeros(len(xy))
    disagreement[~valid]=np.ptp(predictions,axis=1)
    height=heights.reshape(xx.shape)
    dzdy,dzdx=np.gradient(height,step,step)
    return {'height':height,'dzdx':dzdx,'dzdy':dzdy,'xy':xy,'valid':valid,
            'plane_disagreement':disagreement.reshape(xx.shape),
            'origin':[xs[0],ys[0]],'step':step,'rms':rms,'lidar_points':len(points)}


def build(job,site,anchors,cases,models,cameras,teachers,bake_color,rb):
    hulls=support_hulls(anchors,cases,models,cameras)
    vertices=np.concatenate([h.points for h in hulls])
    low,high=vertices.min(0),vertices.max(0)
    step=float(job.get('road_guard_spacing_m',.018))
    xx,yy=np.meshgrid(np.arange(np.floor(low[0]/step)*step,high[0]+step,step),
                      np.arange(np.floor(low[1]/step)*step,high[1]+step,step))
    xy=np.c_[xx.ravel(),yy.ravel()]
    region=np.zeros(len(xy),bool)
    for hull in hulls:region|=hull.find_simplex(xy)>=0
    xy=xy[region]
    data=Path(rb.cfg.source_path)
    field=terrain(data,anchors,low,high,job.get('annotated_road_cache'))
    rc=((xy-np.array(field['origin']))/field['step']).T[::-1]
    height=map_coordinates(field['height'],rc,order=1,mode='nearest')
    dx=map_coordinates(field['dzdx'],rc,order=1,mode='nearest')
    dy=map_coordinates(field['dzdy'],rc,order=1,mode='nearest')
    support_distance=cKDTree(field['xy'][field['valid']]).query(xy,workers=2)[0]
    disagreement=map_coordinates(field['plane_disagreement'],rc,order=1,mode='nearest')
    # Match existing local apron planes as they meet the measured larger terrain.
    blend_weights=[];plane_values=[]
    for anchor in anchors:
        r=np.linalg.norm(xy-anchor.center[:2],axis=1)
        blend_weights.append(np.clip((3.5-r)/1.7,0,1)**2)
        n=anchor.basis[:,2]
        plane_values.append(anchor.center[2]-((xy-anchor.center[:2])*n[:2]).sum(1)/n[2])
    weights=np.array(blend_weights)
    blend=np.clip(weights.sum(0),0,1)
    nearby=np.sum(weights*np.array(plane_values),axis=0)/np.maximum(weights.sum(0),1e-12)
    height=height*(1-blend)+nearby*blend
    anchor_distance=np.array([np.linalg.norm(xy-a.center[:2],axis=1) for a in anchors])
    nearest=np.argsort(anchor_distance,axis=0)[:3]
    predictions=np.take_along_axis(np.array(plane_values),nearest,axis=0)
    agreement=np.ptp(predictions,axis=0)
    plane_weights=1/(np.take_along_axis(anchor_distance,nearest,axis=0)+.3)**2
    continued=np.sum(predictions*plane_weights,axis=0)/plane_weights.sum(0)
    inferred=(support_distance>=.60)&(support_distance<3.)&(agreement<.025)&(anchor_distance.min(0)<5.5)
    height[inferred]=continued[inferred]
    slopes=np.array([-a.basis[:2,2]/a.basis[2,2] for a in anchors])
    sx=np.sum(slopes[nearest,0]*plane_weights,axis=0)/plane_weights.sum(0)
    sy=np.sum(slopes[nearest,1]*plane_weights,axis=0)/plane_weights.sum(0)
    dx[inferred]=sx[inferred];dy[inferred]=sy[inferred]
    xyz=np.c_[xy,height]
    color,coverage,observed=bake_color(xyz,teachers,return_observed=True,allow_image_border=True)
    keep=observed&((support_distance<.60)|inferred)
    # Preserve every variant's full patch, including deep cavities and displaced lids.
    excluded=np.zeros(len(xy),bool)
    for slot,anchor in enumerate(anchors):
        local=(xyz-anchor.center)@anchor.basis
        for name in {case['assets'][slot] for case in cases}:
            _,surface=rb.read_asset(models[name])
            excluded|=rb.sample_field(surface,'alpha',local)>.005
    keep&=~excluded
    # Fade at the geometric patch boundary, not at camera/segmentation boundaries.
    # Fading the last observed image rows makes the road transparent exactly where
    # a grazing shaft projects onto the frame edge.
    grid=np.zeros(xx.size,bool);indices=np.flatnonzero(region);grid[indices[~excluded]]=True
    distance=distance_transform_edt(grid.reshape(xx.shape)).ravel()[indices]*step
    fade=np.minimum(distance/.15,1.)
    keep&=fade>.02
    direct=int(np.sum(keep&(support_distance<.60)))
    inferred=int(np.sum(keep&(support_distance>=.60)))
    max_distance=float(np.max(support_distance[keep]))
    xyz,color,dx,dy,fade=[x[keep] for x in (xyz,color,dx,dy,fade)]
    normal=np.c_[-dx,-dy,np.ones(len(dx))]
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    tangent=np.c_[np.ones(len(dx)),np.zeros(len(dx)),dx]
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    bitangent=np.cross(normal,tangent)
    rotation=Rotation.from_matrix(np.stack([tangent,bitangent,normal],axis=-1)).as_quat()[:,[3,0,1,2]]
    points=np.concatenate([xyz-normal*.002,xyz+np.c_[np.full(len(xyz),step*.5),np.full(len(xyz),step*.5),np.zeros(len(xyz))]-normal*.006])
    count=len(points)
    state={'xyz':rb.tensor(points),'feature_dc':rb.tensor(np.tile((color-.5)/rb.C0,(2,1)))[:,None],
           'feature_rest':torch.zeros(count,3,3,device='cuda'),
           'scaling':rb.tensor(np.tile(np.log([step*.95,step*.95,.0003]),(count,1))),
           'rotation':rb.tensor(np.tile(rotation,(2,1))),
           'opacity':torch.logit(rb.tensor(np.tile(fade*.995,2)[:,None]).clamp(.00001,.99999)),
           'semantic':torch.empty(count,0,device='cuda')}
    info={'points':count,'spacing_m':step,'layers':2,'observed_road_fraction_before_patch_exclusion':coverage,
          'geometry':'native road Gaussian surfels, LiDAR terrain; no image-space masking',
          'lidar_points':field['lidar_points'],'valid_terrain_knots':int(field['valid'].sum()),
          'terrain_fit_rms_median_m':float(np.median(field['rms'])),
          'terrain_fit_rms_max_m':float(np.max(field['rms'])),
          'direct_support_grid_points':direct,'inferred_blind_zone_grid_points':inferred,
          'maximum_support_distance_m':max_distance,
          'blind_zone_inference':'three nearest robust anchor planes agree within 2.5 cm; terrain support <3 m away; nearest anchor <5.5 m; source RGB road observed',
          'world_xy_bounds':[low.tolist(),high.tolist()],
          'source_geometry_sha256':input_provenance(data),
          'cavities_preserved':'all asset alpha fields excluded across every selected variant'}
    if job.get('annotated_road_cache'):
        info.update(terrain_labels='PandaSet per-point Road and road-marking ground truth; forward sensor; training frames 0,3,...78',
                    annotated_road_provenance_sha256=sha(Path(job['annotated_road_cache'])/'provenance.json'))
    print('ROAD GUARD',site,{k:v for k,v in info.items() if k!='source_geometry_sha256'},flush=True)
    return state,info
