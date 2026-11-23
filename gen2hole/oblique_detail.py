"""Fit a photographed broken lip and reconstruct only the unresolved material bandwidth from scans."""
from hashlib import sha256, md5
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def blur(image, sigma):
    return cv2.GaussianBlur(np.float32(image), (0,0), max(.25,sigma))


def normalized_blur(a, weight, sigma):
    den = blur(weight,sigma)
    num = blur(a * (weight[...,None] if a.ndim==3 else weight),sigma)
    return num / np.maximum(den[...,None] if a.ndim==3 else den, 1e-5)


def rim_on_photo(outline, rgb, allowed):
    """Look for a narrow transition between soil and asphalt within three camera pixels of the trace."""
    path=np.float32(outline)
    samples=[]
    for a,b in zip(path,np.roll(path,-1,axis=0)):
        samples.extend(np.linspace(a,b,max(2,round(np.linalg.norm(b-a))),endpoint=False))
    p=np.float32(samples)
    tang=np.roll(p,-2,axis=0)-np.roll(p,2,axis=0)
    inner=np.column_stack([-tang[:,1],tang[:,0]]) / np.maximum(np.linalg.norm(tang,axis=1)[:,None],1e-4)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype('f4')
    shifts=np.linspace(-3,3,13,dtype='f4')
    def profile(dist,im):
        uv=p[:,None,:]+inner[:,None,:]*dist[None,:,None]
        return cv2.remap(im,uv.astype('f4'),None,cv2.INTER_LINEAR)
    a=profile(shifts+2.,lab); b=profile(shifts-2.,lab)
    score=(a[:,:,2]-b[:,:,2])*1.6 + np.minimum(abs(a[:,:,0]-b[:,:,0]),25)*.30
    score-=abs(shifts)*1.2
    index=np.argmax(score,axis=1)
    delta=shifts[index]*np.clip((score.max(1)-2)/8,0,1)
    delta*=profile(np.float32([0]),allowed.astype('f4'))[:,0]
    delta=np.convolve(np.pad(delta,3,mode='wrap'),np.array([1,2,3,4,3,2,1])/16,'valid')
    return np.float32(p+inner*delta[:,None])


def fit_inset(inset, rays, normal, offset, road):
    good=np.isfinite(inset) & (inset>0) & road
    inv=1/inset[good].astype('f8')
    expected=-(rays[good]@normal)/offset
    keep=np.ones(len(inv),bool)
    for _ in range(6):
        scale,shift=np.polyfit(inv[keep],expected[keep],1)
        error=abs(scale*inv+shift-expected)
        keep=error<np.percentile(error,81)
    height=(rays@normal)/(scale/inset+shift)+offset
    return height,dict(inverse_scale=float(scale),inverse_shift=float(shift),road_pixels=int(good.sum()))


def smooth_field(xy,z,footprint,span):
    """Fit scattered visible depth; bridge unsupported near-facing strips to the lip continuously."""
    size=len(footprint)
    coords=(xy*[1.,-1.]+span/2)*((size-1)/span)
    ij=np.floor(coords).astype('i4')
    frac=coords-ij
    sums=np.zeros(size**2,'f8'); counts=np.zeros_like(sums)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        cols,rows=ij[:,0]+dx,ij[:,1]+dy
        use=(cols>=0)&(cols<size)&(rows>=0)&(rows<size)
        at=rows[use]*size+cols[use]
        w=abs((1-dx)-frac[use,0])*abs((1-dy)-frac[use,1])
        counts+=np.bincount(at,weights=w,minlength=size**2)
        sums+=np.bincount(at,weights=w*z[use],minlength=size**2)
    counts=counts.reshape((size,size)); sums=sums.reshape((size,size))
    fit=np.float32(blur(sums,.80)/np.maximum(blur(counts,.80),1e-5))
    w=np.clip(blur(counts,.8)/2,0,1).astype('f4')*footprint
    trusted=w>.3
    nearest=distance_transform_edt(~trusted,return_distances=False,return_indices=True)
    h=fit[tuple(nearest)]*footprint
    # Warm-start long occlusion strips at coarse scales, then relax on the full grid.
    for sigma,steps in [(5.,15),(2.,20),(1.,40)]:
        for _ in range(steps):
            h=(w*fit+(1-w)*blur(h,sigma))*footprint
    return np.minimum(h,0),trusted


def wall_profile(gx,gy,height,rim_xy,profile,K,n,axes,origin,offset):
    """The observed upper and lower pixels constrain the drop of the exposed, steep top band."""
    picks=np.float32(profile['wall_transects'])
    rays=np.c_[picks[:,0],np.ones(len(picks))]@np.linalg.inv(K).T
    tops=rays*(-offset/(rays@n))[:,None]
    tangent=np.gradient(tops,axis=0)
    inward=np.cross(n,tangent)
    inward*=np.sign(np.sum((origin-tops)*inward,axis=1))[:,None]
    inward/=np.linalg.norm(inward,axis=1,keepdims=True)
    # Slight inward batter of the visible upper break; depth remains a single-view estimate.
    wall_dir=-n+.27*inward
    proj=K[:2,None,:]-picks[:,1,:,None].transpose(1,0,2)*K[2]
    a=np.einsum('jik,ik->ij',proj,tops); b=np.einsum('jik,ik->ij',proj,wall_dir)
    dz=np.clip(-(a*b).sum(1)/(b*b).sum(1),.025,.28)
    exact_xy=((tops-origin)@axes)[:,:2]
    exact_xy=rim_xy[cKDTree(rim_xy).query(exact_xy)[1]]
    arcs=[]; drops=[]
    for i in range(len(dz)-1):
        steps=max(2,int(np.linalg.norm(exact_xy[i+1]-exact_xy[i])/.012))
        arcs.extend(np.linspace(exact_xy[i],exact_xy[i+1],steps))
        drops.extend(np.linspace(dz[i],dz[i+1],steps))
    xy=np.stack([gx,gy],-1)
    r,index=cKDTree(arcs).query(xy)
    dist=cKDTree(rim_xy).query(xy)[0]
    drop=np.asarray(drops)[index]
    scarp=-drop*np.minimum(dist/np.maximum(.025,drop*.27),1)
    # Use the nearest actual rim for the profile, and leave the unmarked rim unconstrained.
    blend=np.exp(-np.maximum(r-dist,0)**2/(2*.055**2))*np.exp(-(np.maximum(dist-drop*.27,0)/.105)**2)
    return height*(1-blend)+scarp*blend, dz


def sampled_material(color_path,disp_path,physical_width,gx,gy,rotation):
    color=np.array(Image.open(color_path).convert('RGB'))
    displacement=cv2.imread(str(disp_path),cv2.IMREAD_UNCHANGED)
    if displacement.ndim==3:
        displacement=displacement[:,:,0]
    displacement=displacement.astype('f4')/np.iinfo(displacement.dtype).max
    # The source scans are seamless. The two fields share coordinates, physical scale and filtering.
    c,s=np.cos(rotation),np.sin(rotation)
    u=(c*gx-s*gy)/physical_width+.35
    v=-(s*gx+c*gy)/physical_width+.42
    maps=np.float32(np.stack([u*color.shape[1],v*color.shape[0]],-1))
    sampling_pitch=abs(float(gx[0,1]-gx[0,0]))
    filt=max(.35,sampling_pitch*color.shape[1]/physical_width*.45)
    def read(a):
        return cv2.remap(blur(a,filt),maps,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    return read(color),read(displacement)


def refine(bundle,profile_path,inputs,pavement):
    from oblique_pits import camera_rays, polygon_mask, project, put
    profile=json.loads(profile_path.read_text()); key=profile['id']
    if len(profile['insets'])!=2 or len(profile['wall_transects'])<2:
        raise ValueError('Expected two inset cameras and at least two ordered wall transects')
    cal,ref=bundle/'calibration',bundle/'references'
    meta=json.loads((cal/(key+'.json')).read_text())
    capture=json.loads((cal/(key+'_capture.json')).read_text())
    spec=json.loads((cal/(key+'-outline.json')).read_text())
    src=bundle/meta['source_image']
    rgb=np.array(ImageOps.exif_transpose(Image.open(src)).convert('RGB')); shape=rgb.shape[:2]
    n=np.float32(capture['road_planes']['direct']['normal'])
    axes=np.float32(capture['local_axes_in_camera'])
    o=np.float32(capture['local_origin_in_camera'])
    K=np.float32(capture['road_planes']['direct']['K'])
    offset=capture['road_planes']['direct']['offset_m']
    rays=camera_rays(shape,K)
    blocked=polygon_mask(shape,spec.get('exclude_polygons',[])).astype('u1')
    for x,y,xx,yy,width in spec.get('overlay_segments',[]):
        cv2.line(blocked,(x,y),(xx,yy),1,width+2)
    clear=blocked==0
    traced=rim_on_photo(spec['rim'],rgb,clear)
    mask=np.zeros(shape,'u1')
    cv2.fillPoly(mask,[np.rint(traced*16).astype('i4')],1,shift=4)
    mask=mask.astype(bool)
    road=polygon_mask(shape,spec['road_polygons']) & clear & ~mask
    prior=[]
    with np.load(cal/(key+'-depth.npz')) as cache:
        for side in ['direct','flip']:
            plane=capture['road_planes'][side]
            depth=cache['depth_'+side]
            if 'design_scale' in capture:
                depth=depth*capture['design_scale']['depth_multiplier']
            prior.append(depth*(camera_rays(shape,cache['K_'+side])@np.float32(plane['normal']))+plane['offset_m'])
    base=np.float32((prior[0]+prior[1])*.5)
    support=mask & clear & np.isfinite(base)
    estimates=[]
    fits=[]
    inset_path=inputs/(key+'-insets.npz')
    with np.load(inset_path) as cache:
        if str(cache['source_sha256'])!=meta['source_sha256'] or str(cache['weights_sha256'])!=capture['weights_sha256'] or json.loads(str(cache['insets']))!=profile['insets']:
            raise ValueError('Inset cache and photo/profile/checkpoint do not match')
        for i in range(len(profile['insets'])):
            h,fit=fit_inset(cache['z'+str(i)],rays,n,offset,road)
            fits.append(fit);valid=support & np.isfinite(h)
            delta=np.where(valid,h-base,0)
            estimates.append(np.where(valid,delta-normalized_blur(delta,valid.astype('f4'),14),0))
    correction=np.clip((estimates[0]+estimates[1])*.5,-.027,.027)
    h=np.minimum(base+correction,0)
    if 'design_bed_depth_p95_m' in profile:
        target=float(profile['design_bed_depth_p95_m'])
        inferred=float(np.percentile(-h[support],95))
        if spec.get('reference_kind')!='generated' or not np.isfinite([target,inferred]).all() or min(target,inferred)<=0:
            raise ValueError('A designed bed depth requires a generated reference and a positive depth prior')
        # Solve along the source rays again so the shallower bed retains its image registration.
        h*=target/inferred
        capture.setdefault('design_scale',{}).update(bed_prior_p95_m=inferred,design_bed_depth_p95_m=target)
        meta['design_bed_depth_p95_m']=target
    interior=cv2.distanceTransform(mask.astype('u1'),cv2.DIST_L2,5)
    use=support & (interior>1.5)
    samples=rays[use]*((h[use]-offset)/(rays[use]@n))[:,None]
    local=(samples-o)@axes
    rim_rays=np.c_[traced,np.ones(len(traced))]@np.linalg.inv(K).T
    rim_xy=((rim_rays*(-offset/(rim_rays@n))[:,None]-o)@axes)[:,:2]
    with np.load(cal/(key+'.npz')) as data:
        span=float(data['width'])
    def grid(size):
        line=np.linspace(-span/2,span/2,size,dtype='f4')
        return np.meshgrid(line,line[::-1])
    size=545; texsize=2049
    x,y=grid(size);pitch=span/(size-1); tex_pitch=span/(texsize-1)
    ixy=((rim_xy*[1,-1]+span/2)/pitch)
    mask_grid=np.zeros((size,size),'u1');cv2.fillPoly(mask_grid,[np.rint(ixy*16).astype('i4')],1,shift=4)
    inside=mask_grid>0
    z,measured=smooth_field(local[:,:2],local[:,2],inside,span)
    z,drops=wall_profile(x,y,z,rim_xy,profile,K,n,axes,o,offset)
    dist=distance_transform_edt(inside)*pitch
    z=np.float32(z*np.clip((dist-.35*pitch)/pitch,0,1)*inside)
    up=lambda a:cv2.resize(np.float32(a),(texsize,texsize),interpolation=cv2.INTER_LINEAR)
    gx,gy=grid(texsize)
    tex_z=up(z)
    tex_mask=up(mask_grid)
    world=o+gx[...,None]*axes[:,0]+gy[...,None]*axes[:,1]+tex_z[...,None]*n
    uv=project(world,K)
    # Detect backfacing or severe source foreshortening using the local projection Jacobian.
    dux=cv2.Sobel(uv[:,:,0],cv2.CV_32F,1,0,ksize=3,scale=.125)
    duy=cv2.Sobel(uv[:,:,0],cv2.CV_32F,0,1,ksize=3,scale=.125)
    dvx=cv2.Sobel(uv[:,:,1],cv2.CV_32F,1,0,ksize=3,scale=.125)
    dvy=cv2.Sobel(uv[:,:,1],cv2.CV_32F,0,1,ksize=3,scale=.125)
    det=dux*dvy-duy*dvx
    discriminant=np.maximum((dux**2+duy**2+dvx**2+dvy**2)**2-4*det**2,0)
    stretch=(dux**2+duy**2+dvx**2+dvy**2+np.sqrt(discriminant))*.5/np.maximum(abs(det),1e-5)
    valid_color=(mask & clear).astype('u1')
    photo=cv2.remap(rgb,uv,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype('f4')
    good=(cv2.remap(valid_color,uv,None,cv2.INTER_NEAREST)>0)&(det>0)&(tex_mask>.5)
    nearest=distance_transform_edt(~good,return_distances=False,return_indices=True)
    projected=photo[tuple(nearest)]
    broad=blur(projected, .055/tex_pitch)
    mid=blur(projected,.013/tex_pitch)
    confidence=np.clip((12-stretch)/9,0,1)*good
    color=broad+(mid-broad)*(.36+.64*confidence[...,None])+(projected-mid)*(.16+.77*confidence[...,None])
    # Local road colour supplies the mean tint; use a measured asphalt surface in the apron.
    roadcol=np.median(rgb[road],axis=0)
    roadmap,roadheight=sampled_material(pavement/'Asphalt031_4K-PNG_Color.png',
        pavement/'Asphalt031_4K-PNG_Displacement.png',2.,gx,gy,.16)
    road_detail=roadmap-blur(roadmap,.14/tex_pitch)
    road_color=np.clip(roadcol+.72*road_detail,0,255)
    materials={}
    for k,rec in profile['materials'].items():
        paths=[]
        for field in ['albedo','displacement']:
            info=rec[field]; src_map=inputs/Path(info['url']).name
            if md5(src_map.read_bytes()).hexdigest()!=info['md5']:
                raise ValueError('Incorrect material input '+str(src_map))
            paths.append(src_map)
        materials[k]=sampled_material(*paths,rec['width_m'],gx,gy,.22 if k=='gravelly_sand' else -.54)
    def detail(k, amplitude):
        c,d=materials[k]
        # Use only the material's small-scale luminance. Broad source colours and distinct dark
        # clasts come from the road photograph, independent of scan lighting and hue.
        lum=c@np.float32([.213,.715,.072])
        residual=(lum-blur(lum,.075/tex_pitch))
        dz=(d-blur(d,.10/tex_pitch))
        return residual/np.maximum(8.,np.percentile(abs(residual),85)),np.clip(dz/np.maximum(.012,np.percentile(abs(dz),96)),-2,2)*amplitude
    sand,ds=detail('gravelly_sand',.004)
    stones,dc=detail('stony_dirt_path',.012)
    luminance=color@np.float32([.213,.715,.072])
    breakup=blur(abs(projected[:,:,1]-mid[:,:,1]),.038/tex_pitch)
    coarse=np.clip(.45+.08*breakup+(blur(luminance,.17/tex_pitch)-luminance)/85,.28,.90)
    grain=sand*(1-coarse)*14+stones*coarse*20
    bedcolor=np.clip(color+grain[...,None],0,255)
    mix=tex_mask[...,None]
    final=np.uint8(np.clip(road_color*(1-mix)+bedcolor*mix,0,255))
    dz=(coarse*dc+(1-coarse)*ds)*np.clip(up(dist)/.055,0,1)
    low_dz=cv2.resize(blur(dz,.65*pitch/tex_pitch),(size,size),interpolation=cv2.INTER_LINEAR)
    z=np.minimum(z+low_dz,0)*inside
    # Tangent-space normals describe the relief finer than the exported grid only.
    rd=(roadheight-blur(roadheight,.038/tex_pitch))
    rd=np.clip(rd/np.maximum(.008,np.percentile(abs(rd),97)),-2,2)*.0018
    residual=(dz-up(low_dz))*tex_mask+rd*(1-tex_mask)
    dr,dc=np.gradient(blur(residual,.6),tex_pitch)
    normal=np.stack([-dc,-dr,np.ones_like(dc)],-1)
    normal/=np.linalg.norm(normal,axis=2,keepdims=True)
    normal=np.uint8(np.clip(127.5+127.5*normal,0,255))
    Image.fromarray(final).save(ref/(key+'_ortho.png'))
    Image.fromarray(normal).save(ref/(key+'_normal.png'))
    Image.fromarray(np.uint8(road_color[:650,:650])).save(ref/(key+'_pavement.jpg'),quality=96)
    alpha=np.clip(1-distance_transform_edt(~inside)*pitch/.55,0,1)
    np.savez_compressed(cal/(key+'.npz'),height=z.astype('f4'),alpha=alpha.astype('f4'),defect_mask=inside,
                        water_mask=np.zeros_like(inside),prediction_support=measured&inside,width=span,length=span)
    shutil.copyfile(profile_path,cal/(key+'-detail.json'));shutil.copyfile(inset_path,cal/(key+'-insets.npz'))
    capture['detail_refinement']=dict(inset_registration=fits,marked_wall_transects=len(drops),
        inferred_upper_break_drop_range_m=[float(drops.min()),float(drops.max())],
        assumed_upper_wall_run_per_drop=.27,grid_pitch_m=pitch,texture_pitch_m=tex_pitch,
        projected_texture_anisotropy_median=float(np.median(stretch[good])),materials=profile['materials'],
        unapportioned_grains='Unresolved grain appearance and millimetre relief from scans. No individual stone positions are surveyed.')
    put(cal/(key+'_capture.json'),capture)
    meta.update(recipe_version='oblique-photo-detail-v2',normal_texture='references/'+key+'_normal.png',
        source_grid_pitch_m=pitch,depth_p95_m=float(np.percentile(-z[inside],95)),
        outline_area_m2=float(cv2.contourArea(np.float32(rim_xy))),
        oriented_extent_m=sorted(cv2.minAreaRect(np.float32(rim_xy))[1],reverse=True),
        geometry_source='Global and locally calibrated inset depth, traced break, scan-derived fine relief',
        detail_material_sources=[r['source'] for r in profile['materials'].values()]+['https://ambientcg.com/a/Asphalt031'])
    put(cal/(key+'.json'),meta)
