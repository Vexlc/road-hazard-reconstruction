"""Approximate calm-water reflection with a directional probe from the target camera frame."""
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def linear(rgb):
    value = rgb.astype('f4')/255
    return np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)


def encode(value):
    return np.uint8(np.rint(255*np.clip(np.where(value<=.0031308,12.92*value,
                               1.055*np.maximum(0,value)**(1/2.4)-.055),0,1)))


def bilinear(image,coords):
    height,width = image.shape[:2]
    coords = np.clip(coords,[0,0],[width-1.001,height-1.001])
    col,row = coords.astype('i4').T
    dx,dy = (coords-np.floor(coords)).T[:,:,None]
    return ((1-dx)*image[row,col]+dx*image[row,col+1])*(1-dy)+((1-dx)*image[row+1,col]+dx*image[row+1,col+1])*dy


def uv(camera_points,camera):
    return camera_points[:,:2]/np.maximum(camera_points[:,2:],1.e-6)*camera.diagonal()[:2]+camera[:2,2]-.5


def directional_light(source, mirrored, camera):
    coords = uv(mirrored,camera)
    height,width = source.shape[:2]
    valid = (mirrored[:,2]>0)&(coords[:,0]>=0)&(coords[:,1]>=0)&(coords[:,0]<width-1)&(coords[:,1]<height-1)
    probe = cv2.GaussianBlur(source,(0,0),2.2)
    sky = np.median(probe[:max(1,height//6):8,::8],axis=(0,1))
    light = bilinear(probe,coords)
    outside = np.linalg.norm(coords-np.clip(coords,[0,0],[width-1,height-1]),axis=1)
    weight = np.clip(outside/(.18*height),0,1)[:,None]
    weight[mirrored[:,2]<=0] = 1
    weight *= weight*(3-2*weight)
    return light*(1-weight)+sky*weight,valid


def measured_reflections(light, cloud, source, pose, camera, level, screen, water_z):
    if not len(cloud):
        return 0
    points = cloud[cloud[:,2]>level+.04]
    real = (points-pose[:3,3])@pose[:3,:3]
    reflected = points.copy()
    reflected[:,2] = 2*level-reflected[:,2]
    virtual = (reflected-pose[:3,3])@pose[:3,:3]
    direct_uv, mirror_uv = uv(real,camera),uv(virtual,camera)
    keep = ((real[:,2]>.2)&(virtual[:,2]>.2)&(direct_uv>=0).all(1)&
            (direct_uv<np.array(source.shape[1::-1])-1).all(1)&
            (mirror_uv>screen.min(0)-8).all(1)&(mirror_uv<screen.max(0)+8).all(1))
    if keep.sum()<3:
        return 0
    distances,idx = cKDTree(mirror_uv[keep]).query(screen,k=3,distance_upper_bound=8)
    valid = np.isfinite(distances)
    idx = np.minimum(idx,keep.sum()-1)
    depths = virtual[keep,2][idx]
    valid &= depths>water_z[:,None]
    nearest = np.where(valid,depths,np.inf).min(1)
    valid &= depths<nearest[:,None]+1.
    weights = np.exp(-.5*(distances/3.2)**2)*valid
    support = weights.sum(1)
    colors = bilinear(source,direct_uv[keep])
    reflected_rgb = (weights[:,:,None]*colors[idx]).sum(1)/np.maximum(support[:,None],1e-8)
    blend = np.minimum(support/1.7,1)[:,None]
    light[:] = light*(1-blend)+reflected_rgb*blend
    return int((blend>.5).sum())


class ShallowPool:
    def __init__(self, folder, placement, metadata):
        from road_geometry import Surface
        self.bed = Surface(folder/'asset.glb',exclude=(4,))
        self.basis = np.asarray(placement['basis_columns'])
        self.centre = np.asarray(placement['center_world'])
        self.level = self.centre[2]+metadata['water']['free_surface_m']
        with np.load(folder/'surface.npz') as field:
            self.width = float(field['width'])
        refs = folder.parents[2]/'references'
        self.texture = np.asarray(Image.open(refs/(folder.name+'_ortho.png')).convert('RGB'))
        donor = np.asarray(Image.open(refs/(folder.name+'_pavement.jpg')).convert('RGB'))
        self.reference = np.maximum(1,np.median(donor[::4,::4],axis=(0,1)))
        noise = np.random.default_rng(metadata['seed']).normal(size=(192,192)).astype('f4')
        ripple = cv2.GaussianBlur(noise,(0,0),4.,sigmaY=2.)
        along,across = np.gradient(ripple)
        slopes = np.stack((across,-along),axis=-1)
        self.slopes = np.clip(.006*slopes/np.sqrt((slopes**2).sum(2).mean()),-.018,.018)

    def apply(self, after, source, parts, depth, alpha, protected, pose, camera, rect, cloud):
        result = dict(refracted_bed_samples=0,bank_reflection_samples=0,lidar_reflection_samples=0,
                      in_frame_samples=0,fallback_samples=0,water_samples=0)
        bedmask = (parts==2)&~protected&(alpha>.05)
        wet = (parts==4)&~protected&(alpha>.05)
        if not (bedmask|wet).any():
            return result
        base = cv2.GaussianBlur(source,(0,0),3.5)
        def locate(mask):
            rows,cols = np.nonzero(mask)
            screen = np.column_stack((cols+rect[0],rows+rect[1]))
            cam = np.column_stack(((screen+.5-camera[:2,2])/camera.diagonal()[:2],np.ones(len(rows))))
            direction = cam@pose[:3,:3].T
            point = pose[:3,3]+direction*depth[mask,None]
            return point,direction/np.linalg.norm(direction,axis=1,keepdims=True),screen
        def substrate(local, clarity=1.):
            tex = (local[:,:2]/[self.width,-self.width]+.5)*self.texture.shape[1::-1]-.5
            road = local[:,:2]@self.basis[:,:2].T+self.centre
            coords = uv((road-pose[:3,3])@pose[:3,:3],camera)
            low,grain = bilinear(base,coords),bilinear(source,coords)
            albedo = np.clip(bilinear(self.texture,tex)/self.reference,.48,1.4)
            return linear(np.clip((low+.75*clarity*(grain-low))*albedo,0,255))
        if bedmask.any():
            point,_,screen = locate(bedmask)
            local = (point-self.centre)@self.basis
            height = np.maximum(point[:,2]-self.level,0)
            distance = cv2.distanceTransform((~(parts==4)).astype('u1'),cv2.DIST_L2,5)[bedmask]
            damp = np.exp(-height/.010)*np.exp(-distance/14)
            bed = substrate(local)*(1-.28*damp[:,None])
            blend = .56*alpha[bedmask,None]
            material = linear(after[bedmask])*(1-blend)+bed*blend
            cut = np.clip(-local[:,2:]/.024,0,1)
            cut = cut*(2-cut)
            host = linear(source[screen[:,1],screen[:,0]])
            after[bedmask] = encode(material*cut+host*(1-cut))
        if not wet.any():
            return result
        point,incident,_ = locate(wet)
        local = (point-self.centre)@self.basis
        detail = bilinear(self.slopes,(local[:,:2]/[self.width,-self.width]+.5)*191)
        resolved = np.clip((camera[0,0]*.04/depth[wet]-1)/3,0,1)[:,None]
        perturb = (detail*resolved)@self.basis[:,:2].T
        normal = perturb + [0.,0.,1.]
        normal /= np.linalg.norm(normal,axis=1,keepdims=True)
        cosine = np.clip(-(incident*normal).sum(1),0,1)
        transmitted_cos = np.sqrt(1-(1-cosine**2)/1.333**2)
        refracted = incident/1.333+(cosine/1.333-transmitted_cos)[:,None]*normal
        direction = refracted@self.basis
        path,_,_ = self.bed.cast(local,direction)
        hit = np.isfinite(path)&(path>=0)
        thickness = np.where(hit,path,0)
        bottom = substrate(local+direction*thickness[:,None],np.exp(-thickness[:,None]/.055))
        trans = np.exp(-thickness[:,None]*[7.,5.5,4.5])
        bottom *= .72*(trans+.12*(1-trans))
        mirrored = incident+2*cosine[:,None]*normal
        light, valid = directional_light(source,mirrored@pose[:3,:3],camera)
        planar = mirrored.copy()
        planar[:,2] *= -1
        measured = measured_reflections(light,cloud,source,pose,camera,self.level,uv(planar@pose[:3,:3],camera),depth[wet])
        upwards = mirrored@self.basis
        bounce,_,_ = self.bed.cast(local+upwards*.0003,upwards)
        bank = np.isfinite(bounce)&(bounce>0)
        reflected = linear(light)
        if bank.any():
            reflected[bank] = .79*substrate(local[bank]+upwards[bank]*(bounce[bank,None]+.0003))
        rs = (cosine-1.333*transmitted_cos)/(cosine+1.333*transmitted_cos)
        rp = (1.333*cosine-transmitted_cos)/(1.333*cosine+transmitted_cos)
        fresnel = .5*(rs**2+rp**2)
        cover = np.clip(thickness*transmitted_cos/.003,0,1)
        fraction = (fresnel*cover)[:,None]
        shaded = bottom*(1-fraction)+reflected*fraction
        coverage = alpha[wet,None]*hit[:,None]
        after[wet] = encode(shaded*coverage+linear(after[wet])*(1-coverage))
        result.update(water_samples=len(path),refracted_bed_samples=int(hit.sum()),bank_reflection_samples=int(bank.sum()),
                      lidar_reflection_samples=measured,in_frame_samples=int(valid.sum()),fallback_samples=int((~valid).sum()))
        return result


def reflect(after, source, parts, alpha, protected, pose, camera, rect):
    rows, cols = np.nonzero((parts==4)&~protected&(alpha>.05))
    if not len(rows):
        return dict(in_frame_samples=0, fallback_samples=0)
    rays = np.column_stack(((cols+rect[0]+.5-camera[0,2])/camera[0,0],
                            (rows+rect[1]+.5-camera[1,2])/camera[1,1],np.ones(len(rows))))
    world = rays@pose[:3,:3].T
    world /= np.linalg.norm(world,axis=1,keepdims=True)
    cosine = np.clip(-world[:,2],0,1)
    world[:,2] *= -1
    mirrored = world@pose[:3,:3]
    uv = mirrored[:,:2]/np.maximum(mirrored[:,2:],1e-8)*camera.diagonal()[:2]+camera[:2,2]-.5
    height,width = source.shape[:2]
    valid = (mirrored[:,2]>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<width-1)&(uv[:,1]<height-1)
    probe = cv2.GaussianBlur(source,(0,0),2.2)
    sky = np.median(probe[:max(1,height//6):8,::8],axis=(0,1))
    light = np.broadcast_to(sky,(len(rows),3)).copy()
    above = (mirrored[:,2]>0)&(uv[:,1]<0)
    if above.any():
        column = np.rint(uv[above,0]).clip(0,width-1).astype('i4')
        upper = cv2.GaussianBlur(probe[:8].mean(axis=0,keepdims=True),(0,0),6)[0]
        sky_weight = np.clip(-uv[above,1]/(.18*height),0,1)[:,None]
        sky_weight *= sky_weight*(3-2*sky_weight)
        light[above] = upper[column]*(1-sky_weight)+sky*sky_weight
    coords = uv[valid]; col,row = coords.astype('i4').T
    dx,dy = (coords-np.floor(coords)).T[:,:,None]
    light[valid] = ((1-dx)*probe[row,col]+dx*probe[row,col+1])*(1-dy) + ((1-dx)*probe[row+1,col]+dx*probe[row+1,col+1])*dy
    normal_reflectance = ((1.333-1)/(1.333+1))**2
    fraction = (normal_reflectance+(1-normal_reflectance)*(1-cosine)**5)*alpha[rows,cols]
    shaded = linear(after[rows,cols])*(1-fraction[:,None])+linear(light)*fraction[:,None]
    srgb = np.where(shaded<=.0031308,12.92*shaded,1.055*shaded**(1/2.4)-.055)
    after[rows,cols] = np.uint8(np.clip(np.rint(srgb*255),0,255))
    return dict(in_frame_samples=int(valid.sum()),fallback_samples=int((~valid).sum()))
