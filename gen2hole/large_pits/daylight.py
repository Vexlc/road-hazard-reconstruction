"""Camera-filtered surface colour, diffuse sky visibility and sun shadows on the delivered mesh."""
import io

import cv2
import numpy as np
from PIL import Image

from road_geometry import glb_triangles

LUMA = np.float32([.2126,.7152,.0722])


def bed_texture(path, normal=False):
    for name, _, _, _, blob in glb_triangles(path,names=True,appearance=True,normal_map=normal):
        if name == 'Fracture lip and exposed bed':
            with Image.open(io.BytesIO(blob)) as im:
                return np.array(im.convert('RGB' if normal else 'L'),dtype='f4')
    raise ValueError('Expected the native exposed-bed material')


def sky_visibility(height,pitch):
    """Cosine-weighted fraction of the upper hemisphere above the sampled surface horizon."""
    row,col = np.indices(height.shape,dtype='f4')
    sky = np.zeros_like(height)
    for angle in np.linspace(0,2*np.pi,16,endpoint=False):
        horizon = np.zeros_like(height)
        for r in np.geomspace(1,len(height),26):
            raised = cv2.remap(height,col+r*np.cos(angle),row+r*np.sin(angle),cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT) - height
            horizon = np.maximum(horizon,raised/(r*pitch))
        sky += 1/(1+horizon**2)
    return sky/16


def unit(a):
    return a/np.maximum(1e-7,np.linalg.norm(a,axis=-1,keepdims=True))


def light_maps(glb,width,sun):
    albedo = bed_texture(glb)
    normal = unit(bed_texture(glb,normal=True)/127.5-1)
    with np.load(glb.with_name('surface.npz')) as field:
        height = field['height']
    pitch = width/(len(height)-1)
    h,w = albedo.shape
    # GLB UVs at the outer vertices are 0 and 1; pixel centres lie half a texel inside.
    row,col = np.indices((h,w),dtype='f4')
    field_uv = np.stack(((col+.5)*(len(height)-1)/w,(row+.5)*(len(height)-1)/h),-1)
    def interp(a):
        return cv2.remap(a,field_uv,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    dr,dc = (interp(q) for q in np.gradient(height,pitch))
    geom = unit(np.stack((-dc,dr,np.ones_like(dc)),-1))
    tangent = unit(np.stack((np.ones_like(dc),np.zeros_like(dc),dc),-1))
    bitangent = -np.cross(geom,tangent)
    observed = unit(normal[:,:,:1]*tangent + normal[:,:,1:2]*bitangent + normal[:,:,2:]*geom)
    # A tilted face receives (1+n·up)/2 of a uniform sky. The height field also occludes the sky.
    ambient = interp(sky_visibility(height,pitch))*(1+observed[:,:,2])*.5
    direct = np.maximum(observed@sun,0)/max(.10,float(sun[2]))
    return np.stack((albedo, ambient, direct),-1)


class PitLight:
    def __init__(self,glb,model,sun_info):
        placement,self.mesh,field = model
        self.basis = np.float32(placement['basis_columns'])
        self.centre = np.array(placement['center_world'])
        self.width,self.length = float(field['width']),float(field['length'])
        sun = unit(np.float32(sun_info['sun_world'])@self.basis)
        self.mips = [light_maps(glb,self.width,sun)]
        while min(self.mips[-1].shape[:2])>4:
            self.mips.append(cv2.pyrDown(self.mips[-1]))
        self.directions = []
        if sun_info['direct_light_used'] and sun[2]>0:
            tangent = np.cross(sun,[0,0,1])
            tangent = unit(tangent) if np.linalg.norm(tangent)>1e-5 else np.array([1.,0.,0.])
            ortho = np.cross(sun,tangent)
            for angle in np.arange(4)*np.pi/2:
                d = sun+np.tan(np.deg2rad(.19))*(np.cos(angle)*tangent+np.sin(angle)*ortho)
                self.directions.append(unit(d).astype('f4'))

    def filtered(self,uv,mask):
        """Three taps along the camera pixel's long footprint, mipmapped across the short axis."""
        h,w = self.mips[0].shape[:2]
        centres = uv*np.float32([w,h])-.5
        dx,dy = np.zeros_like(uv),np.zeros_like(uv)
        dx[:,1:] = np.diff(centres,axis=1)*(mask[:,1:] & mask[:,:-1])[...,None]
        dy[1:,:] = np.diff(centres,axis=0)*(mask[1:,:] & mask[:-1,:])[...,None]
        uu = dx[:,:,0]**2+dy[:,:,0]**2
        vv = dx[:,:,1]**2+dy[:,:,1]**2
        cross = dx[:,:,0]*dx[:,:,1]+dy[:,:,0]*dy[:,:,1]
        angle = .5*np.arctan2(2*cross,uu-vv)
        disc = np.hypot(uu-vv,2*cross)
        major = np.sqrt(np.maximum(0,(uu+vv+disc)*.5))
        minor = np.sqrt(np.maximum(0,(uu+vv-disc)*.5))
        level = np.clip(np.log2(np.maximum(1,np.maximum(minor,major/3))),0,len(self.mips)-1)
        step = np.stack((np.cos(angle),np.sin(angle)),-1)*(np.maximum(0,major-minor)/3)[...,None]
        result = np.zeros((*mask.shape,3),'f4')
        for i,tex in enumerate(self.mips):
            weight = np.maximum(0,1-abs(level-i))
            if np.any(mask & (weight>.001)):
                value = np.zeros_like(result)
                for offset in (-1,0,1):
                    grid = (centres+offset*step)/(2**i)
                    value += cv2.remap(tex,grid,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                result += weight[...,None]*(value/3)
        return result.transpose(2,0,1)

    def apply(self,rgb,depth,part,alpha,protected,pose,K,rect,cast,shade):
        mask = (part==2) & ~protected & (alpha>.05)
        shadow = np.zeros(part.shape,'f4')
        if not mask.any():
            return shadow
        y,x = np.nonzero(mask)
        dirs = np.column_stack(((x+rect[0]+.5-K[0,2])/K[0,0],
                                (y+rect[1]+.5-K[1,2])/K[1,1],np.ones(len(x))))
        local = ((pose[:3,3]-self.centre)@self.basis+
                 (dirs@pose[:3,:3].T@self.basis)*depth[y,x,None]).astype('f4')
        uv = np.zeros((*part.shape,2),'f4')
        uv[y,x] = local[:,:2]/np.float32([self.width,-self.length])+.5
        tex,ambient,direct = self.filtered(uv,mask)
        lum = rgb@LUMA
        weight = mask.astype('f4')*alpha
        den = cv2.GaussianBlur(weight,(0,0),2.)
        def smooth(values):
            num = cv2.GaussianBlur(values*weight,(0,0),2.)
            return np.maximum(num/np.maximum(den,.0001),8)
        ratio = (tex/smooth(tex))/np.maximum(.3,lum/smooth(lum))
        detail = np.clip(ratio[y,x],.72,1.35)
        opacity = alpha[y,x,None]
        irradiance = ambient[y,x,None]
        if self.directions:
            origin = local.copy()
            origin[:,2] += .001
            for d in self.directions:
                distance,_,_ = self.mesh.cast(origin,np.broadcast_to(d,origin.shape))
                shadow[y,x] += np.isfinite(distance)/len(self.directions)
            shadow *= alpha
            covered = np.maximum(cast[y,x,None],shadow[y,x,None])/opacity
            sky_fraction = shade**2.2
            irradiance = irradiance*sky_fraction + (1-covered)*direct[y,x,None]*(1-sky_fraction)
        light = np.clip(irradiance,.2,1.65)**(1/2.2)
        gain = 1+opacity*(light*(1+.78*(detail[:,None]-1))-1)
        rgb[y,x] = np.rint(np.clip(rgb[y,x]*gain,0,255))
        return shadow
