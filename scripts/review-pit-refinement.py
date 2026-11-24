"""Compare two P190 GLBs on the same asphalt datum, with identical cameras, lighting and normal mapping."""
import argparse
from functools import lru_cache
import io
import json
from pathlib import Path
import struct
import sys

import cv2
import imageio.v2 as imageio
import nvdiffrast.torch as dr
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles


def unit(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-10)


def read(path):
    return json.loads(path.read_text())


def normal_image(path):
    data=path.read_bytes()
    length=struct.unpack_from('<I',data,12)[0]
    scene=json.loads(data[20:20+length])
    i=scene['materials'][0]['normalTexture']['index']
    view=scene['bufferViews'][scene['images'][scene['textures'][i]['source']]['bufferView']]
    start=28+length+view.get('byteOffset',0)
    return np.array(Image.open(io.BytesIO(data[start:start+view['byteLength']])).convert('RGB'))


def horizon(z,pitch,light):
    y,x=np.indices(z.shape,dtype='f4')
    sky=np.zeros_like(z)
    def slope(dx,dy):
        m=np.zeros_like(z)
        for distance in np.geomspace(pitch*1.3,1.9,19):
            v=cv2.remap(z,x+dx*distance/pitch,y-dy*distance/pitch,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            m=np.maximum(m,(v-z)/distance)
        return m
    for theta in np.linspace(0,2*np.pi,14,endpoint=False):
        m=slope(np.cos(theta),np.sin(theta))
        sky+=1/(1+m*m)
    horizontal=np.linalg.norm(light[:2])
    obstruction=slope(*(light[:2]/horizontal))
    direct=np.clip(.5+3.5*(light[2]/horizontal-obstruction),0,1)
    return np.stack([sky/14,direct],-1)


def gpu(a,dtype=torch.float32):
    return torch.as_tensor(np.array(a,copy=True,order='C'),device='cuda',dtype=dtype)


class Scene:
    def __init__(self,bundle,ctx):
        self.ctx=ctx
        folder=bundle/'assets/pothole/P190'
        glb=folder/'asset.glb'
        # Load all triangles and the embedded material of the delivered GLB.
        vv=[];ff=[];uu=[];ptr=0
        for name,vertices,faces,uv,texture in glb_triangles(glb,names=True,appearance=True):
            vv.append(vertices);ff.append(faces.astype('i4')+ptr);uu.append(uv);ptr+=len(vertices)
        vertices=np.vstack(vv);faces=np.vstack(ff);uv=np.vstack(uu)
        self.light=unit(np.float32([-.65,.33,.64]))
        with np.load(folder/'surface.npz') as f:
            z=f['height'];self.span=float(f['width']);pitch=self.span/(len(z)-1)
        sky=horizon(z,pitch,self.light)
        # A shared road apron extends past the square tile for an uninterrupted camera view.
        corners=np.float32([[-100,-100,-.002],[100,-100,-.002],[100,100,-.002],[-100,100,-.002]])
        # Only the ring beyond the tile is meshed, so it can never occlude the pit floor.
        inner=corners.copy();inner[:,:2]*=self.span/200;inner[:,2]=0
        ext=np.vstack([inner,corners]);ring=[]
        for j in range(4):
            k=(j+1)%4
            ring.extend([(j,4+j,4+k),(j,4+k,k)])
        faces=np.vstack([faces,np.int32(ring)+len(vertices)])
        verts=np.vstack([vertices,ext])
        combined_uv=np.vstack([uv,np.column_stack([ext[:,0]/self.span+.5,.5-ext[:,1]/self.span])])
        self.v=gpu(verts);self.f=gpu(faces,torch.int32)
        dy,dx=np.gradient(z,pitch)
        grid_rc=np.rint(uv[:,::-1]*[len(z)-1,len(z)-1]).astype(int)
        n=unit(np.column_stack([-dx[tuple(grid_rc.T)],dy[tuple(grid_rc.T)],np.ones(len(uv))]))
        normals=np.vstack([n,np.tile([0,0,1],(8,1))])
        self.attr=gpu(np.column_stack([combined_uv,normals,verts]))
        self.albedo=gpu(np.array(Image.open(io.BytesIO(texture)).convert('RGB'))/255.)[None]
        self.normal=gpu(normal_image(glb)/255.)[None]
        self.sky=gpu(sky)[None]
        self.apron=gpu(np.asarray(Image.open(bundle/'references/P190_pavement.jpg').convert('RGB'))/255.)[None]

    @torch.inference_mode()
    def render(self,eye,target,w,h):
        forward=unit(target-eye);right=unit(np.cross(forward,[0,0,1.]));down=np.cross(forward,right)
        camera=(self.v-gpu(eye))@gpu(np.column_stack([right,down,forward]))
        cx,cy,depth=camera.unbind(1)
        f=.5*h/np.tan(np.deg2rad(32)/2)
        clip=torch.stack([cx*(2*f/w),cy*(2*f/h),depth*1.0002-.020002,depth],-1).contiguous()
        raster,der=dr.rasterize(self.ctx,clip[None],self.f,(h,w))
        attrs,grads=dr.interpolate(self.attr[None],raster,self.f,rast_db=der,diff_attrs=[0,1])
        uv=attrs[...,:2].contiguous(); uv_da=grads[...,:4].contiguous()
        def sample(tex):
            return dr.texture(tex,uv,uv_da=uv_da,filter_mode='linear-mipmap-linear',boundary_mode='clamp',
                              max_mip_level=7)
        # Non-power-of-two raster inputs cannot build mipmaps. Resample once during loading.
        color=sample(self.albedo)
        normal=sample(self.normal)*2-1
        visibility=dr.texture(self.sky,uv.contiguous(),filter_mode='linear',boundary_mode='clamp')
        geom=torch.nn.functional.normalize(attrs[...,2:5],dim=-1)
        tangent=torch.zeros_like(geom);tangent[...,0]=1
        tangent=torch.nn.functional.normalize(tangent-geom*geom[...,:1],dim=-1)
        bitangent=-torch.cross(geom,tangent,dim=-1)
        ns=torch.nn.functional.normalize(geom*normal[...,2:]+tangent*normal[...,:1]+bitangent*normal[...,1:2],dim=-1)
        edge=torch.max(torch.abs(uv-.5),dim=-1,keepdim=True)[0]*self.span
        ext=((edge-.5*self.span+.15)/.15).clamp(0,1)
        road_uv=(attrs[...,5:7]*gpu([1.,-1.])/2.45).contiguous()
        road=dr.texture(self.apron,road_uv,filter_mode='linear',boundary_mode='wrap')
        color=color*(1-ext)+road*ext
        ns=ns*(1-ext)+gpu([0.,0.,1.])*ext
        visible=visibility*(1-ext)+ext
        # One diffuse overcast sky and one low, broad sun are fixed for both models and every frame.
        light=(.49+.12*ns[...,2:])*visible[...,:1]+.56*(ns*gpu(self.light)).sum(-1,keepdim=True).clamp(0,1)*visible[...,1:]
        lit=(color.clamp(0,1).pow(2.2)*light).clamp(0,1).pow(1/2.2)
        return np.uint8(np.clip(lit[0].cpu().numpy()*255+.5,0,255))


def resize_textures(scene):
    import torch.nn.functional as F
    for name in ('albedo','normal'):
        tex=getattr(scene,name)
        n=2**int(np.rint(np.log2(tex.shape[1])))
        if tex.shape[1:3]!=(n,n):
            tex=F.interpolate(tex.permute(0,3,1,2),size=(n,n),mode='bilinear',align_corners=True)
            setattr(scene,name,tex.permute(0,2,3,1).contiguous())


@lru_cache()
def font(n):
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',n,index=2)


def caption(im,xy,text,n=21):
    ImageDraw.Draw(im).text(xy,text,font=font(n),anchor='lt',fill='#243031')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stills-only',action='store_true')
    a=p.parse_args();out=a.out;out.mkdir(exist_ok=True,parents=True)
    ctx=dr.RasterizeCudaContext(device='cuda')
    old,new=[Scene(folder,ctx) for folder in (a.old,a.new)]
    for scene in (old,new):resize_textures(scene)
    cap=read(a.new/'calibration/P190_capture.json');plane=cap['road_planes']['direct']
    K=np.array(plane['K']);axes=np.array(cap['local_axes_in_camera']);origin=np.array(cap['local_origin_in_camera'])
    def local(x,y):
        ray=np.linalg.solve(K,[x,y,1.]);return (ray*(-plane['offset_m']/(ray@plane['normal']))-origin)@axes
    with np.load(a.new/'assets/pothole/P190/surface.npz') as field:
        rows,cols=np.where(field['defect_mask']>.5)
        pitch=float(field['width'])/(field['height'].shape[0]-1)
        xy=np.float32(np.column_stack([cols*pitch-new.span/2,new.span/2-rows*pitch]))
        edge=cv2.convexHull(xy).reshape(-1,2)
        hull=np.vstack([np.column_stack([edge,np.full(len(edge),z)]) for z in (0,-.25)])
    # Frame along the actual lip, then move in toward its dark, broken left shoulder.
    far=local(607,313);near=local(966,438)
    mid=np.r_[((far+near)/2)[:2],-.10]
    tangent=unit((near-far)[:2]);across=np.float32([-tangent[1],tangent[0]])
    if across[1]>0:across=-across
    theta0=np.arctan2(across[1],across[0])
    wall=local(665,317);wall=.72*wall+.28*mid;wall[2]=-.095
    def view(scene,phase,w,h,close=False):
        phase=float(phase)
        if close:
            target=(1-phase)*(.55*mid+.45*wall)+phase*wall
            th=theta0+.42-.57*phase;elev=np.deg2rad(39+11*phase);radius=4.5-1.45*phase
        else:
            target=mid;th=theta0-.45+.90*phase;elev=np.deg2rad(35+18*np.sin(np.pi*phase))
            away=np.array([np.cos(th)*np.cos(elev),np.sin(th)*np.cos(elev),np.sin(elev)])
            right=unit(np.cross(-away,[0,0,1.]));up=np.cross(away,right)
            basis=np.column_stack([right,up,away])
            bounds=hull@basis
            target=.5*(bounds.min(0)+bounds.max(0))@basis.T
            screen=(hull-target)@basis
            slope=np.tan(np.deg2rad(32)/2)*.86
            radius=np.max(screen[:,2]+np.maximum(abs(screen[:,0])/(slope*w/h),abs(screen[:,1])/slope))
        eye=target+radius*np.array([np.cos(th)*np.cos(elev),np.sin(th)*np.cos(elev),np.sin(elev)])
        return Image.fromarray(scene.render(eye,target,w,h))
    spec=read(a.new/'calibration/P190-outline.json')
    ref=Image.open(a.new/'references/P190.jpg').convert('RGB').crop(tuple(spec['crop']))
    plate=Image.new('RGB',(1520,1050),'#f5f4ee')
    caption(plate,(30,26),'P190   护栏旁的路面塌坑',39)
    caption(plate,(31,100),'同一机位、灯光与比例',22)
    for i,(scene,title) in enumerate([(old,'上一版'),(new,'优化版')]):
        x=20+i*750;caption(plate,(x+10,160),title,27)
        plate.paste(view(scene,.48,730,382),(x,207))
        plate.paste(view(scene,.92,730,338,True),(x,625))
    caption(plate,(30,982),'下排为同一区域的断口与坑底近景。细粒材质来自扫描样本。',21)
    plate.save(out/'before-after.jpg',quality=96,subsampling=0)
    view(new,.48,1440,810).save(out/'refined.jpg',quality=96,subsampling=0)
    if a.stills_only:return
    fps=24
    with imageio.get_writer(str(out/'comparison.partial.mp4'),fps=fps,codec='libx264',quality=8,
            macro_block_size=2,output_params=['-threads','3','-movflags','+faststart']) as movie:
        start=Image.new('RGB',(1440,810),'#f5f4ee')
        caption(start,(35,24),'P190   护栏旁的路面塌坑',37)
        caption(start,(38,86),'原始照片',22)
        raw=ImageOps.contain(ref,(1380,640))
        start.paste(raw,((1440-raw.width)//2,145))
        for _ in range(36):movie.append_data(np.asarray(start))
        scene=start.copy()
        ImageDraw.Draw(scene).rectangle((29,81,1040,117),fill='#f5f4ee')
        caption(scene,(38,86),'原机位 · 优化模型合成到实拍街景',22)
        road=ImageOps.contain(Image.open(out/'street-view.jpg'),(1380,640))
        scene.paste(road,((1440-road.width)//2,145))
        for _ in range(36):movie.append_data(np.asarray(scene))
        page=Image.new('RGB',(1440,810),'#f5f4ee')
        caption(page,(27,27),'上一版',32);caption(page,(739,27),'优化版',32)
        caption(page,(29,713),'同一机位 · 同一灯光',23)
        caption(page,(740,713),'实拍轮廓与断口 · 扫描颗粒细节',23)
        for j in range(120):
            t=.5-.5*np.cos(np.pi*j/119)
            for i,scene in enumerate((old,new)):page.paste(view(scene,t,704,550),(12+i*712,106))
            movie.append_data(np.asarray(page))
        page=Image.new('RGB',(1440,810),'#f5f4ee')
        caption(page,(28,23),'优化版   /   坑沿与土石近景',30)
        for j in range(132):
            t=.5-.5*np.cos(np.pi*j/131)
            page.paste(view(new,t,1440,718,True),(0,80))
            movie.append_data(np.asarray(page))
    (out/'comparison.partial.mp4').replace(out/'comparison.mp4')


if __name__=='__main__':
    main()
