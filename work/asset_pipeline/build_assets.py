"""Reconstruct portable PBR surface meshes from publicly licensed scan height fields."""
from pathlib import Path
import json, shutil
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'work/asset_sources'
OUT=ROOT/'outputs/road_assets'
OUT.mkdir(exist_ok=True)

def resize(a, n):
    return np.array(Image.fromarray(a).resize((n,n), Image.Resampling.BILINEAR))

def make(name, color, height, mask, normal, rough, metal, width, length, meta):
    d=OUT/name;d.mkdir(exist_ok=True)
    # All geometry is in metres, local X right, Y up in the image, Z above road.
    alpha=np.uint8(np.clip(mask,0,1)*255)
    rgba=np.dstack([np.asarray(color)[:,:,:3],alpha])
    Image.fromarray(rgba).save(d/'basecolor.png')
    Image.fromarray(np.uint8(np.clip(normal,0,1)*255)).save(d/'normal.png')
    arm=np.dstack([np.ones_like(rough),rough,metal])
    Image.fromarray(np.uint8(np.clip(arm,0,1)*255)).save(d/'metallic_roughness.png')
    np.savez_compressed(d/'surface.npz',height=height.astype('f4'),alpha=mask.astype('f4'),width=width,length=length)
    mat=trimesh.visual.material.PBRMaterial(name=name,baseColorTexture=Image.open(d/'basecolor.png'),normalTexture=Image.open(d/'normal.png'),metallicRoughnessTexture=Image.open(d/'metallic_roughness.png'),metallicFactor=1,roughnessFactor=1,alphaMode='BLEND',doubleSided=True)
    meshes={}
    for level,n in [('high',769),('medium',385),('low',193)]:
        h=resize(height.astype('f4'),n);a=resize(mask.astype('f4'),n)
        yy,xx=np.meshgrid(np.linspace(.5,-.5,n),np.linspace(-.5,.5,n),indexing='ij')
        vertices=np.stack([xx*width,yy*length,h],-1).reshape(-1,3)
        ids=np.arange(n*n).reshape(n,n)
        # Positive Z winding, image row direction is negative Y.
        f=np.concatenate([np.stack([ids[:-1,:-1],ids[1:,:-1],ids[:-1,1:]],-1).reshape(-1,3),np.stack([ids[1:,:-1],ids[1:,1:],ids[:-1,1:]],-1).reshape(-1,3)])
        f=f[(a.reshape(-1)[f].max(axis=1)>.015)]
        uv=np.stack([xx+.5,yy+.5],-1).reshape(-1,2)
        mesh=trimesh.Trimesh(vertices,f,visual=trimesh.visual.TextureVisuals(uv=uv,material=mat),process=False)
        mesh.remove_unreferenced_vertices()
        mesh.export(d/f'{name}_{level}.glb')
        if level=='medium':
            np.savez_compressed(d/'mesh.npz',vertices=np.asarray(mesh.vertices,dtype='f4'),faces=np.asarray(mesh.faces,dtype='i4'),uv=np.asarray(mesh.visual.uv,dtype='f4'),normals=np.asarray(mesh.vertex_normals,dtype='f4'))
        meshes[level]={'vertices':len(mesh.vertices),'triangles':len(mesh.faces)}
    meta.update(name=name,units='metres',size_m=[width,length,float(height.max()-height.min())],height_range_m=[float(height.min()),float(height.max())],meshes=meshes,geometry_method='dense triangulated scan height field; no image generation',vertical_scale_note='Authoring depth is a chosen physically plausible scale, not a surveyed measurement.')
    (d/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(name,meta['size_m'],meshes,flush=True)

def manhole():
    p=SRC/'ManholeCover010_4K-PNG'
    def load(k):return np.asarray(Image.open(next(p.glob('*_'+k+'.png'))))
    c=load('Color');h=load('Displacement').astype('f4')/65535;a=load('Opacity').astype('f4')/255
    dist=ndimage.distance_transform_edt(a>.5)
    # Feather only the outside 8 mm. Preserve the cast iron lettering and holes.
    a=a*np.clip(dist/32,0,1)
    rim=(a>.5)&(dist<120)
    ref=float(np.median(h[rim]));height=(h-ref)*.055
    height=np.where(a>.01,height,0)
    normal=load('NormalGL').astype('f4')/65535
    if normal.max()<.01:normal=load('NormalGL').astype('f4')/255
    # PNG scalar maps may be either 8 or 16 bit.
    def scalar(k):
        v=load(k);return v.astype('f4')/(65535 if v.dtype.itemsize==2 else 255)
    make('manhole_scan_010',c,height,a,normal,scalar('Roughness'),scalar('Metalness'),1.,1.,{'source':'https://ambientcg.com/view?id=ManholeCover010','license':'CC0-1.0','source_method':'height field photogrammetry','source_resolution':[4096,4096],'source_physical_extent_m':[1,1],'author':'Lennart Demes / ambientCG'})

def pothole():
    h=np.load(SRC/'road_height.npy')
    labels,n=ndimage.label(h<.50)
    counts=np.bincount(labels.ravel());counts[0]=0
    component=labels==int(counts.argmax())
    # A measured scan component, plus a short rim around the broken aggregate.
    outside=ndimage.distance_transform_edt(~component)
    mask=1-np.clip((outside-75)/80,0,1)
    box=(slice(280,1780),slice(140,2480))
    mask=mask[box];hc=h[box]
    edge=(mask>.03)&(mask<.45)
    yy,xx=np.mgrid[:hc.shape[0],:hc.shape[1]]
    A=np.stack([xx[edge]/4096,yy[edge]/4096,np.ones(edge.sum())],1)
    b=np.linalg.lstsq(A,hc[edge],rcond=None)[0]
    plane=b[0]*xx/4096+b[1]*yy/4096+b[2]
    # 16 cm per scan unit gives this pit about 10 cm maximum depth.
    height=np.minimum((hc-plane)*.16,.004)*np.clip(mask*3,0,1)
    height=np.where(mask>.01,height,0).astype('f4')
    c=np.asarray(Image.open(SRC/'road_damaged_2_diff_4k.png'))[box]
    normal=np.asarray(Image.open(SRC/'road_damaged_2_nor_gl_4k.png'))[box]
    normal=normal.astype('f4')/(65535 if normal.dtype.itemsize==2 else 255)
    r=np.asarray(Image.open(SRC/'road_damaged_2_rough_4k.png'))[box];r=r.astype('f4')/(65535 if r.dtype.itemsize==2 else 255)
    make('pothole_scan_road2',c,height,mask,normal,r,np.zeros_like(r),2.2*(2480-140)/4096,2.2*(1780-280)/4096,{'source':'https://polyhaven.com/a/road_damaged_2','license':'CC0-1.0','author':'Dimitrios Savva / Poly Haven','source_resolution':[4096,4096],'source_physical_extent_m':[2.2,2.2],'crop_xyxy_pixels':[140,280,2480,1780],'source_height_format':'float EXR'})

if __name__=='__main__':
    manhole();pothole()
    import runpy
    runpy.run_path(str(Path(__file__).with_name('package_assets.py')))
