"""Export the positioned portable mesh pair plus the matching road-cut Gaussian scene."""
from pathlib import Path
import json, numpy as np, trimesh
from PIL import Image
from scipy.ndimage import map_coordinates
from plyfile import PlyData,PlyElement

ROOT=Path(__file__).resolve().parents[2]
R=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)

def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055)

def main():
    for label,sid in [('A','056'),('B','055')]:
        dest=ROOT/'outputs'/f'inserted_{label}_{sid}';m=json.loads((dest/'placement.json').read_text())
        source=ROOT/'outputs'/f'scene_{label}_{sid}'/'scene_frame040.ply'
        ply=PlyData.read(source);data=ply['vertex'].data;xyz=np.stack([data[k] for k in ['x','y','z']],1);cut=np.zeros(len(xyz),bool)
        scene=trimesh.Scene()
        for placement in m['placements']:
            name=placement['asset'];d=ROOT/'outputs/road_assets'/name
            c=np.array(placement['center_world']);basis=np.array(placement['basis_columns'])
            surface=np.load(d/'surface.npz');local=(xyz-c)@basis
            row=(.5-local[:,1]/float(surface['length']))*(surface['alpha'].shape[0]-1);col=(local[:,0]/float(surface['width'])+.5)*(surface['alpha'].shape[1]-1)
            alpha=map_coordinates(surface['alpha'],[row,col],order=1,mode='constant',cval=0)
            cut|=(alpha>.015)&(abs(local[:,2])<.35)
            a=np.load(d/'mesh.npz');v=a['vertices']@basis.T+c
            color=np.array(Image.open(d/'basecolor.png'));cal=placement['calibration'];base=linear(color[:,:,:3].astype('f4')/255)
            lum=np.sum(base*np.array([.2126,.7152,.0722]),-1,keepdims=True);fraction=cal['neutral_color_fraction']
            base=((1-fraction)*base+fraction*lum*np.array(cal['target_tint']))*np.array(cal['linear_gain'])
            color[:,:,:3]=np.uint8(np.clip(srgb(base),0,1)*255)
            # The combined mesh is expressed in glTF's Y-up world axes.
            v=v@R[:3,:3].T
            mat=trimesh.visual.material.PBRMaterial(name=f'{label}_{name}',baseColorTexture=Image.fromarray(color),normalTexture=Image.open(d/'normal.png'),metallicRoughnessTexture=Image.open(d/'metallic_roughness.png'),metallicFactor=1,roughnessFactor=1,alphaMode='BLEND',doubleSided=True)
            mesh=trimesh.Trimesh(v,a['faces'],visual=trimesh.visual.TextureVisuals(uv=a['uv'],material=mat),process=False)
            scene.add_geometry(mesh,node_name=name,geom_name=name)
        scene.export(dest/'assets_world.glb')
        PlyData([PlyElement.describe(data[~cut].copy(),'vertex')],text=False).write(dest/'background_with_road_cut_frame040.ply')
        bundle={'units':'metres','gaussian_axes':'original PandaSet world Z-up','glb_axes':'same world converted to glTF Y-up; x,y,z_glb = x,z,-y_world','gaussians':'background_with_road_cut_frame040.ply','assets':'assets_world.glb','placement':'placement.json','frame':40,'note':'Load Gaussian scene and mesh pair together. Dynamic objects are frozen at frame 40 in this PLY. Use the provided renderer and original checkpoint for the full dynamic sequence.','removed_gaussians':int(cut.sum())}
        (dest/'scene_bundle.json').write_text(json.dumps(bundle,indent=2));print(label,bundle['removed_gaussians'],flush=True)

if __name__=='__main__':main()
