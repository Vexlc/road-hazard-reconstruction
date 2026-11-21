"""Render the delivered glTFs with Cycles in an overcast inspection setup."""
import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from mesh_io import ROOT


def ring_ground(extent, meta, grain):
    # This photographic apron exists only outside the finite delivered GLB.
    mat=bpy.data.materials.new('Inspection ground outside GLB');mat.use_nodes=True
    tree=mat.node_tree;shader=tree.nodes.get('Principled BSDF')
    image=tree.nodes.new('ShaderNodeTexImage');image.image=bpy.data.images.load(str(grain.resolve()));image.extension='MIRROR'
    tree.links.new(image.outputs['Color'],shader.inputs['Base Color'])
    shader.inputs['Roughness'].default_value=.9
    bump=tree.nodes.new('ShaderNodeBump');bump.inputs['Distance'].default_value=.0008;bump.inputs['Strength'].default_value=.2
    tree.links.new(image.outputs['Color'],bump.inputs['Height']);tree.links.new(bump.outputs['Normal'],shader.inputs['Normal'])
    q=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    xy=np.vstack([q*extent,q*extent*6.0])
    face=[(i,4+i,4+(i+1)%4,(i+1)%4) for i in range(4)]
    me=bpy.data.meshes.new('Outside footprint');me.from_pydata(np.c_[xy,np.zeros(8)],[],face);me.update()
    ob=bpy.data.objects.new('Inspection ground extension',me);bpy.context.collection.objects.link(ob);me.materials.append(mat)
    uv=me.uv_layers.new();cx,cy,px,py,n=meta['road_projection']
    factor=(n-1)/(479.*(image.image.size[0]-1))
    for loop in me.loops:
        x,y=xy[loop.vertex_index]
        uv.data[loop.index].uv=((cx+x*px)*factor,1-(cy-y*py)*factor)


def prepare_view(dest,cache,kind,width,height):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str((dest/'asset.glb').resolve()))
    scene=bpy.context.scene
    road=next(x for x in scene.objects if x.type=='MESH' and x.name.startswith('Road '))
    with np.load(dest/'surface.npz') as f:extent=float(f['width'])/2
    meta=json.loads((dest/'metadata.json').read_text())
    ring_ground(extent,meta,cache/(meta['id']+'_ground.jpg'))
    scene.render.engine='CYCLES'
    scene.cycles.samples=24
    scene.cycles.use_denoising=True
    scene.cycles.max_bounces=5
    scene.render.threads_mode='FIXED';scene.render.threads=12
    scene.world=bpy.data.worlds.new('Overcast sky');scene.world.use_nodes=True
    world=scene.world.node_tree.nodes.get('Background')
    world.inputs[0].default_value=(.94,.965,1,1);world.inputs[1].default_value=.72
    lamps=[((-1.6,1.7,3.8),180.,4.5, (1.,.975,.93)),((2.4,-.4,2.6),40.,3.0,(.95,.97,1.))]
    for i,(p,power,size,col) in enumerate(lamps):
        light=bpy.data.lights.new('Diffuse sky '+str(i),'AREA');light.energy=power;light.shape='DISK';light.size=size;light.color=col
        ob=bpy.data.objects.new(light.name,light);scene.collection.objects.link(ob);ob.location=p
        ob.rotation_euler=(-ob.location).to_track_quat('-Z','Y').to_euler()
    scene.view_settings.view_transform='Standard'
    scene.view_settings.look='None'
    with np.load(cache/(meta['id']+'.npz')) as m:
        names=m['part_names']
        include=np.array([('Fracture' in x or 'Water' in x or 'Cast ring' in x or x.startswith('Lid') or 'collar' in x) for x in names])
        points=m['xyz'][np.unique(m['faces'][include[m['face_part']]])]
    cat=meta['category']
    if cat=='pothole':
        direction=[-.22,-1.,1.33] if kind=='inspection' else [.8,.91,.78]
    elif meta['state']=='side_hinge':
        direction=[-.49,-1.,1.44] if kind=='inspection' else [1.0,-.64,.61]
    elif meta['state']=='end_hinge' and meta['recipe_version']=='image-fit-v1':
        direction=[1.,.36,1.25] if kind=='inspection' else [-.53,-1.,.82]
    elif meta['state']=='end_hinge':
        direction=[-.50,-1.,1.12] if kind=='inspection' else [.84,.91,.86]
    else:
        direction=[.16,-1.,1.6] if kind=='inspection' else [-.43,.9,.79]
    away=np.array(direction);away/=np.linalg.norm(away)
    right=np.cross([0,0,1.],away);right/=np.linalg.norm(right)
    up=np.cross(away,right)
    p=points@np.stack([right,up,away],axis=1)
    lo=p.min(0);hi=p.max(0)
    center=((hi+lo)/2)@np.stack([right,up,away])
    aspect=width/height; scale=max((hi[0]-lo[0])*1.23,(hi[1]-lo[1])*aspect*1.23)
    data=bpy.data.cameras.new('Inspection');data.type='ORTHO';data.ortho_scale=scale
    cam=bpy.data.objects.new('Inspection',data);scene.collection.objects.link(cam)
    cam.location=center+away*3.4;cam.rotation_euler=Vector(-away).to_track_quat('-Z','Y').to_euler();scene.camera=cam
    scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='JPEG';scene.render.image_settings.quality=96
    return scene, points


def render(dest,cache,kind,output,width=840,height=568):
    scene, _ = prepare_view(dest, cache, kind, width, height)
    scene.render.filepath=str(output.resolve())
    bpy.ops.render.render(write_still=True)
    print('DONE',output,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--angles',default='inspection,reverse')
    ap.add_argument('ids',nargs='*')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for d in sorted((ROOT/'assets').glob('*/*/asset.glb')):
        if args.ids and d.parent.name not in args.ids:continue
        for view in args.angles.split(','):
            render(d.parent,args.cache,view,args.out/(d.parent.name+'_'+view+'.jpg'))
    # The bpy wheel can crash when its background compositor is finalized at process exit.
    # Every synchronous render has finished writing at this point.
    import os,sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
