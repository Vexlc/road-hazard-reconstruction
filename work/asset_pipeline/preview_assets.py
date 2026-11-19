from pathlib import Path
import bpy, numpy as np, math, json
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/road_assets'

def camera_at(camera, position, target):
    camera.location=position
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()

def material(d):
    m=bpy.data.materials.new(d.name);m.use_nodes=True
    ns=m.node_tree.nodes;lk=m.node_tree.links;p=ns.get('Principled BSDF')
    def tex(file,linear=False):
        n=ns.new('ShaderNodeTexImage');n.image=bpy.data.images.load(str(d/file))
        if linear:n.image.colorspace_settings.name='Non-Color'
        return n
    c=tex('basecolor.png');lk.new(c.outputs['Color'],p.inputs['Base Color']);lk.new(c.outputs['Alpha'],p.inputs['Alpha'])
    m.surface_render_method='DITHERED' if hasattr(m,'surface_render_method') else None
    arm=tex('metallic_roughness.png',True);s=ns.new('ShaderNodeSeparateColor');lk.new(arm.outputs['Color'],s.inputs[0]);lk.new(s.outputs[1],p.inputs['Roughness']);lk.new(s.outputs[2],p.inputs['Metallic'])
    normal=tex('normal.png',True);n=ns.new('ShaderNodeNormalMap');n.inputs['Strength'].default_value=.55;lk.new(normal.outputs['Color'],n.inputs['Color']);lk.new(n.outputs['Normal'],p.inputs['Normal'])
    return m

def main():
    for name in ['manhole_scan_010','pothole_scan_road2']:
        d=OUT/name
        bpy.ops.wm.read_factory_settings(use_empty=True)
        s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=80;s.cycles.use_denoising=True
        pref=bpy.context.preferences.addons['cycles'].preferences
        try:
            pref.compute_device_type='OPTIX';pref.get_devices()
            for dev in pref.devices:dev.use=dev.type!='CPU'
            s.cycles.device='GPU'
        except Exception as e:print(e,flush=True)
        a=np.load(d/'mesh.npz');v=a['vertices'];f=a['faces'];uv=a['uv']
        mesh=bpy.data.meshes.new(name);mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update()
        layer=mesh.uv_layers.new(name='UVMap');uvs=uv[np.asarray([l.vertex_index for l in mesh.loops])]
        layer.data.foreach_set('uv',uvs.reshape(-1))
        obj=bpy.data.objects.new(name,mesh);s.collection.objects.link(obj);obj.data.materials.append(material(d))
        for p in mesh.polygons:p.use_smooth=True
        world=bpy.data.worlds.new('Soft daylight');s.world=world;world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.42,.46,.52,1);world.node_tree.nodes['Background'].inputs[1].default_value=.65
        bpy.ops.object.light_add(type='AREA',location=(-1.3,-.8,2));bpy.context.object.data.energy=180;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=1.2
        bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.lens=55;s.camera=cam
        s.render.resolution_x=1400;s.render.resolution_y=1100;s.render.resolution_percentage=100
        s.view_settings.view_transform='AgX';s.render.image_settings.file_format='PNG'
        camera_at(cam,(1.05,-1.4,1.45),(0,0,-.025));s.render.filepath=str(d/'preview.png');bpy.ops.render.render(write_still=True)
        clay=bpy.data.materials.new('Geometry diagnostic');clay.diffuse_color=(.48,.51,.55,1);s.view_layers[0].material_override=clay
        s.render.filepath=str(d/'geometry.png');bpy.ops.render.render(write_still=True);s.view_layers[0].material_override=None
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(d/f'{name}.blend'))
        print('DONE',name,flush=True)

if __name__=='__main__':main()
