"""Render the actual 3D meshes in Cycles; export reusable native and glTF files."""
from pathlib import Path
import os,json,sys,math
import bpy,numpy as np
from mathutils import Vector
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/road_hazards'
BASES=['asphalt_pothole_deep','asphalt_missing_intact','asphalt_cover_raised65','concrete_pothole_deep','concrete_missing_broken','concrete_cover_offset']

def set_camera(cam,pos,target):
    cam.location=pos;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()

def load_material(meta):
    m=bpy.data.materials.new(meta['name']);m.use_nodes=True;ns=m.node_tree.nodes;lk=m.node_tree.links;p=ns.get('Principled BSDF')
    def tex(file,linear=False):
        path=Path(meta[file]);proxy=path.with_stem(path.stem+'_2k')
        if not proxy.exists():
            im=Image.open(path);im.thumbnail((2048,2048));im.save(proxy)
        n=ns.new('ShaderNodeTexImage');n.image=bpy.data.images.load(str(proxy));n.extension='REPEAT'
        if linear:n.image.colorspace_settings.name='Non-Color'
        return n
    color=tex('color');lk.new(color.outputs['Color'],p.inputs['Base Color'])
    rough=tex('roughness',True);lk.new(rough.outputs['Color'],p.inputs['Roughness']);p.inputs['Metallic'].default_value=meta['metallic']
    normal=tex('normal',True);nm=ns.new('ShaderNodeNormalMap');nm.inputs['Strength'].default_value=.15 if meta['name']=='iron' else .5;lk.new(normal.outputs['Color'],nm.inputs['Color']);lk.new(nm.outputs['Normal'],p.inputs['Normal'])
    return m

def render_one(name,secondary=False):
    d=OUT/name;a=np.load(d/'mesh.npz');meta=json.loads((d/'materials.json').read_text());v=a['vertices'];f=a['faces'];uv=a['uv']
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=96;scene.cycles.use_denoising=True
    prefs=bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type='OPTIX';prefs.get_devices()
        for device in prefs.devices:device.use=device.type!='CPU'
        scene.cycles.device='GPU'
    except Exception as e:print(e,flush=True)
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update()
    layer=mesh.uv_layers.new(name='UVMap');ids=np.empty(len(mesh.loops),dtype='i4');mesh.loops.foreach_get('vertex_index',ids);layer.data.foreach_set('uv',uv[ids].reshape(-1))
    obj=bpy.data.objects.new(name,mesh);scene.collection.objects.link(obj)
    for m in meta:mesh.materials.append(load_material(m))
    mesh.polygons.foreach_set('material_index',a['face_material'])
    for poly in mesh.polygons:poly.use_smooth=meta[poly.material_index]['name']!='brick'
    if hasattr(mesh,'set_sharp_from_angle'):mesh.set_sharp_from_angle(angle=math.radians(50))
    world=bpy.data.worlds.new('Overcast sky');scene.world=world;world.use_nodes=True;bg=world.node_tree.nodes['Background'];bg.inputs[0].default_value=(.55,.62,.72,1);bg.inputs[1].default_value=.6
    bpy.ops.object.light_add(type='SUN');sun=bpy.context.object;sun.rotation_euler=(math.radians(22),math.radians(-28),math.radians(-30));sun.data.energy=1.5;sun.data.angle=math.radians(14)
    bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.lens=50;scene.camera=camera
    set_camera(camera,(1.10,-1.50,1.82),(0,0,-.05))
    scene.render.resolution_x=1200;scene.render.resolution_y=900;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='AgX';scene.render.image_settings.file_format='PNG';scene.render.filepath=str(d/'preview.png');bpy.ops.render.render(write_still=True)
    if name in BASES:
        set_camera(camera,(.05,-1.85,.44),(0,0,-.04));scene.render.filepath=str(d/'driver_view.png');bpy.ops.render.render(write_still=True)
        set_camera(camera,(1.10,-1.50,1.82),(0,0,-.05))
        bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(d/(name+'.blend')))
        bpy.context.view_layer.objects.active=obj;obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(d/(name+'.glb')),export_format='GLB',use_selection=True,export_yup=True)
    print('DONE',name,flush=True)

def contacts(catalog):
    font_path=os.environ.get("ROAD_LABEL_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font=ImageFont.truetype(font_path,25)
    titles={'pothole_deep':'道路深坑','missing_intact':'井盖缺失：完整井框','missing_broken':'井盖缺失：破损井沿／砖壁','cover_raised65':'井盖掀起 65°','cover_offset':'井盖横向错位'}
    im=Image.new('RGB',(1800,1000),'#18202a');draw=ImageDraw.Draw(im)
    for i,name in enumerate(BASES):
        p=OUT/name/'preview.png'
        if not p.exists():continue
        tile=Image.open(p).resize((600,450));x=i%3*600;y=i//3*500;im.paste(tile,(x,y+45));prefix='沥青' if name.startswith('asphalt') else '水泥';title=titles.get(name.split('_',1)[1],name)
        draw.text((x+12,y+9),f'{i+1} · {prefix} · {title}',font=font,fill='white')
    im.save(OUT/'六类基础候选.jpg',quality=96)
    font=ImageFont.truetype(font_path,20);allim=Image.new('RGB',(1800,1620),'#18202a');draw=ImageDraw.Draw(allim)
    for i,cfg in enumerate(catalog):
        p=OUT/cfg['id']/'preview.png'
        if not p.exists():continue
        tile=Image.open(p).resize((300,225));x=i%6*300;y=i//6*540;allim.paste(tile,(x,y+36))
        draw.text((x+6,y+8),cfg['id'].replace('asphalt','A').replace('concrete','C'),font=font,fill='white')
        if cfg['type']=='pothole':text=f"宽 {cfg['rx']*2:.2f}m 深 {cfg['depth']*100:.0f}cm"
        elif cfg['type']=='cover':text=f"开盖 {cfg['angle']}°  错位 {cfg['offset']:.2f}m"
        else:text=f"井深 {cfg['depth']:.2f}m"
        draw.text((x+6,y+265),text,font=font,fill='white')
    allim.crop((0,0,1800,3*540-200)).save(OUT/'十八种道路病害候选.jpg',quality=96)

if __name__=='__main__':
    catalog=json.loads((OUT/'catalog.json').read_text());order=BASES+[c['id'] for c in catalog if c['id'] not in BASES]
    for name in order:
        if not (OUT/name/'preview.png').exists():render_one(name)
        contacts(catalog)
