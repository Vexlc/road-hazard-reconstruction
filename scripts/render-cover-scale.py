"""Render captured road-cover families using a common orthographic camera in metres."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import bpy
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gen2hole/recipes/common'),str(ROOT/'gen2hole/recipes/series')]
from render_mesh import prepare_view

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def render(runs,out):
    width,height=420,310
    scale=1.28
    margin=25
    pitch=width+14
    row_height=height+110
    page=Image.new('RGB',(2*margin+pitch*3-14,130+len(runs)*row_height+45),'#f3f2ed')
    draw=ImageDraw.Draw(page)
    def word(x,y,s,size,bold=False):
        font=ImageFont.truetype(FONT.replace('Regular','Bold') if bold else FONT,size,index=2)
        draw.text((x,y),s,font=font,fill='#183629',anchor='lt')
    word(margin,18,'实拍道路盖板　同尺度三维对照',40,True)
    word(margin,77,'六个 GLB 使用相同正交相机、光照和米制取景范围。右侧标尺为画面水平方向的实长。',21)
    with TemporaryDirectory() as tmp:
        for idx,run in enumerate(runs):
            bundle=run/'series'
            spec=json.loads((bundle/'calibration/survey.json').read_text())
            obs=json.loads((bundle/'calibration/capture-fit.json').read_text())
            if len(spec['states'])!=3:
                raise ValueError('Select three states per source for this comparison')
            top=126+idx*row_height
            title=f"PandaSet {obs['sequence']}    金属外径 {obs['frame_diameter_m']:.3f} m"
            if 'apron_xy_extent_m' in obs:
                w,h=obs['apron_xy_extent_m']
                title+=f"    修补圈 {w:.3f} × {h:.3f} m"
            word(margin,top,title,26,True)
            x0=page.width-margin-120
            bar=round(.25*width/scale)
            draw.line((x0,top+21,x0+bar,top+21),fill='#183629',width=3)
            for xx in (x0,x0+bar):
                draw.line((xx,top+15,xx,top+26),fill='#183629',width=3)
            word(x0+5,top-5,'0.25 m',15)
            for j,state in enumerate(spec['states']):
                folder=bundle/'assets'/('missing' if state.get('absent') else 'raised')/state['id']
                scene,_=prepare_view(folder,bundle/'mesh_cache','inspection',width,height)
                direction=Vector((0.,-.65,1.2)).normalized()
                scene.camera.location=Vector((0.,0.,.12))+direction*3.4
                scene.camera.rotation_euler=(-direction).to_track_quat('-Z','Y').to_euler()
                scene.camera.data.ortho_scale=scale
                target=Path(tmp)/(state['id']+'.jpg')
                scene.render.filepath=str(target)
                bpy.ops.render.render(write_still=True)
                x=margin+j*pitch
                with Image.open(target) as im:
                    page.paste(im,(x,top+53))
                word(x,top+height+64,state['id']+'   '+state['label'],22)
    word(margin,page.height-36,'原始照片与激光点来自 PandaSet。异常状态及遮挡面使用模型设计值。Scale AI / Hesai · CC BY 4.0 and PandaSet Dataset Terms.',16)
    page.save(out,quality=94,subsampling=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    render(args.runs,args.out)
