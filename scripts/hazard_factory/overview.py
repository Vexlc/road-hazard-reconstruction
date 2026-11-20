"""Paginated 4x4 physical-design catalog, with actual 360-degree renders."""
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

from present import orbit,font
from run import atomic_json
from media_publish import publication


def main():
    job=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf-8'))
    root=Path(job['output']).parent
    library=Path(job['libraries'][-1])
    models=json.loads((library/'asset_manifest.json').read_text(encoding='utf-8'))
    plan=Path(job['work'])/'case_plan.json';order={}
    if plan.exists():
        names=[n for g in json.loads(plan.read_text(encoding='utf-8'))['groups'] for n in g['assets']]
        order={n:i for i,n in enumerate(names)}
    models.sort(key=lambda m:(order.get(m['id'],10**9),m['stream_seed'],m['stream_index'],m['id']))
    cache=Path(job['work'])/'orbits';cache.mkdir(exist_ok=True)
    pages=[models[first:first+16] for first in range(0,len(models),16)]
    labels={'pothole':'坑洞','missing':'缺盖井','raised':'异常井盖'}
    with publication(root,'4x4视频索引.json') as stage:
        output=stage/f'{len(models)}个设计资产_4x4环绕.mp4';index=[]
        with imageio.get_writer(str(output),fps=24,codec='libx264',quality=8,macro_block_size=2,
                                output_params=['-threads','3','-movflags','+faststart']) as video:
            for page,records in enumerate(pages):
                frames=[orbit(dict(m,ply=str(library/m['ply'])),cache) for m in records]
                base=Image.new('RGB',(2560,1920),(18,23,29));d=ImageDraw.Draw(base)
                d.text((26,16),f'{len(models)} 个设计资产 · 4×4 环绕 · {page+1}/{len(pages)} 页 · 尺度 1.0×',font=font(40),fill='white')
                for i,m in enumerate(records):
                    x=16+(i%4)*632;y=80+(i//4)*448
                    d.text((x+12,y+411),f"{page*16+i+1:02d} {labels[m['category']]} · {m['width_m']*100:.0f}×{m['length_m']*100:.0f} cm · {m['material']}",font=font(21),fill='white')
                d.text((26,1880),'扫描材质驱动的合成三维几何；尺寸为设计值，非实物测量。空格保持留空。',font=font(23),fill='#bac8d6')
                for frame in range(96):
                    image=base.copy()
                    for i,values in enumerate(frames):
                        x=16+(i%4)*632+44;y=80+(i//4)*448
                        image.paste(Image.fromarray(values[frame]).resize((544,408),Image.Resampling.LANCZOS),(x,y))
                    video.append_data(np.asarray(image))
                    if frame==0:image.save(stage/f'设计资产_4x4预览_{page+1:02d}.jpg',quality=93)
                index.append({'page':page+1,'start_seconds':page*4,'seconds':4,'assets':[m['id'] for m in records],
                              'empty_cells':16-len(records)})
        atomic_json(stage/'4x4视频索引.json',{'file':output.name,'pages':index,'assets':len(models),'seconds':len(pages)*4})
    print('4x4 OVERVIEW COMPLETE',flush=True)


if __name__=='__main__':main()
