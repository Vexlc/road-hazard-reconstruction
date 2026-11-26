"""Render an evidence sheet with source photos or labelled point elevations beside each Gaussian."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

HERE=Path(__file__).resolve().parent


def survey_picture(run,key,wh):
    folder=run/'points';meta=json.loads((folder/'series/assets/pothole'/key/'metadata.json').read_text())
    data=np.load(folder/'evaluation'/key/'survey.npz')
    coords=(data['xyz']-meta['source_origin_xyz'])@np.array(meta['source_local_basis_columns'])
    mask=data['defect'].astype(bool);pit=coords[mask]
    lo=pit[:,:2].min(0);hi=pit[:,:2].max(0);center=(hi+lo)/2
    w,h=wh;scale=min((w-36)/(hi[0]-lo[0]),(h-30)/(hi[1]-lo[1]))
    dest=Image.new('RGB',wh,'#394742');pen=ImageDraw.Draw(dest)
    for keep in [~mask,mask]:
        for p in coords[keep]:
            x,y=(p[:2]-center)*[scale,-scale]+[w/2,h/2]
            t=np.clip(-p[2]/max(np.percentile(-pit[:,2],95),.01),0,1)
            color=tuple(np.uint8(np.array([224,210,149])*(1-t)+np.array([201,103,55])*t)) if keep is mask else (106,120,111)
            pen.ellipse((x-1,y-1,x+1,y+1),fill=color)
    return dest


def review(models,views,out,font_path):
    q=json.loads((views/'quality.json').read_text())
    keys=['A560','M561','P562','P557','P558','P559']
    captions=['Redhill Drive  ·  Chris Downer','Bourne  ·  Bob Harvey','Riverview Road  ·  Kenneth Allen',
              'PointPSSN  ·  test 7','PointPSSN  ·  train 409','PointPSSN  ·  test 3']
    font=lambda size:ImageFont.truetype(str(font_path),size)
    im=Image.new('RGB',(1400,955),'#f3f1e9');d=ImageDraw.Draw(im)
    d.text((30,17),'井盖、路面标线与分层坑槽',font=font(38),fill='#183b32')
    d.text((30,78),'每组左侧为现场照片或扫描高程，右侧为原生高斯模型。扫描图中橙色越深，测点越低。',font=font(21),fill='#425950')
    for i,key in enumerate(keys):
        x=30+(i%2)*684;y=134+(i//2)*269;w,h=317,160
        d.text((x,y),key+'  '+q[key]['label'],font=font(22),fill='#17392f')
        if key.startswith('P5') and key!='P562':
            left=survey_picture(models,key,(w,h))
        else:
            src=Image.open(HERE/'references'/(key+'.jpg')).convert('RGB')
            if key=='M561':src=src.crop((1570,905,5070,3910))
            if key=='P562':src=src.crop((142,399,744,674))
            left=ImageOps.fit(src,(w,h))
        right=Image.open(views/(key+'-near.jpg')).convert('RGB');a=np.asarray(right);bg=a[0,0]
        yy,xx=np.where(np.max(np.abs(a.astype(float)-bg),axis=2)>16)
        right=right.crop((max(0,xx.min()-12),max(0,yy.min()-12),min(right.width,xx.max()+12),min(right.height,yy.max()+12)))
        tile=Image.new('RGB',(w,h),tuple(bg));right.thumbnail((w,h),Image.Resampling.LANCZOS)
        tile.paste(right,((w-right.width)//2,(h-right.height)//2))
        im.paste(left,(x,y+40));im.paste(tile,(x+w+4,y+40))
        d.text((x,y+209),captions[i],font=font(17),fill='#3e554b')
    out.parent.mkdir(parents=True,exist_ok=True);im.save(out,quality=92)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','views','out','font_path'):p.add_argument(key,type=Path)
    review(**vars(p.parse_args()))
