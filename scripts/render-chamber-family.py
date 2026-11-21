"""Compare one asset family on the same calibrated camera frame and crop in every state."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD=FONT.replace('Regular','Bold')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def extract(path, number, ffmpeg):
    output=subprocess.check_output([ffmpeg,'-v','error','-i',str(path),'-vf',f'select=eq(n\\,{number})',
                                    '-frames:v','1','-f','image2pipe','-vcodec','png','-threads','1','pipe:1'])
    return Image.open(BytesIO(output)).convert('RGB')


def shared_frame(bundle, site, ids):
    cases={}
    for folder in sorted((bundle/'sensor_views').glob(site+'_*')):
        record=read(folder/'frames.json')
        if len(record['assets'])==1 and record['assets'][0] in ids:
            aid,=record['assets']
            if aid in cases:raise ValueError('More than one placed '+aid+' on '+site)
            cases[aid]=(folder,record)
    if set(cases)!=set(ids):raise ValueError('Missing front views for some family states')
    visible={}
    for aid,(_,record) in cases.items():
        width,height=record['size']
        good={}
        for i,row in enumerate(record['frames']):
            boxes=[b for b in row['class_bboxes_xyxy'] if b]
            if len(boxes)==1:
                box=boxes[0]
                if box[0]>10 and box[1]>10 and box[2]<width-10 and box[3]<height-10:
                    good[row['frame']]=(max(row['class_pixels']),box,i)
        visible[aid]=good
    common=set.intersection(*(set(v) for v in visible.values()))
    if not common:raise ValueError('No frame contains all states fully inside the image')
    frame=max(sorted(common),key=lambda n:min(visible[k][n][0] for k in ids))
    boxes=[visible[k][frame][1] for k in ids]
    x0,y0,x1,y1=[fn(b[i] for b in boxes) for i,fn in enumerate((min,min,max,max))]
    w=max(x1-x0+68,int((y1-y0+32)*2.35))
    h=round(w/2.35)
    x=max(0,min(width-w,(x0+x1-w)//2));y=max(0,min(height-h,(y0+y1-h)//2))
    crop=(x,y,x+w,y+h)
    views=bundle/'comparison'/site
    views.mkdir(parents=True,exist_ok=True)
    cfg=read(ROOT/'gen2hole/config/local.json')
    with Image.open(Path(cfg['pandaset_processed'])/site/'images'/f'{frame:03d}_0.jpg') as im:
        im.crop(crop).save(views/'before.png')
    ffmpeg = shutil.which('ffmpeg') or subprocess.check_output([cfg['compute_python'],'-c',
        'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'],text=True).strip()
    for k,(folder,rec) in cases.items():
        extract(folder/'front_after.mp4',visible[k][frame][2],ffmpeg).crop(crop).save(views/(k+'.png'))
    meta=dict(site=site,frame=frame,crop_xyxy=crop,source_size=[width,height],camera='front_camera',
              states={k:dict(case=cases[k][0].name,visible_pixels=visible[k][frame][0]) for k in ids},
              after_source='decoded native Gaussian front_after.mp4',before_source='registered original front photograph')
    (views/'comparison.json').write_text(json.dumps(meta,indent=2)+'\n')
    return views,meta


def render(run,site,out):
    bundle=run/'series'
    spec=read(bundle/'calibration/survey.json')
    ids=[s['id'] for s in spec['states']]
    folder,match=shared_frame(bundle,site,ids)
    W=1680;gap=18;margin=36;cw=(W-2*margin-2*gap)//3
    H=470+((len(ids)+2)//3)*620
    paper='#f4f3ed';ink='#233c30'
    page=Image.new('RGB',(W,H),paper);draw=ImageDraw.Draw(page)
    def words(x,y,text,size,bold=False,fill=ink):
        draw.text((x,y),text,font=ImageFont.truetype(BOLD if bold else FONT,size,index=2),fill=fill,anchor='lt')
    def picture(path,box,contain=False):
        x,y,w,h=box
        with Image.open(path) as image:
            if contain:
                # Source crops retain all annotation landmarks and all visible geometry.
                im=ImageOps.contain(image,(w,h),Image.Resampling.LANCZOS)
                page.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
            else:page.paste(ImageOps.fit(image,(w,h),Image.Resampling.LANCZOS),(x,y))
    words(margin,25,'同一套窨井　六种状态',50,True)
    words(margin,98,'两张实拍提供井圈和盖面纹理。绝对尺寸、井深与姿态采用设计值。',25)
    bh=174
    for i,(file,title) in enumerate([(bundle/'references/frame.png','井圈原图  ·  图 D'),
                                (bundle/'references/lid.png','盖面原图  ·  图 B'),(folder/'before.png','同一路段  ·  迁入前')]):
        x=margin+i*(cw+gap)
        draw.rectangle((x,153,x+cw,153+bh),fill='#e4e4da')
        picture(file,(x,153,cw,bh),contain=i!=2)
        words(x,339,title,21,True)
    words(margin,381,f"上图为模型斜视，下图为 PandaSet {site} 前视相机第 {match['frame']} 帧的同一区域。",20)
    for i,state in enumerate(spec['states']):
        key=state['id']
        x=margin+i%3*(cw+gap);y=425+i//3*620
        words(x,y,key+'   '+state['label'],27,True)
        if not state.get('absent'):
            dx,dy=state['offset_m']
            tag=(f"平移 {(dx*dx+dy*dy)**.5:.2f} m" if dx or dy else f"{state['angle_deg']}°")
            words(x+cw-160,y+3,tag,23)
        picture(bundle/'views'/(key+'_inspection.jpg'),(x,y+42,cw,338),contain=True)
        picture(folder/(key+'.png'),(x,y+386,cw,219),contain=True)
    if spec.get('credit'):
        credit=spec.get('plate_credit',spec['credit'])
        width=ImageFont.truetype(FONT,16,index=2).getlength(credit)
        words(margin,H-31,credit,min(16,int(16*(W-2*margin)/max(1,width))))
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,quality=94,subsampling=0)
    return match


def road_matrix(run,sites,out):
    bundle=run/'series'
    spec=read(bundle/'calibration/survey.json')
    ids=[s['id'] for s in spec['states']]
    config=read(ROOT/'gen2hole/config/local.json')
    cell=420;left=154;step=148
    page=Image.new('RGB',(left+len(sites)*(cell+10)+24,369+(len(ids)+1)*step),'#f4f3ed')
    draw=ImageDraw.Draw(page)
    def word(x,y,text,size,bold=False):
        draw.text((x,y),text,font=ImageFont.truetype(BOLD if bold else FONT,size,index=2),fill='#233c30',anchor='lt')
    word(26,17,'同一套井盖在不同道路的前视画面',41,True)
    word(26,77,'各列使用各自路段的同一锚点、相机帧和裁切区域。上方方框标出资产在完整相机画面中的位置。',19)
    labels=['迁入前']+[s['label'] for s in spec['states']]
    for i,text in enumerate(labels):word(26,329+i*step+50,text,23,True)
    for j,site in enumerate(sites):
        folder,match=shared_frame(bundle,site,ids)
        x=left+j*(cell+10)
        word(x,122,f"PandaSet {site}   ·   帧 {match['frame']}",23,True)
        with Image.open(Path(config['pandaset_processed'])/site/'images'/f"{match['frame']:03d}_0.jpg") as photo:
            overview=photo.convert('RGB')
            ImageDraw.Draw(overview).rectangle(match['crop_xyxy'],outline='#ffcf5d',width=10)
            thumb=ImageOps.contain(overview,(cell,160),Image.Resampling.LANCZOS)
            page.paste(thumb,(x+(cell-thumb.width)//2,157))
        for i,key in enumerate(['before']+ids):
            with Image.open(folder/(key+'.png')) as im:
                thumb=ImageOps.contain(im,(cell,step-7),Image.Resampling.LANCZOS)
                page.paste(thumb,(x+(cell-thumb.width)//2,329+i*step))
    if spec.get('credit'):
        credit=spec.get('plate_credit',spec['credit'])+' 道路画面来自 PandaSet。'
        width=ImageFont.truetype(FONT,15,index=2).getlength(credit)
        word(26,page.height-29,credit,min(15,int(15*(page.width-52)/max(1,width))))
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,quality=94,subsampling=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--site',default='034')
    p.add_argument('--sites',nargs='+',help='Multiple sites to display in a matrix of matched camera frames')
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.sites:road_matrix(a.run,a.sites,a.out)
    else:print(json.dumps(render(a.run,a.site,a.out),ensure_ascii=False,indent=2))
