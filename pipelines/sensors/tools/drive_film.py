from presentation import *
from camera_geometry import camera_hits,foreground,track
from render_pairs import gather,render
from road_geometry import load_inventory
import torch,imageio.v2 as imageio

OUT=ROOT/'双向车载与雷达.mp4'


def roi(points,pose,K,size):
    rects=[]
    for w in points:
        cam=(w-pose[:3,3])@pose[:3,:3];g=cam[:,2]>.12
        if not g.any():continue
        xy=cam[g,:2]/cam[g,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        lo,hi=xy.min(0),xy.max(0)
        if np.any(hi<0) or np.any(lo>size):continue
        grow=np.maximum((hi-lo)*.80,12)
        lo=np.floor(lo-grow).clip([0,0],size).astype(int);hi=np.ceil(hi+grow).clip([0,0],size).astype(int)
        if np.all(hi>lo):rects.append([*lo,*hi])
    if not rects:return None
    v=np.asarray(rects);return [int(v[:,0].min()),int(v[:,1].min()),int(v[:,2].max()),int(v[:,3].max())]


def sensor_view(raw,processed,site,cam,frame,points,models,splats,num):
    _,poses,K,size=track(str(raw),site,cam);pose=poses[frame]
    image=Image.open(raw/site/'camera'/cam/f'{frame:02d}.jpg').convert('RGB')
    box=roi(points,pose,K,size)
    if not box:return image,None,None,0
    depth,part,inst,norm=camera_hits(pose,K,box,models)
    dynamic,static,ego=foreground(str(raw),str(processed),ROOT,site,cam,frame,box,depth);protected=dynamic|static|ego
    good=(part>1)&(inst==num)&~protected
    if not ((part>1)&~protected).any():return image,None,None,0
    src=np.asarray(image.crop(box))
    composite,_,_,_=render(splats,pose,K,box,src,depth,part,inst,protected,models,site,cam)
    image.paste(Image.fromarray(composite),tuple(box[:2]))
    if good.sum()<8:return image,None,None,0
    yy,xx=np.where(good);bounds=[int(xx.min()+box[0]),int(yy.min()+box[1]),int(xx.max()+1+box[0]),int(yy.max()+1+box[1])]
    return image,image.crop(crop_box(bounds,size)),bounds,int(good.sum())


def marked(frame,bounds,dim):
    pane=frame.resize(dim,Image.Resampling.LANCZOS)
    if bounds:
        x,y,r,b=[int(round(v*(dim[0]/frame.width))) for v in bounds]
        dr=ImageDraw.Draw(pane);length=max(4,min(14,(r-x)//3,(b-y)//2));x-=3;y-=3;r+=3;b+=3
        for xx,sx in [(x,1),(r,-1)]:
            for yy,sy in [(y,1),(b,-1)]:dr.line((xx+sx*length,yy,xx,yy,xx,yy+sy*length),fill='#ffc868',width=2)
    return pane


@torch.no_grad()
def main(raw,processed):
    assets,scenes,_=load_inventory(ROOT.parent)
    episodes=[('034_13','A13','侧铰掀盖',22,69),('110_30','A30','翻起的方井盖',0,29)]
    frames=[]
    with imageio.get_writer(str(OUT),format='FFMPEG',fps=10,codec='libx264',quality=8,macro_block_size=2,
                            output_params=['-threads','3','-movflags','+faststart']) as video:
        for case,aid,label,start,end in episodes:
            file,info=scenes[case];site=info['site'];splats,models,pts=gather(file.parent/'trained_model/iteration_7000.pth',info['placements'],assets)
            points=[v@np.asarray(p['basis_columns']).T+p['center_world'] for v,(p,_,_) in zip(pts,models)]
            n=next(i for i,p in enumerate(info['placements'],1) if p['asset_id']==aid)
            for frame in range(start,end+1):
                result=[sensor_view(raw,processed,site,cam,frame,points,models,splats,n) for cam in ('front_camera','back_camera')]
                im=Image.new('RGB',(1440,808),PAPER);dr=ImageDraw.Draw(im)
                dr.text((26,12),label+'  '+aid,font=font(33,True),fill=INK)
                dr.text((1083,21),'路段 '+site+'    源帧 %03d'%frame,font=font(20),fill=MUTED)
                size=(682,384)
                for i,(full,crop,bounds,px) in enumerate(result):
                    x=28+i*706;im.paste(marked(full,bounds,size),(x,83))
                    dr.text((x,53),['前视相机','后视相机'][i],font=font(16,True),fill=INK)
                best=int(result[1][3]>result[0][3]);crop=result[best][1]
                dr.text((28,482),'病害局部',font=font(19,True),fill=INK)
                if crop:
                    thumb=ImageOps.contain(crop,(465,239),method=Image.Resampling.LANCZOS)
                    im.paste(thumb,(28,519));dr.text((28,768),['前视','后视'][best]+' · 同帧局部放大',font=font(15),fill=MUTED)
                else:
                    dr.rounded_rectangle((28,519,468,757),radius=5,fill='#e8eae1')
                    dr.text((110,622),'当前双向画面中不可见',font=font(17),fill=MUTED)
                dr.text((510,482),'同编号扫描 · 激光回波侧视',font=font(19,True),fill=INK)
                chart,count=cloud_section(case,aid,frame,574,239);im.paste(chart,(510,519))
                dr.text((510,768),'纵轴为局部高度，单位米。',font=font(15),fill=MUTED)
                dr.text((1115,532),str(count),font=font(43,True),fill=INK);dr.text((1115,600),'束回波命中本资产',font=font(17),fill=MUTED)
                for j,(color,lbl) in enumerate([(GREY,'原始扫描'),(CYAN,'坑壁或井壁'),(GOLD,'井沿'),(ORANGE,'盖板或铰链')]):
                    dr.rectangle((1115,650+j*24,1134,654+j*24),fill=color)
                    dr.text((1146,640+j*24),lbl,font=font(14),fill=MUTED)
                video.append_data(np.asarray(im))
                frames.append(dict(case=case,asset=aid,source_frame=frame,front_target_pixels=result[0][3],rear_target_pixels=result[1][3],lidar_hits=count))
            print('film',case,len(frames),flush=True)
            del splats,models,pts;torch.cuda.empty_cache()
    (ROOT/'checks/film_frames.json').write_text(json.dumps(frames,indent=1)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--processed',type=Path,required=True)
    args=p.parse_args()
    main(args.raw,args.processed)
