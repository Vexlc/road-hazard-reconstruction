"""Prepare a driving film, a view-by-view contact book, and an annotated display."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile,ZIP_STORED

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG="#131b24"
FG="#f2f6fa"
MUTE="#afc5d0"
ACCENT="#92d8d8"
LABELS={"stereo":"双目坑洞","rgbd":"RGB-D 坑洞","pothole":"建模坑洞","missing":"缺盖窨井","raised":"掀盖／移位井盖"}

def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT,n)

def fit(path,size):
    with Image.open(path) as im:
        return ImageOps.fit(im.convert("RGB"),size,method=Image.Resampling.LANCZOS)

def write_video(path,fps):
    return imageio.get_writer(path,fps=fps,codec="libx264",quality=8,macro_block_size=2,
                             ffmpeg_log_level="error",output_params=["-movflags","+faststart"])

def title_card(line, sub):
    plate=Image.new("RGB",(1600,964),BG)
    draw=ImageDraw.Draw(plate)
    draw.text((29,7),line,font=font(30,True),fill=FG)
    draw.text((29,43),sub,font=font(15),fill=ACCENT)
    return plate

def movie(root, scenes):
    showcase=[c for c in scenes if c["showcase"]]
    filename=root/"多资产同场行车与近景.mp4"
    with write_video(filename,24) as out:
        for batch in showcase:
            scene=root/("scene_"+batch["id"])
            header=f'PandaSet {batch["site"]} · 场景 {batch["id"]} · {len(batch["assets"])} 个资产同场'
            card=title_card(header,"原始行车轨迹 · 1600 × 900 场景渲染")
            with imageio.get_reader(scene/"drive.mp4") as reader:
                for f,frame in enumerate(reader):
                    x=card.copy()
                    x.paste(Image.fromarray(frame),(0,64))
                    repeats=(f+1)*24//10-f*24//10
                    payload=np.asarray(x)
                    for _ in range(repeats):
                        out.append_data(payload)
            with imageio.get_reader(scene/"inspect.mp4") as reader:
                for f,frame in enumerate(reader):
                    i=f//40
                    a=batch["placements"][i]
                    sub=f'{i+1:02d}  {LABELS[a["category"]]}  ·  {a["width_m"]*100:.0f}×{a["length_m"]*100:.0f} cm  ·  同场景近景   {a["asset_id"]}'
                    x=title_card(header,sub)
                    x.paste(Image.fromarray(frame).resize((1600,900),Image.Resampling.BILINEAR),(0,64))
                    out.append_data(np.asarray(x))
    frames,_=imageio_ffmpeg.count_frames_and_secs(str(filename))
    assert frames==len(showcase)*(120+8*40)
    print("SHOWCASE MOVIE",frames,flush=True)

def scene_plate(root,batch):
    folder=root/("scene_"+batch["id"])
    canvas=Image.new("RGB",(1920,1080),BG)
    d=ImageDraw.Draw(canvas)
    d.text((23,10),f'场景 {batch["id"]}   ·   PandaSet {batch["site"]}   ·   {len(batch["assets"])} 个资产同场',font=font(35,True),fill=FG)
    d.text((23,58),"上方两格为行车轨迹   ·   下方为同一场景内的逐处近景",font=font(20),fill=ACCENT)
    for i,f in enumerate((37,44)):
        x=24+944*i
        with Image.open(folder/f"drive_{f}.jpg") as im:
            img=im.crop((0,145,1600,850))
            canvas.paste(ImageOps.fit(img,(928,395)),(x,92))
        d.text((x+15,107),f"F{f}",font=font(21,True),fill=FG,stroke_width=2,stroke_fill="#151c26")
    for i,a in enumerate(batch["placements"]):
        x=24+(i%4)*472
        y=504+(i//4)*285
        canvas.paste(fit(folder/a["preview_file"],(456,249)),(x,y))
        d.text((x+4,y+252),f'{i+1:02d}  {LABELS[a["category"]]}  ·  {a["width_m"]*100:.0f}×{a["length_m"]*100:.0f} cm',
               font=font(17,True),fill=FG)
    return canvas

def detailed_catalog(root,scenes,work):
    sheet_dir=work/"sheets"
    sheet_dir.mkdir(exist_ok=True)
    lookup={}
    for batch in scenes:
        plate=scene_plate(root,batch)
        plate.save(sheet_dir/(batch["id"]+".jpg"),quality=91)
        for i,a in enumerate(batch["placements"],1):
            lookup[a["asset_id"]]={"contact_sheet":batch["id"]+".jpg","tile":i,"scene":batch["id"],
                                   "pandaset":batch["site"],"category":a["category"]}
    assert len(lookup)==sum(len(c["assets"]) for c in scenes)
    with ZipFile(root/"150组同场近景图册.zip","w",ZIP_STORED) as book:
        for batch in scenes:
            path=sheet_dir/(batch["id"]+".jpg")
            book.write(path,path.name)
        book.writestr("按资产ID检索.json",json.dumps(lookup,ensure_ascii=False,indent=2))
    with ZipFile(root/"150组同场近景图册.zip") as book:
        assert len(book.namelist())==len(scenes)+1 and book.testzip() is None
    out=root/"全库150组场景巡览.mp4"
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,"-y","-v","error","-framerate","2/3","-start_number","1","-i",str(sheet_dir/"%03d.jpg"),
                    "-r","12","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p",
                    "-movflags","+faststart",str(out)],check=True)
    frames,_=imageio_ffmpeg.count_frames_and_secs(str(out))
    assert frames==len(scenes)*18,(frames,len(scenes))
    (root/"asset_scene_lookup.json").write_text(json.dumps(lookup,ensure_ascii=False,indent=2)+"\n")
    print("CATALOG",len(scenes),len(lookup),flush=True)

def hero(root,scenes):
    batch=scenes[0]
    compare=Image.new("RGB",(1800,1100),BG)
    d=ImageDraw.Draw(compare)
    d.text((28,17),"八个道路病害同时放进同一段街道",font=font(42,True),fill=FG)
    d.text((32,76),"PandaSet 055 · 原始相机第 44 帧",font=font(21),fill=ACCENT)
    for i,(label,path) in enumerate([("原场景",root/"original_055_44.jpg"),
                                     ("插入后",root/"scene_001"/"drive_44.jpg")]):
        pos=(30+i*880,120)
        compare.paste(fit(path,(860,484)),pos)
        d.text((pos[0]+18,138),label,font=font(29,True),fill=FG,stroke_width=2,stroke_fill="#10151a")
    for i,index in enumerate((1,2,3,5)):
        p=batch["placements"][index-1]
        x=29+i*439
        compare.paste(fit(root/"scene_001"/p["preview_file"],(422,244)),(x,666))
        label={1:"方形缺盖 · 井壁和踏步",2:"实拍坑洞 · 保留拍摄尺度",3:"圆盖掀开 · 厚度和井口",
               5:"盖板移到井旁 · 矩形井口"}[index]
        d.text((x+2,922),label,font=font(22,True),fill=FG)
    d.text((31,1010),"1,199 个资产 · 150 组场景 · 每组最多 8 个",font=font(25),fill=ACCENT)
    compare.save(root/"多资产同场前后对照.jpg",quality=95,subsampling=0)

def main(root,work):
    scenes=[json.loads(p.read_text()) for p in sorted(root.glob("scene_*/scene.json"))]
    assert len(scenes)==150 and sum(len(x["assets"]) for x in scenes)==1199
    part=os.environ.get("ROAD_PRESENT_PART")
    if part in (None,"hero"):
        hero(root,scenes)
    if part in (None,"movie"):
        movie(root,scenes)
    if part in (None,"catalog"):
        detailed_catalog(root,scenes,work)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("work",type=Path)
    a=p.parse_args()
    main(a.root,a.work)
