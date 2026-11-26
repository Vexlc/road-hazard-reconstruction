"""Make a source, Gaussian and driving-camera contact sheet for M544."""
import argparse
from importlib import import_module
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

style = import_module("review-laser-fractures")


def cropped_road(view):
    photo = Image.open(view/"after.png").convert("RGB")
    with np.load(view/"rays.npz") as p:
        y,x = np.where(p["part"]>=163)
    cx,cy = (x.min()+x.max())//2,(y.min()+y.max())//2
    width = int(max(np.ptp(x)*2.0, 100))
    height = int(width*.56)
    left = int(np.clip(cx-width//2,0,photo.width-width))
    top = int(np.clip(cy-height//2,0,photo.height-height))
    return photo.crop((left,top,left+width,top+height))


def render(photo, views, dest):
    sheet=Image.new("RGB",(1480,925),style.PAPER)
    draw=ImageDraw.Draw(sheet)
    def caption(x,y,text,size,bold=False):
        draw.text((x,y),text,font=style.font(size,bold),fill=style.INK)
    caption(38,18,"M544　锈穿的菱纹钢井盖",36,True)
    caption(40,77,"同一块钢板，从现场近照到车辆距离。孔位取自照片，板厚与井深按估值建模。",21)
    panels=[
        (Image.open(photo),"现场原照","Lusi Lindwurm  ·  CC BY-SA 4.0"),
        (Image.open(views/"gaussians/normal.jpg"),"高斯模型 · 俯视","孔缘沿原照描出，外围井框为补建"),
        (Image.open(views/"gaussians/near.jpg"),"高斯模型 · 斜视","近处孔缘可见钢板厚度和内侧锈色"),
        (Image.open(views/"front/M544/drive.jpg"),"驾驶视野 · 前摄","PandaSet 109，第 15 帧，路面合成"),
        (cropped_road(views/"front/M544"),"前摄局部 · 放大","同帧的车载像素，放大仅供辨认位置"),
        (cropped_road(views/"back/M544"),"后摄局部 · 放大","第 28 帧，同一位置经过后回望"),
    ]
    for i,(im,title,sub) in enumerate(panels):
        x,y=38+i%3*476,125+i//3*384
        caption(x,y,title,23,True)
        style.inset(sheet,im,(x,y+43,452,270))
        caption(x,y+323,sub,17)
    sheet.save(dest,quality=94,subsampling=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("photo","views","dest"):p.add_argument(n,type=Path)
    render(**vars(p.parse_args()))
