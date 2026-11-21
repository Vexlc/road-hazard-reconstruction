"""Render the atlas overview plates from actual local model, camera and height-field files."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "tools/asset_atlas"
OUT = ROOT / "docs/figures"
PAPER, INK, SOFT, LINE = "#f4f3ed", "#233c30", "#4e6457", "#c4c9bd"
REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def label(draw, xy, content, size=22, bold=False, fill=INK):
    draw.text(xy,content,font=ImageFont.truetype(BOLD if bold else REGULAR,size,index=2),fill=fill,anchor="lt")


def tile(sheet, path, box, trim_background=False):
    with Image.open(path) as opened:
        im = opened.convert("RGB")
    if trim_background:
        data = np.array(im,dtype=np.int16)
        mask = np.max(abs(data-data[0,0]),axis=2)>20
        y,x = np.where(mask)
        if len(x):
            pad = max(np.ptp(x),np.ptp(y))*.14
            im = im.crop((max(0,int(x.min()-pad)),max(0,int(y.min()-pad)),
                          min(im.width,int(x.max()+pad+1)),min(im.height,int(y.max()+pad+1))))
    left,top,width,height=box
    # Preserve the entire object and pixel aspect ratio, including vertical lids.
    dst=Image.new("RGB",(width,height),im.getpixel((0,0)))
    thumb=ImageOps.contain(im,(width,height),Image.Resampling.LANCZOS)
    dst.paste(thumb,((width-thumb.width)//2,(height-thumb.height)//2))
    sheet.paste(dst,(left,top))


def profile(draw, row, x, y, w, h):
    if row["method"]=="生成 RGB":
        geo=ROOT / "gen2hole/published/道路病害_传感器与几何增强/geometry" / (row["id"]+".npz")
    elif row["id"]=="P90":
        geo=ROOT / "runs/rgb-box-P90/sensors/geometry/P90.npz"
    else:
        geo=None
    with np.load(geo or ATLAS/row["surface"]) as data:
        if geo:
            middle=int(np.abs(data["y_m"]).argmin())
            top=data["top_z_m"][middle]; sub=data["substrate_z_m"][middle]; xs=data["x_m"]
        else:
            field=data["height"]
            top=field[field.shape[0]//2];sub=top
            xs=np.linspace(-float(data["width"])/2,float(data["width"])/2,len(top))
    low=min(float(np.nanmin(top)),float(np.nanmin(sub)),0.)
    high=max(float(np.nanmax(top)),0.)
    dx=(w-28)/(float(xs[-1])-float(xs[0])); dz=(h-54)/max(high-low,.005)
    def xy(a,b):
        return (x+14+(float(a)-float(xs[0]))*dx,y+12+(high-float(b))*dz)
    draw.line([xy(xs[0],0),xy(xs[-1],0)],fill="#849488",width=1)
    def trace(values,color,width):
        prev=None
        for a,b in zip(xs,values):
            cur=xy(a,b) if np.isfinite(b) else None
            if prev and cur: draw.line([prev,cur],fill=color,width=width)
            prev=cur
    if not np.array_equal(top,sub,equal_nan=True):trace(sub,"#be8743",3)
    trace(top,"#2d6e5c",3)
    label(draw,(x+12,y+h-24),f"Z   {low:+.2f} … {high:+.2f} m",16,fill=SOFT)


def plate(rows):
    picks=[("P90","实拍单目","P90 · 粗框引导"),("rgbd_pothole_20250227_135438","实拍 RGB-D","与 P90 同一帧"),
           ("stereo_pothole_fan_1_03","实拍双目","Fan · 1 / 03"),("P02_wet_wheel_track","生成 RGB","P02 · 积水车辙"),
           ("M11","生成 RGB","M11 · 矩形敞井"),("A13","生成 RGB","A13 · 侧铰掀盖")]
    sheet=Image.new("RGB",(1660,190+len(picks)*178+55),PAPER);d=ImageDraw.Draw(sheet)
    label(d,(38,28),"同一件资产，从参考图到道路",48,True)
    label(d,(40,97),"RGB、RGB-D 与双目实例  /  局部剖面通过资产中心，Z = 0 为路面。",24,fill=SOFT)
    for x,title in zip([38,249,567,907,1235],["来源 / 实例","参考 RGB","重建模型","中心横剖面（米）","PandaSet 迁入"]):
        label(d,(x,151),title,20,True)
    for i,(key,method,caption) in enumerate(picks):
        r=rows[key];y=190+i*178
        d.line((38,y-7,1622,y-7),fill=LINE,width=1)
        label(d,(38,y+20),method,24,True);label(d,(38,y+60),caption,18)
        scene=r["scenes"][0]
        label(d,(38,y+98),"路段 "+scene["label"].split(" · ")[0],17,fill=SOFT)
        tile(sheet,ATLAS/r["reference"],(248,y,296,161))
        tile(sheet,ATLAS/r["preview"],(565,y,316,161),r["method"] in ("实拍双目","实拍 RGB-D"))
        profile(d,r,896,y,308,161)
        tile(sheet,ATLAS/scene["after"],(1232,y,390,161))
    label(d,(38,sheet.height-34),"绿色为可见表面，金色为移除盖板或水面后的已知基底。各行纵轴独立缩放。单目实例采用设计坑深。",18,fill=SOFT)
    sheet.save(OUT/"reconstruction-evidence.jpg",quality=91,subsampling=0)


def diversity(rows):
    picks=[("坑洞",["P04","P07","P16","P25","P37"]),
           ("缺盖井",["M01_cast_iron_circular","M03_brick_lined","M11","M24","M30"]),
           ("异常井盖",["A13","A03_tilted_square","A20","A29","A30"])]
    sheet=Image.new("RGB",(1632,1070),PAPER);d=ImageDraw.Draw(sheet)
    label(d,(36,32),"坑洞、井口与井盖的形态差异",48,True)
    label(d,(38,104),"15 件带纹理 GLB 的斜视渲染  /  选自 109 件生成 RGB 资产",23,fill=SOFT)
    for i,(kind,keys) in enumerate(picks):
        y=163+i*289
        label(d,(38,y),kind,26,True); d.line((175,y+15,1594,y+15),fill=LINE,width=2)
        for j,key in enumerate(keys):
            r=rows[key];x=38+j*314
            tile(sheet,ATLAS/r["preview"],(x,y+43,301,203))
            label(d,(x,y+255),key[:3]+"   "+r["label"],18)
    label(d,(38,1030),"图中尺度随取景变化。可在离线图谱中按编号查看米制尺寸、反向视角与同帧道路对照。",17,fill=SOFT)
    sheet.save(OUT/"asset-diversity.jpg",quality=91,subsampling=0)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((ATLAS/"catalog.js").read_text().removeprefix("window.roadAssets=").rstrip(";\n"))
    rows={r["id"]:r for r in catalog}
    plate(rows)
    diversity(rows)
