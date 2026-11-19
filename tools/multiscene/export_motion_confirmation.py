"""Export unambiguous global-only motion and compatible, uniquely named videos."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

WORK=Path(__file__).parent
ROOT=PROJECT/'outputs/道路病害_多场景迁入'
OUT=ROOT/'全局动态确认版'
LOG=WORK/'global_motion_confirm'
FF=imageio_ffmpeg.get_ffmpeg_exe()
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

SOURCES=[('01_双目坑洞',ROOT/'双目坑洞_几何放大10倍/双目来源坑洞_几何放大10倍_逐帧前后对照.mp4'),
 ('02_RGB-D坑洞',ROOT/'逐帧行车与双目增强版/02_RGB-D实拍坑洞_逐帧行车与局部对照.mp4'),
 ('03_建模坑洞',ROOT/'逐帧行车与双目增强版/03_建模坑洞_逐帧行车与局部对照.mp4'),
 ('04_缺盖窨井',ROOT/'逐帧行车与双目增强版/04_缺盖窨井_逐帧行车与局部对照.mp4'),
 ('05_异常井盖',ROOT/'逐帧行车与双目增强版/05_掀盖与移位窨井_逐帧行车与局部对照.mp4')]

def transcode(item):
    name,source=item;dest=OUT/(name+'_全局动态综合版_1080p.mp4')
    cmd=[FF,'-y','-loglevel','error','-i',str(source),'-vf','scale=1920:1080:flags=lanczos',
         '-an','-c:v','libx264','-preset','veryfast','-crf','19','-profile:v','baseline','-level:v','4.1',
         '-pix_fmt','yuv420p','-bf','0','-g','24','-threads','4','-movflags','+faststart',str(dest)]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    print('COMPATIBLE VIDEO',name,flush=True)
    return {'name':name,'file':dest.name,'source':str(source),'frames':2880,'fps':24,'seconds':120,'resolution':[1920,1080]}

def global_preview():
    source=SOURCES[0][1]
    dest=OUT/'00_只看全局前后_连续8秒.mp4'
    # Take the actual two global panels directly out of the final combined video.
    cmd=[FF,'-y','-loglevel','error','-i',str(source),'-t','8','-vf','crop=1744:486:800:208,scale=1920:536:flags=lanczos',
         '-an','-c:v','libx264','-preset','veryfast','-crf','19','-profile:v','baseline','-level:v','4.1',
         '-pix_fmt','yuv420p','-bf','0','-g','24','-threads','4','-movflags','+faststart',str(dest)]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    # A looping GIF makes the same motion observable without relying on a video player.
    cmd=[FF,'-loglevel','error','-i',str(source),'-t','4','-vf','fps=10,crop=1744:486:800:208,scale=960:268:flags=lanczos',
         '-f','rawvideo','-pix_fmt','rgb24','-threads','2','-']
    raw=subprocess.run(cmd,capture_output=True);assert raw.returncode==0,raw.stderr
    count=len(raw.stdout)//(960*268*3);assert count==40
    font=ImageFont.truetype(FONT,25);small=ImageFont.truetype(FONT,17)
    frames=[]
    for i in range(count):
        picture=Image.frombytes('RGB',(960,268),raw.stdout[i*960*268*3:(i+1)*960*268*3])
        canvas=Image.new('RGB',(960,334),'#101923');canvas.paste(picture,(0,40));draw=ImageDraw.Draw(canvas)
        draw.text((16,5),'全局迁入前',font=font,fill='#b2d5ea');draw.text((496,5),'全局迁入后',font=font,fill='#ffd09a')
        draw.text((16,311),f'连续行车 / 原始帧 {20+i:02d} / 10 fps / 循环播放',font=small,fill='#b7d0df')
        frames.append(canvas)
    frames[0].save(OUT/'00_全局前后_循环动态图.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False,disposal=2)
    print('GLOBAL-ONLY MOTION READY',flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    global_preview()
    with ThreadPoolExecutor(max_workers=3) as pool:videos=list(pool.map(transcode,SOURCES))
    (OUT/'播放索引.json').write_text(json.dumps({'global_only':'00_只看全局前后_连续8秒.mp4',
       'looping_preview':'00_全局前后_循环动态图.gif','videos':videos},ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL MOTION EXPORTS COMPLETE',flush=True)

if __name__=='__main__':main()
