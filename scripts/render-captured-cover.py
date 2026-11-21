"""Publish a matched photo, orthophoto, held-out projection, GLB and transfer plate."""
import argparse
import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from_file=importlib.util.spec_from_file_location('family',Path(__file__).with_name('render-chamber-family.py'))
family=importlib.util.module_from_spec(from_file)
from_file.loader.exec_module(family)


def render(run,site,report,out):
    bundle=run/'series'
    survey=family.read(bundle/'calibration/survey.json')
    fit=family.read(bundle/'calibration/capture-fit.json')
    verify=family.read(report)
    states=survey['states']
    roads,match=family.shared_frame(bundle,site,[s['id'] for s in states])
    if len(states)!=3:
        raise ValueError('The capture plate uses three state columns')
    page=Image.new('RGB',(1416,1050),'#f2f1eb')
    draw=ImageDraw.Draw(page)
    ink='#172e26'
    def text(x,y,value,size=19,bold=False,color=ink):
        font=ImageFont.truetype(family.BOLD if bold else family.FONT,size,index=2)
        draw.text((x,y),value,font=font,fill=color,anchor='lt')
    def panel(path,x,y,w,h,bg):
        draw.rectangle((x,y,x+w-1,y+h-1),fill=bg)
        with Image.open(path) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(w,h),Image.Resampling.LANCZOS)
            page.paste(thumb,(x+(w-thumb.width)//2,y+(h-thumb.height)//2))
    text(30,19,'从实拍道路重建带修补圈的小型盖板' if 'apron_xy_extent_m' in fit else '从实拍道路重建一套可动井盖',42,True)
    detail=f"盖面直径 {fit['lid_diameter_m']:.3f} m"
    if 'apron_xy_extent_m' in fit:
        w,h=fit['apron_xy_extent_m']
        detail=f"修补圈 {w:.3f} × {h:.3f} m"
    text(31,80,f"PandaSet {fit['sequence']}   ·   金属外缘直径 {fit['frame_diameter_m']:.3f} m   ·   "+detail,22)
    xcols=[30,488,946]
    key=fit['heldout_frames'][0]
    source_images=[bundle/'references/capture.png',bundle/'references/frame.png',run/'reprojection'/(key+'-registered.png')]
    source_names=['前视原图　拟合帧 '+fit['fit_frames'][0].rsplit('_',1)[1],'双帧融合　正射纹理','留出前视帧 '+key.rsplit('_',1)[1]+'　GLB 外圈回投']
    for x,title,img in zip(xcols,source_names,source_images):
        text(x,134,title,20,True)
        panel(img,x,170,440,174,'#dedfd7')
    text(30,358,f"黄线为人工标注，青线为局部配准后的网格。{verify['heldout_lid_returns']} 个留出激光点与盖面网格的高度绝对误差中位数为 {verify['lidar_abs_error_median_m']*1000:.1f} mm。",19)
    for x,state in zip(xcols,states):
        key=state['id']
        text(x,421,key+'   '+state['label'],25,True)
        panel(bundle/'views'/(key+'_inspection.jpg'),x,464,440,250,'#e7e7df')
        panel(roads/(key+'.png'),x,787,440,163,'#dedfd7')
    text(30,735,f"迁入 PandaSet {site}　·　相同世界坐标、前视第 {match['frame']} 帧、相同裁切",22,True)
    designed='、'.join(s['label'] for s in states[1:])
    text(30,972,'实拍记录了落座状态。'+designed+'为派生状态，井深、盖厚和遮挡面采用设计值。',20)
    text(30,1014,'PandaSet · Scale AI, Inc. and Hesai Photonics Technology Co., Ltd. · CC BY 4.0 and PandaSet Dataset Terms.',15)
    page.save(out,quality=94,subsampling=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--site',required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    render(a.run,a.site,a.report,a.out)
