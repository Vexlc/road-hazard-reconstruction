"""Check all-frame driving playback, tracking coverage and stereo close cameras."""
from collections import Counter
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import re
import subprocess
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

WORK=Path(__file__).parent
OUT=PROJECT/'outputs/道路病害_多场景迁入/逐帧行车与双目增强版'

def main():
    index=json.loads((OUT/'视频与资产索引.json').read_text(encoding='utf-8'))
    assert len(index['videos'])==5 and len(index['segments'])==150
    ids=Counter(s['asset_id'] for s in index['segments']);assert len(ids)==50 and set(ids.values())=={3}
    assert Counter(s['scene'] for s in index['segments'])=={'034':50,'109':50,'110':50}
    for s in index['segments']:
        frames=s['global_native_frame_indices']
        assert len(frames)==40 and all(b==a+1 for a,b in zip(frames,frames[1:]))
        assert s['global_source_fps']==10 and s['global_before_after_synchronized']
        assert s['scale_multiplier']==1.
        if s['category']=='stereo':
            assert s['stereo_tracking_zoom']==8 and s['stereo_close_camera']
            assert s['tracking_visible_output_frames']>0
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();checks=[]
    for video in index['videos']:
        path=OUT/video['file']
        decode=subprocess.run([ffmpeg,'-hide_banner','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        assert decode.returncode==0,decode.stderr
        count=int(re.findall(r'frame=\s*(\d+)',decode.stderr)[-1]);assert count==2880==video['frames']
        assert '2560x1440' in decode.stderr and '24 fps' in decode.stderr
        reader=imageio.get_reader(str(path),input_params=['-threads','1'])
        motion=[]
        for site in range(3):
            # Read actual imagery away from frame labels and stereo magnifier overlays.
            a=reader.get_data(site*96).astype('f4');b=reader.get_data(site*96+48).astype('f4')
            before=float(np.abs(a[470:690,1060:1650]-b[470:690,1060:1650]).mean())
            after=float(np.abs(a[470:690,1940:2530]-b[470:690,1940:2530]).mean())
            assert before>.5 and after>.5,(video['category'],site,before,after)
            motion.append({'scene':['034','109','110'][site],'global_before_motion_mean_pixel_change':before,
                           'global_after_motion_mean_pixel_change':after})
        reader.close()
        checks.append({**video,'fully_decoded_frames':count,'global_motion_checks':motion,'bytes':path.stat().st_size})
        print('VERIFIED DYNAMIC',video['category'],count,flush=True)
    details=[]
    for p in sorted((WORK/'composite4/stereo').glob('*/complete.json')):
        d=json.loads(p.read_text());assert d['geometry_scale']==1 and d['matched_before_after_cameras']
        assert d['camera_height_m']<.20 and d['camera_distance_m']<.20
        details.append(d)
    assert len(details)==30
    result={'videos':checks,'unique_assets':50,'segments':150,'global_consecutive_source_frames':40,
            'global_source_fps':10,'global_uses_video_not_stills':True,'stereo_tracking_window_zoom':8,
            'stereo_detail_renders':30,'asset_geometry_scale_preserved':True}
    (OUT/'展示验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 逐帧行车与双目坑洞增强版','',
           '保留五分区：左侧资产环绕、右上全局前后、右下局部前后。每类 10 个资产依次展示三个道路场景，每份视频 2 分钟。','',
           '全局前后已改为同步行车视频，每段播放 40 个连续原始相机帧，原序列为 10 fps；输出封装为 24 fps，通过重复显示原始帧保持原时长。画面显示原始行车帧编号，定位框随相机逐帧更新。',
           '双目坑洞的环绕镜头改为对准坑口，局部检查相机离路面约 10～20 厘米，并使用 1280×720 原生渲染；全局前后各增加 8× 跟随放大窗。视野外的坑洞会明确显示“不在画内”。',
           '全局迁入后保留同组五类资产，逐帧更新的定位框对应当前展示目标。',
           '所有资产仍保留原尺寸、原位置和原深度。放大窗是对全局图像中同一位置的放大，不增加原始像素细节；局部面板通过更近的 3D 相机重新渲染。','',
           '| 类别 | 视频 | 时长 |','|---|---|---:|']
    for v in checks:
        title=v['file'][3:].replace('_逐帧行车与局部对照.mp4','')
        lines.append(f"| {title} | [观看]({v['file']}) | 2:00 |")
    lines+=['','`视频与资产索引.json` 给出资产 ID、道路、视频时间段及连续行车帧范围；`展示验证.json` 记录全帧解码和全局画面实际运动检查。',
            '局部前后使用相同检查轨迹，局部动态物体固定于该处的参考时刻；全局动态物体随行车序列时间播放。背景仍继承原始前视 3DGS 重建的模糊、拉伸及少量漂浮伪影。','']
    (OUT/'观看说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print('DYNAMIC PRESENTATION VERIFIED',flush=True)

if __name__=='__main__':main()
