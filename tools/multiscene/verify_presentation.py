"""Check the complete five-panel videos, coverage, and visible orbit motion."""
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
OUT=PROJECT/'outputs/道路病害_多场景迁入/环绕与迁入前后综合展示'

def main():
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    index=json.loads((OUT/'视频与资产索引.json').read_text(encoding='utf-8'))
    assert len(index['videos'])==5 and len(index['segments'])==150
    expected={n for ids in plan['selected_assets'].values() for n in ids}
    assert Counter(s['asset_id'] for s in index['segments'])==Counter({n:3 for n in expected})
    assert Counter(s['scene'] for s in index['segments'])=={'034':50,'109':50,'110':50}
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();checks=[]
    for video in index['videos']:
        path=OUT/video['file']
        result=subprocess.run([ffmpeg,'-hide_banner','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        count=int(re.findall(r'frame=\s*(\d+)',result.stderr)[-1])
        assert count==video['frames']==2880
        assert '2560x1440' in result.stderr and '24 fps' in result.stderr
        reader=imageio.get_reader(str(path),input_params=['-threads','1'])
        a=reader.get_data(0).astype('f4');b=reader.get_data(48).astype('f4');reader.close()
        orbit_change=float(np.abs(a[312:882,20:780]-b[312:882,20:780]).mean())
        global_change=float(np.abs(a[208:694,800:1664]-a[208:694,1680:2544]).mean())
        local_change=float(np.abs(a[848:1334,800:1664]-a[848:1334,1680:2544]).mean())
        assert orbit_change>.1 and global_change>.05 and local_change>.1,(video['category'],orbit_change,global_change,local_change)
        checks.append({**video,'fully_decoded_frames':count,'bytes':path.stat().st_size,
                       'orbit_half_turn_mean_pixel_change':orbit_change,'global_pair_mean_pixel_change':global_change,
                       'local_pair_mean_pixel_change':local_change,'complete_decode':True})
        print('VERIFIED',video['category'],count,round(orbit_change,2),round(global_change,2),round(local_change,2),flush=True)
    for segment in index['segments']:
        assert segment['matched_global_camera'] and segment['matched_local_camera_sweep']
        assert segment['end_seconds']-segment['start_seconds']==4
        assert segment['full_orbit_degrees']==360
    validation={'videos':checks,'unique_assets':50,'categories':5,'scenes':3,'segments':150,
                'seconds_per_asset_per_scene':4,'five_panels_verified':True,'full_video_decode':True}
    (OUT/'展示验证.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 资产环绕 + 全局迁入前后 + 局部迁入前后','',
           '五类各 10 个资产，每个资产依次展示 PandaSet 034、109、110 三条道路；每个资产在每条道路上展示 4 秒，每份分类视频共 2 分钟。原有资产选择、位置和尺寸保持不变。','',
           '画面左侧为独立资产的完整 360° 环绕；右上两格为全局迁入前后；右下两格为局部迁入前后。全局和局部的前后对比分别使用相同的相机与时间。',
           '全局采用固定行车相机，定位框标出当前资产；迁入后保留原有同组五类资产。局部前后沿相同轨迹同步移动。独立环绕采用资产自身坐标，不代表整条道路的环绕重建。','',
           '| 类别 | 视频 | 资产数 | 道路数 | 时长 |','|---|---|---:|---:|---:|']
    for v in checks:
        title=v['file'][3:].replace('_环绕与迁入前后.mp4','')
        lines.append(f"| {title} | [观看]({v['file']}) | 10 | 3 | 2:00 |")
    lines+=['','`视频与资产索引.json` 记录每个资产和道路在视频中的起止时间；`展示验证.json` 记录全帧解码、覆盖数量和面板变化检查。',
            '已核对 50 个资产、150 个资产与场景组合全部覆盖，五份视频均通过完整解码。',
            '近景背景仍保留原始前视 3DGS 重建的模糊与拉伸。RGB-D 尺寸和深度采用估计内参；程序化资产的尺寸为建模设定。','']
    (OUT/'观看说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print('FIVE-PANEL PRESENTATION VERIFIED',flush=True)

if __name__=='__main__':main()
