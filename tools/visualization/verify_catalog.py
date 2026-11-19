import json
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/道路病害_五类4x4视频'
WORK=BASE/'work/visualization'
WORK.mkdir(parents=True, exist_ok=True)
LIB=BASE/'outputs/road_hazard_expansion'
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
index=json.loads((OUT/'资产视频索引.json').read_text())
source=json.loads((LIB/'asset_manifest.json').read_text())
assert len(index['videos'])==5
ids=[r['id'] for r in index['assets']]
assert len(ids)==len(set(ids))==1199
assert set(ids)==set(r['id'] for r in source)
checks=[]
for v in index['videos']:
    path=OUT/v['file']
    # Decode every frame and check the exact expected video properties.
    result=subprocess.run([ffmpeg,'-hide_banner','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '2560x1920' in result.stderr and '24 fps' in result.stderr
    frames=int(re.findall(r'frame=\s*(\d+)',result.stderr)[-1])
    assert frames==v['frames'],(v,frames)
    cat=v['category']
    # Inspect decoded views half a rotation apart, independently of source screenshots.
    pattern=str(WORK/(cat+'_decoded_%02d.jpg'))
    sample=subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(path),'-vf',
        'select=eq(n\\,0)+eq(n\\,48),scale=1280:960','-vsync','vfr','-frames:v','2',pattern],capture_output=True,text=True)
    assert sample.returncode==0,sample.stderr
    a=np.asarray(Image.open(WORK/(cat+'_decoded_01.jpg')),dtype='f4')
    b=np.asarray(Image.open(WORK/(cat+'_decoded_02.jpg')),dtype='f4')
    change=float(np.mean(np.abs(a-b)))
    assert change>.5,(cat,'motion missing',change)
    checks.append(dict(category=cat,assets=v['assets'],frames=frames,duration_s=v['seconds'],
                       decoded_all_frames=True,half_rotation_mean_pixel_change=round(change,3),
                       file_size_bytes=path.stat().st_size))
    print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
validation={'asset_ids':1199,'unique_asset_ids':1199,'missing_ids':0,'extra_ids':0,
            'width':2560,'height':1920,'fps':24,'grid':[4,4],'seconds_per_page':4,'videos':checks}
(OUT/'视频验证记录.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')

lines=['# 道路病害资产：五类 4×4 旋转视频','',
       '覆盖最新扩充库的全部 1,199 条资产记录；共 77 页。每页最多同时展示 16 个资产，持续 4 秒，每个资产完整旋转一周。',
       '视频为 2560×1920、24 fps、H.264 MP4。画面来自实际 3DGS PLY 的新视角渲染。末页不足 16 个时留空，不重复填充。','',
       '| 视频 | 资产数 | 页数 | 时长 |','|---|---:|---:|---:|']
for v in index['videos']:
    lines.append(f"| [{v['title']}]({v['file']}) | {v['assets']} | {v['pages']} | {v['seconds']//60:02d}:{v['seconds']%60:02d} |")
lines += ['', '## 来源说明','',
          '最新扩充库中的 64 个缺盖窨井、128 个掀盖与移位窨井，其结构均为程序化建模；路面、混凝土及铸铁外观使用摄影测量来源的表面材质，不能视为逐个实物的扫描重建。',
          '早期另有扫描高度场井盖，以及两件照片辅助的井口资产；照片辅助资产的井筒、背面与尺度仍由建模补全。这些早期资产不计入本次 1,199 条扩充库，也不混入五个视频。',
          '双目与 RGB-D 组按采集帧计数，相同病害可能在多个帧中出现；RGB-D 组使用估计内参，尺寸与深度仅为近似值。',
          '每格独立归一化取景，屏幕中的大小不能用于比较实物尺寸，请看每格尺寸标签。', '',
          '## 检索','',
          '`资产视频索引.json` 为每个资产记录视频名、页码、行列、起止秒数、完整 ID 和源 PLY 路径。',
          '`视频验证记录.json` 确认资产 ID 全覆盖且不重复，并记录五个视频的完整解码与旋转画面检查结果。','']
(OUT/'观看说明.md').write_text('\n'.join(lines))
print('VERIFIED: 5 videos, all 1199 assets, exact frame counts, visible rotation.',flush=True)
