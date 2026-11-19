"""Independently verify the actual point/covariance scaling in all thirty scenes."""
import gc
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import re
import subprocess
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from plyfile import PlyData
import torch
from verify import equal_subset

WORK=Path(__file__).parent
ROOT=PROJECT/'outputs/道路病害_多场景迁入'
OUT=ROOT/'双目坑洞_几何放大10倍'
LIB=PROJECT/'outputs/road_hazard_expansion'

def main():
    torch.set_num_threads(4)
    index=json.loads((OUT/'视频与资产索引.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in json.loads((LIB/'asset_manifest.json').read_text(encoding='utf-8'))}
    cache={};reports=[]
    for site in ['034','109','110']:
        original=torch.load(ROOT/'重建场景'/site/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
        for i in range(1,11):
            folder=OUT/'迁入场景'/f'{site}_{i:02d}'
            summary=json.loads((folder/'scene.json').read_text(encoding='utf-8'));assert summary['stereo_geometry_scale']==10
            state=torch.load(folder/'trained_model/iteration_7000.pth',map_location='cpu',weights_only=False)
            assert set(state)==set(original)
            for key in state:
                if key not in ['background','iter']:assert equal_subset(original[key],state[key]),(site,i,key)
            for d in summary['placements']:
                name=d['asset_id'];factor=10 if d['category']=='stereo' else 1
                assert d['size_multiplier']==factor
                if name not in cache:
                    a=PlyData.read(LIB/models[name]['ply'],mmap='r')['vertex'].data
                    cache[name]=(np.stack([a[n] for n in ['x','y','z']],1),np.stack([a[f'scale_{j}'] for j in range(3)],1))
                xyz,logs=cache[name];expected=xyz*factor@np.array(d['basis_columns']).T+np.array(d['center_world'])
                start=d['merged_asset_offset'];n=d['asset_points'];actual=state['background']['xyz'][start:start+n].detach().numpy()
                error=float(np.abs(actual-expected).max());assert error<2e-5,(site,i,name,error)
                scales=state['background']['scaling'][start:start+n].detach().numpy()
                scale_error=float(np.abs(scales-(logs+np.log(factor))).max());assert scale_error<1e-6
                if factor==10:
                    reports.append({'scene':site,'sample':i,'asset_id':name,'point_scale':10,'gaussian_scale':10,
                       'original_width_m':d['original_width_m'],'display_width_m':d['width_m'],
                       'original_length_m':d['original_length_m'],'display_length_m':d['length_m'],
                       'max_position_error_m':error,'max_log_scale_error':scale_error,'actors_sky_unchanged':True})
            del state;gc.collect()
        del original;gc.collect();print('GEOMETRY VERIFIED',site,flush=True)
    assert len(reports)==30
    path=OUT/index['file'];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    result=subprocess.run([ffmpeg,'-hide_banner','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    frames=int(re.findall(r'frame=\s*(\d+)',result.stderr)[-1]);assert frames==2880==index['frames']
    reader=imageio.get_reader(str(path),input_params=['-threads','1'])
    motion=[]
    for j,site in enumerate(['034','109','110']):
        a=reader.get_data(j*96).astype('f4');b=reader.get_data(j*96+48).astype('f4')
        difference=float(np.abs(a[208:694,800:1664]-b[208:694,800:1664]).mean());assert difference>.5
        motion.append({'scene':site,'global_motion_mean_pixel_change':difference})
    reader.close()
    validation={'stereo_assets':10,'scenes':3,'scaled_placements':30,'uniform_geometry_scale':10,
                'point_and_gaussian_scales_verified':True,'original_actor_sky_parameters_preserved':True,
                'video_frames':frames,'video_full_decode':True,'global_motion':motion,'placements':reports}
    (OUT/'放大验证.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    preview=OUT/'00_几何放大10倍_4秒预览.mp4'
    cmd=[ffmpeg,'-y','-loglevel','error','-i',str(path),'-frames:v','96','-an','-c:v','libx264',
         '-preset','veryfast','-crf','20','-threads','4','-movflags','+faststart',str(preview)]
    short=subprocess.run(cmd,capture_output=True,text=True);assert short.returncode==0,short.stderr
    text=['# 双目来源坑洞：三维几何实际放大 10 倍','',
          '[4 秒动态预览](00_几何放大10倍_4秒预览.mp4) · [完整 2 分钟视频]('+index['file']+')','',
          '这次直接改变场景中的坑洞三维几何。10 个双目来源资产的局部 XYZ 坐标和高斯空间尺度均统一放大 10 倍；坑口约由 5～15 厘米扩大为 0.5～1.5 米，深度也同步放大。',
          '这是可视化演示尺寸，不能视作原实拍病害的实测尺寸。原资产、原尺寸版本以及前三条道路背景均保留。','',
          '视频维持五分区：独立资产 360° 环绕、全局迁入前后、局部迁入前后。全局播放连续 40 个原始行车帧，前后同相机同步；每个资产依次展示 034、109、110 三条道路，每段 4 秒，总长 2 分钟。',
          '全局定位框对应当前双目来源坑洞；同组其他四类资产仍以原尺寸保留。','',
          '`放大资产尺寸.json` 记录原尺寸与放大尺寸；`视频与资产索引.json` 记录视频时间段和检查点路径；`放大验证.json` 核对全部 30 个原生场景中的点坐标、高斯尺度、其他资产尺度、车辆与天空参数。',
          '所有放大场景位于 `迁入场景/`，包含可编辑检查点及渲染配置。',
          '图像纹理清晰度仍受原始双目素材和高斯采样密度限制；周边道路保留原前视重建的模糊和少量漂浮伪影。','']
    (OUT/'观看说明.md').write_text('\n'.join(text),encoding='utf-8')
    print('ENLARGED STEREO VERIFIED',flush=True)

if __name__=='__main__':main()
