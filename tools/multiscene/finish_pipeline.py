"""Finish scene insertion, presentation and validation after training ends."""
import json
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import shutil
import subprocess
import time

from prepare_train import PROJECT,WORK,OUT,REPO,PY,ENV

ENV=dict(ENV,ROAD_BATCH_WORK=str(WORK),PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
ENV.pop('ROAD_ONLY_BATCHES',None)

def run(name,args,cwd=REPO):
    print('START',name,flush=True)
    (WORK/'finish_progress.json').write_text(json.dumps({'stage':name,'started_unix':time.time()},indent=2))
    with (WORK/(name+'.log')).open('w',encoding='utf-8') as f:
        p=subprocess.run([str(PY),*map(str,args)],cwd=cwd,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    print('END',name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(name+' failed; see log')

def main():
    missing=[s for s in ['034','109','110'] if not (OUT/'重建场景'/s/'trained_model/iteration_7000.pth').is_file()]
    if missing:raise SystemExit('Restore runtime packages or finish training first: '+','.join(missing))
    for site in ['034','109','110']:
        run('render_'+site,[WORK/'render_scenes.py','--config',WORK/f'pandaset_{site}.yaml','mode','trajectory'])
        run('reload_'+site,[WORK/'reload_check.py','--config',OUT/'迁入场景'/f'{site}_01/render_config.yaml','mode','trajectory'])
    run('present',[WORK/'present.py'],PROJECT)
    run('verify',[WORK/'verify.py'],PROJECT)
    plan=json.loads((WORK/'plan.json').read_text(encoding='utf-8'))
    validation=json.loads((OUT/'全量验证.json').read_text(encoding='utf-8'))
    lines=['# 50 个随机道路病害资产 × 3 个新增重建场景','',
           '新增 PandaSet 034、109、110 三个独立动态 3DGS 重建场景。加上原有 055、056，目前共有 5 个基础场景。本次迁入使用三个新场景。',
           '固定随机种子 20261119，在双目、RGB-D、建模坑洞、缺盖窨井、掀盖与移位窨井五类中各均匀随机抽取 10 条资产记录，不放回抽样。相同的 50 个资产分别迁入三个新场景，共 150 次原尺度放置。每个场景分成 10 个版本，每个版本五类各放 1 个。','',
           '## 直接查看','',
           '- [三种道路：原图与重建对照](三种道路_原图与重建对照.mp4)：8 秒，上为源图，下为实际重建。',
           '- [三种道路：迁入前后行车对照](三种道路_迁入前后.mp4)：40 秒，上为迁入前，下为迁入后，依次覆盖 10 组。编号 1–5 分别为双目、RGB-D、建模坑洞、缺盖窨井、掀盖与移位窨井。编号为位置提示。',
           '- 五份 4×4 总览：每份 9 秒，三条道路各一页，每页同时展示本类 10 个资产，其余格留空。',
           '- 五份逐件近景：每份 30 秒，每个资产 3 秒，三个道路场景并排展示同一资产。','',
           '| 类别 | 4×4 总览（9秒） | 三道路逐件对照（30秒） | 唯一资产 | 放置次数 |','|---|---|---|---:|---:|']
    for v in validation['videos']:
        if v.get('kind')=='grid_overview':
            detail=next(x for x in validation['videos'] if x.get('category')==v['category'] and x.get('kind')!='grid_overview')
            title=v['file'][3:].replace('_4x4迁入总览.mp4','')
            lines.append(f"| {title} | [4×4 总览]({v['file']}) | [三道路逐件对照]({detail['file']}) | 10 | 30 |")
    lines+=['','## 三条道路与重建','',
            '道路名称描述源图中的外观。034 为浅色分块接缝路面；109 为深色均匀路面；110 使用片段前段的灰色裂缝封补区域，放置点分布于道路上的两个相邻车道。',
            '每个新场景均使用全部 80 帧前视相机、960×540 分辨率，从 LiDAR 初始化开始独立训练 7,000 轮；视频导出为 1280×720。未运行扩散蒸馏。','',
            '| 场景 | 路面外观 | PSNR (dB) | SSIM | LPIPS |','|---|---|---:|---:|---:|']
    for scene in validation['scenes']:
        site=scene['scene'];m=scene['metrics']['mean']
        lines.append(f"| {site} | {plan['scene_titles'][site]} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} |")
    lines+=['','指标使用 9 个留出帧，在原生 960×540 分辨率渲染计算，衡量同片段内插值。初始化使用整段的 LiDAR 和颜色信息。',
            '近景中非资产区域仍可能出现前视单相机重建的模糊与拉伸。实拍资产保留采集光照，场景适配使用曝光调整和周边路面颜色烘焙，未进行完整物理重光照。','',
            '## 迁入与验证','',
            '使用 LiDAR 拟合局部道路平面，按其法线与世界坐标放置；保留源资产米制尺寸，切除覆盖病害的旧道路高斯，再补充与宿主道路一致的过渡高斯。资产、道路和原有动态物体通过同一次 3DGS 渲染生成。',
            '三个场景均完整覆盖 50 个选中资产。逐个读取全部 30 个场景检查点，核对 150 处资产点坐标、尺度、有限参数，以及原有车辆、轨迹、天空参数保持一致。每个新场景还重新加载一个原生迁入检查点，复现五类近景。12 个汇总视频均完成全帧解码校验。','',
            '## 项目文件','',
            '- `重建场景/<场景ID>/trained_model/iteration_7000.pth`：三份新重建检查点。',
            '- `迁入场景/<场景ID>_<组号>/`：30 个原生场景版本，含检查点、渲染配置、近景、行车视频和放置参数；每个场景的第 01 组另附完整场景 PLY 快照。',
            '- `随机抽样与迁入计划.json`：随机种子、50 个 ID、三个场景的放置点与分组。',
            '- `迁入资产检索.json`：按资产 ID 找到视频时间、世界坐标与检查点。',
            '- `全量验证.json`：重建指标、位置比对、参数完整性与视频解码结果。','',
            '换机后先恢复运行包，再运行 scripts/multiscene-configs.py 生成当前克隆目录的配置；相机、标定和实例信息位于 tools/multiscene/data/processed。',
            '双目与 RGB-D 资产按来源帧计数，同一病害可能重复出现；RGB-D 尺度采用估计内参。程序化资产的尺寸是建模设定。','']
    (OUT/'观看与项目说明.md').write_text('\n'.join(lines),encoding='utf-8')
    for site in ['034','109','110']:
        shutil.copy2(WORK/f'pandaset_{site}.yaml',OUT/'重建场景'/site/'训练配置.yaml')
    (WORK/'complete.json').write_text(json.dumps({'complete':True,'output':str(OUT)},ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL WORK COMPLETE',flush=True)

if __name__=='__main__':main()
