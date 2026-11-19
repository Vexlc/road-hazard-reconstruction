"""Audit actual global image content in every segment, excluding frame captions."""
import hashlib
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import subprocess
import numpy as np
from PIL import Image,ImageChops,ImageStat
import imageio_ffmpeg

WORK=Path(__file__).parent
OUT=PROJECT/'outputs/道路病害_多场景迁入/全局动态确认版'
FF=imageio_ffmpeg.get_ffmpeg_exe()

def read_exact(stream,size):
    chunks=[];remaining=size
    while remaining:
        data=stream.read(remaining)
        if not data:break
        chunks.append(data);remaining-=len(data)
    return b''.join(chunks)

def main():
    index=json.loads((OUT/'播放索引.json').read_text(encoding='utf-8'));reports=[]
    for video in index['videos']:
        path=OUT/video['file']
        cmd=[FF,'-hide_banner','-i',str(path),'-vf','crop=1308:364:600:156,scale=654:182,fps=10',
             '-f','rawvideo','-pix_fmt','rgb24','-threads','2','-']
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        per_segment=[];sets=[set(),set()];first=None;middle=None;count=0
        while True:
            raw=read_exact(proc.stdout,654*182*3)
            if not raw:break
            assert len(raw)==654*182*3
            im=np.frombuffer(raw,dtype='uint8').reshape(182,654,3)
            # The upper scene area contains buildings/trees, not the frame counter.
            parts=[im[:120,:324],im[:120,330:654]]
            for j,part in enumerate(parts):sets[j].add(hashlib.sha256(part.tobytes()).hexdigest())
            pos=count%40
            if pos==0:first=[p.astype('f4').copy() for p in parts]
            if pos==20:middle=[p.astype('f4').copy() for p in parts]
            count+=1
            if pos==39:
                changes=[float(np.abs(a-b).mean()) for a,b in zip(first,middle)]
                assert min(changes)>.5,(video['file'],count//40,changes)
                assert min(len(s) for s in sets)>=35,(video['file'],count//40,[len(s) for s in sets])
                per_segment.append({'segment':count//40,'unique_global_before_frames':len(sets[0]),
                    'unique_global_after_frames':len(sets[1]),'before_scene_change':changes[0],'after_scene_change':changes[1]})
                sets=[set(),set()]
        stderr=proc.stderr.read().decode('utf-8',errors='replace');code=proc.wait()
        assert code==0,stderr
        assert count==1200 and len(per_segment)==30
        assert 'Constrained Baseline' in stderr and '1920x1080' in stderr
        reports.append({**video,'global_decoded_frames_at_10fps':count,'segments_checked':30,
                        'actual_scene_motion_passed':True,'segment_audit':per_segment})
        print('ALL SEGMENTS MOVING',video['name'],min(min(p['unique_global_before_frames'],p['unique_global_after_frames']) for p in per_segment),flush=True)
    gif=Image.open(OUT/index['looping_preview']);assert gif.n_frames==40 and gif.info.get('loop')==0
    gif.seek(0);a=gif.convert('RGB').crop((0,40,960,308));gif.seek(20);b=gif.convert('RGB').crop((0,40,960,308))
    delta=ImageStat.Stat(ImageChops.difference(a,b)).mean;assert np.mean(delta)>.5
    preview=subprocess.run([FF,'-hide_banner','-i',str(OUT/index['global_only']),'-f','null','-'],capture_output=True,text=True)
    assert preview.returncode==0
    (OUT/'运动逐段验证.json').write_text(json.dumps({'videos':reports,'all_segments':150,'both_global_panels_animated':True,
          'gif_frames':40,'gif_infinite_loop':True,'gif_scene_motion_mean_change':float(np.mean(delta)),
          'global_only_mp4_decodes':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 全局动态确认版','',
           '[只看全局前后：连续 8 秒](00_只看全局前后_连续8秒.mp4)','',
           '循环动图为同一视频中的实际全局两格，40 帧、10 fps、无限循环。下方综合版采用 1920×1080、H.264 Constrained Baseline、YUV420p 编码，并使用全新文件名。','',
           '双目综合版保留几何放大10倍；其他四类沿用原尺寸。每份均为2分钟，含10个资产在3条道路上的结果。','',
           '| 类别 | 综合视频 |','|---|---|']
    for v in reports:lines.append(f"| {v['name'][3:]} | [观看]({v['file']}) |")
    lines+=['','已直接解码五个最终文件，逐段检查全部150段的全局迁入前、迁入后画面；每段检查实际场景内容的连续变化，而非只检查帧号文字。结果见 `运动逐段验证.json`。','']
    (OUT/'观看说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print('GLOBAL MOTION CONFIRMED',flush=True)

if __name__=='__main__':main()
