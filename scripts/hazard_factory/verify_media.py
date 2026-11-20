"""Match both final global panels to the actual native driving frames, in order."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from run import atomic_json,sha


def read(path,filter,width,height):
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24',output_params=['-vf',filter])
    meta=next(reader)
    frames=[np.frombuffer(raw,dtype='u1').reshape(height,width,3).copy() for raw in reader]
    return meta,frames


def descriptor(patches):
    return np.stack([np.asarray(Image.fromarray(p).resize((32,20),Image.Resampling.BILINEAR)).reshape(-1)
                     for p in patches]).astype('f4')


def main(args):
    job=json.loads(args.job.read_text(encoding='utf-8'));scenes=Path(job['output'])
    root=scenes.parent;folder=root/'五类动态展示'
    index=json.loads((folder/'播放索引.json').read_text(encoding='utf-8'))
    for name,digest in index.get('publication',{}).get('files_sha256',{}).items():
        if sha(folder/name)!=digest:raise ValueError(f'Published media hash mismatch: {name}')
    if index.get('source_review_files_sha256'):
        for case,files in index['source_review_files_sha256'].items():
            for name,digest in files.items():
                if sha(scenes/case/name)!=digest:raise ValueError(f'Source review input changed: {case}/{name}')
    else:
        for case,digest in index.get('source_complete_sha256',{}).items():
            if sha(scenes/case/'complete.json')!=digest:raise ValueError(f'Source scene changed after video publication: {case}')
    results=[]
    for video in index['videos']:
        path=folder/video['file']
        meta,frames=read(path,'crop=1352:378:560:160,scale=676:189,fps=10',676,189)
        assert meta['source_size']==(1920,1080) and meta['fps']==24
        assert len(frames)==len(video['segments'])*40
        segments=[]
        for segment,first in zip(video['segments'],range(0,len(frames),40)):
            _,native=read(scenes/segment['case']/'global_before_after.mp4','scale=672:189',672,189)
            assert len(native)==40
            panels=[]
            for half in (0,1):
                source=[f[40:150,half*336+20:half*336+315] for f in native]
                actual=[f[40:150,half*340+20:half*340+315] for f in frames[first:first+40]]
                a,b=descriptor(actual),descriptor(source)
                distance=np.mean(abs(a[:,None,:]-b[None,:,:]),axis=2)
                matched=np.argmin(distance,axis=1)
                unique=len(np.unique(matched))
                errors=[float(np.mean(abs(p.astype('f4')-source[j].astype('f4')))) for p,j in zip(actual,matched)]
                if unique<38 or np.any(np.diff(matched)<0) or max(errors)>5.:
                    raise ValueError(f"{video['category']} {segment['case']} panel {half}: unique={unique}; MAE={max(errors):.3f}; indices={matched.tolist()}")
                panels.append({'panel':'before' if half==0 else 'after','matched_native_frame_indices':matched.tolist(),
                               'distinct_native_frames':unique,'monotonic_frame_order':True,
                               'max_mean_absolute_pixel_error':max(errors)})
            segments.append({'case':segment['case'],'asset_id':segment['asset_id'],'panels':panels})
        results.append({'category':video['category'],'file':video['file'],'sha256':sha(path),
                        'seconds':meta['duration'],'checked_segments':segments})
        print('MEDIA MATCH VERIFIED',video['category'],flush=True)
    atomic_json(root/'动态视频验证.json',{'passed':True,'videos':results,'index_sha256':sha(folder/'播放索引.json'),
               'total_segments':sum(len(v['checked_segments']) for v in results),
               'method':'Independent scene-pixel matching of both global panels to their native 40-frame clips; labels excluded.',
               'minimum_distinct_native_frames':min(p['distinct_native_frames'] for v in results
                    for s in v['checked_segments'] for p in s['panels'])})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);main(p.parse_args())
