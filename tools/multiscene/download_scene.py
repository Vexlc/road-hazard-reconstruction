from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import json,struct,zlib,concurrent.futures,time, argparse
parser=argparse.ArgumentParser();parser.add_argument("--scene",default="056");scene=parser.parse_args().scene
assert scene in {"034", "109", "110"}
import sys
sys.path.insert(0,str(PROJECT/'outputs/reproducibility/preparation_scripts'))
from remote_zip import RemoteZipIO
entries=sorted([x for x in json.loads(Path(__file__).with_name('pandaset_zip_index.json').read_text()) if x['name'].startswith(f'pandaset/{scene}/')],key=lambda x:x['offset'])
start=entries[0]['offset'];end=entries[-1]['offset']+30+len(entries[-1]['name'].encode())+256+entries[-1]['compressed'];step=32*1024*1024
r=RemoteZipIO('https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip',44520528731)
cache=Path(__file__).parent/'downloads'/scene;cache.mkdir(exist_ok=True,parents=True)
def part(t):
 i,a,b=t;p=cache/f'{i:03d}.bin'
 if p.exists() and p.stat().st_size==b-a+1:return p
 for attempt in range(3):
  try:
   data=r.fetch(a,b);p.with_suffix('.tmp').write_bytes(data);p.with_suffix('.tmp').replace(p);print('chunk',i,'bytes',len(data),flush=True);return p
  except Exception as e:
   if attempt==2:raise
   print('retry',i,str(e),flush=True);time.sleep(1)
parts=[(i,a,min(a+step-1,end)) for i,a in enumerate(range(start,end+1,step))]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:paths=list(ex.map(part,parts))
block=b''.join(p.read_bytes() for p in paths);root=Path(__file__).parent/'data/raw'
for x in entries:
 p=root/x['name'].removeprefix('pandaset/')
 if x['name'].endswith('/'):p.mkdir(parents=True,exist_ok=True);continue
 offset=x['offset']-start;h=struct.unpack('<4s5H3I2H',block[offset:offset+30]);a=offset+30+h[-2]+h[-1];c=block[a:a+x['compressed']]
 b=zlib.decompress(c,-15) if x['method']==8 else c
 assert len(b)==x['size'] and (zlib.crc32(b)&0xffffffff)==x['crc'],x['name']
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
print('verified and extracted',len(entries),'entries',len(block),'bytes',flush=True)
