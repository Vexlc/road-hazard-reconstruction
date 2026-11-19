import io,requests,zipfile,json,struct,zlib
from pathlib import Path
class RemoteZipIO(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.total=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def fetch(self,start,end):
  if end<start:return b''
  r=requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=45,stream=True)
  if r.status_code!=206:r.close();raise RuntimeError(f'Range request refused: {r.status_code}')
  b=r.content;r.close()
  if len(b)!=end-start+1:raise RuntimeError(f'Unexpected range length {len(b)} vs {end-start+1}')
  self.total+=len(b);return b
 def read(self,n=-1):
  n=min(self.size-self.pos,self.size if n<0 else n)
  b=self.fetch(self.pos,self.pos+n-1);self.pos+=len(b);return b
 def extract_member(self,info):
  hdr=self.fetch(info.header_offset,info.header_offset+29)
  fields=struct.unpack('<4s5H3I2H',hdr)
  start=info.header_offset+30+fields[-2]+fields[-1]
  compressed=self.fetch(start,start+info.compress_size-1)
  b=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed
  if zlib.crc32(b)&0xffffffff!=info.CRC:raise RuntimeError('CRC mismatch')
  return b
if __name__=='__main__':
 r=RemoteZipIO('https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip',44520528731)
 z=zipfile.ZipFile(r)
 infos=z.infolist();print('members',len(infos),'directory_bytes',r.total,flush=True)
 Path('work/pandaset_zip_index.json').write_text(json.dumps([{'name':x.filename,'size':x.file_size,'compressed':x.compress_size,'offset':x.header_offset,'crc':x.CRC,'method':x.compress_type} for x in infos]))
 candidates=[x for x in infos if '/camera/front_camera/00.jpg' in x.filename]
 print('front previews',len(candidates),[x.filename for x in candidates],flush=True)
 from concurrent.futures import ThreadPoolExecutor,as_completed
 Path('work/pandaset_previews').mkdir(exist_ok=True)
 def get(info):
  scene=info.filename.split('/camera/')[0].split('/')[-1]
  dest=Path('work/pandaset_previews')/(scene+'.jpg')
  if not dest.exists():dest.write_bytes(r.extract_member(info))
  return scene
 with ThreadPoolExecutor(max_workers=8) as ex:
  for f in as_completed([ex.submit(get,x) for x in candidates]):
   try:print('preview',f.result(),flush=True)
   except Exception as e:print('ERROR',str(e),flush=True)
 print('download_bytes',r.total,flush=True)
