import os
os.environ.setdefault('HF_HOME', 'work/hf_cache')
from pathlib import Path
import torch,numpy as np
from PIL import Image
from transformers import AutoImageProcessor,AutoModelForSemanticSegmentation
model_id='nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
processor=AutoImageProcessor.from_pretrained(model_id, revision='21b3847fae21ddee674abd31129307b6a1235bd9')
model=AutoModelForSemanticSegmentation.from_pretrained(model_id, revision='21b3847fae21ddee674abd31129307b6a1235bd9').cuda().eval()
print('labels',model.config.id2label,flush=True)
import sys
scene=sys.argv[1]
root=Path('work/data/processed')/scene;dest=root/'sky_masks';dest.mkdir(exist_ok=True)
for i,f in enumerate(sorted((root/'images').glob('*_0.jpg'))):
 im=Image.open(f).convert('RGB')
 inputs=processor(images=im,return_tensors='pt').to('cuda')
 with torch.inference_mode():
  logits=model(**inputs).logits
  pred=torch.nn.functional.interpolate(logits,size=(im.height,im.width),mode='bilinear',align_corners=False).argmax(1)[0]
  mask=(pred==10).cpu().numpy().astype(np.uint8)*255
 Image.fromarray(mask).save(dest/(f.stem+'.png'))
 if i%10==0:print('sky',i,'pixels',int((mask>0).sum()),flush=True)
print('sky masks complete',i+1,flush=True)
