import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(Path(__file__).parent/'extra_deps'))
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor,AutoModelForSemanticSegmentation

ROOT=Path(__file__).parent/'data/processed'/sys.argv[1]
dest=ROOT/'sky_masks';dest.mkdir(exist_ok=True)
model_id='nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
revision='21b3847fae21ddee674abd31129307b6a1235bd9'
processor=AutoImageProcessor.from_pretrained(model_id,revision=revision,local_files_only=False)
model=AutoModelForSemanticSegmentation.from_pretrained(model_id,revision=revision,local_files_only=False).cuda().eval()
torch.set_num_threads(4)
for i,f in enumerate(sorted((ROOT/'images').glob('*_0.jpg'))):
    im=Image.open(f).convert('RGB')
    data=processor(images=im,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        logits=model(**data).logits
        pred=torch.nn.functional.interpolate(logits,size=(im.height,im.width),mode='bilinear',align_corners=False).argmax(1)[0]
    Image.fromarray((pred==10).cpu().numpy().astype('uint8')*255).save(dest/(f.stem+'.png'))
    if i%10==0: print('sky',i,flush=True)
print('SKY MASKS COMPLETE',i+1,flush=True)
