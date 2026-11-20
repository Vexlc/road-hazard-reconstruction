"""Cache road-only source RGB support, excluding dataset dynamic-object masks."""
import json
import os
from pathlib import Path
import sys

job=json.loads(Path(os.environ['HAZARD_FACTORY_JOB']).read_text(encoding='utf-8'))
sys.path.insert(0,job.get('transformers_path',''))
import numpy as np
from PIL import Image
import torch
import yaml
from run import atomic_json,sha

MODEL='nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
REVISION='21b3847fae21ddee674abd31129307b6a1235bd9'


def main():
    torch.set_num_threads(4)
    # A restored, hash-verified cache must work without downloading model weights
    # or importing transformers. Load the pinned model only for a missing frame.
    processor=model=None;road_id=0
    report={'model':MODEL,'revision':REVISION,'road_label_id':road_id,'scenes':{}}
    provenance=Path(job['work'])/'road_masks/provenance.json'
    previous=json.loads(provenance.read_text(encoding='utf-8')) if provenance.exists() else {}
    for site,config in job['configs'].items():
        data=Path(yaml.safe_load(Path(config).read_text(encoding='utf-8'))['source_path'])
        dest=Path(job['work'])/'road_masks'/site;dest.mkdir(parents=True,exist_ok=True)
        records=[]
        compatible=(previous.get('revision')==REVISION and previous.get('model')==MODEL
                    and previous.get('road_label_id')==road_id)
        old={r['frame']:r for r in previous.get('scenes',{}).get(site,[])} if compatible else {}
        for frame in range(0,80,3):
            source=data/'images'/f'{frame:03d}_0.jpg'
            output=dest/(source.stem+'.png')
            dynamic=data/'dynamic_masks/all'/f'{frame:03d}_0.png'
            if not dynamic.is_file():raise FileNotFoundError(dynamic)
            source_hash,dynamic_hash=sha(source),sha(dynamic)
            record=old.get(frame)
            if (record and output.is_file() and record['source_sha256']==source_hash
                    and record['dynamic_mask_sha256']==dynamic_hash and record['output_sha256']==sha(output)):
                records.append(record);continue
            if model is None:
                from transformers import AutoImageProcessor,AutoModelForSemanticSegmentation
                processor=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION,local_files_only=True,use_fast=False)
                model=AutoModelForSemanticSegmentation.from_pretrained(MODEL,revision=REVISION,local_files_only=True).cuda().eval()
                assert next(int(i) for i,label in model.config.id2label.items() if label=='road')==road_id
            im=Image.open(source).convert('RGB')
            inputs=processor(images=im,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                logits=model(**inputs).logits
                labels=torch.nn.functional.interpolate(logits,size=(540,960),mode='bilinear',align_corners=False).argmax(1)[0]
            mask=(labels==road_id).cpu().numpy()
            objects=np.array(Image.open(dynamic).convert('L').resize((960,540),Image.Resampling.NEAREST))>0
            mask &= ~objects
            Image.fromarray(mask.astype('u1')*255).save(output)
            records.append({'frame':frame,'source_sha256':source_hash,
                            'dynamic_mask_sha256':dynamic_hash,'road_fraction':float(mask.mean()),
                            'output_sha256':sha(output)})
        report['scenes'][site]=records
        print('ROAD MASKS',site,len(records),flush=True)
    atomic_json(Path(job['work'])/'road_masks/provenance.json',report)


if __name__=='__main__':main()
