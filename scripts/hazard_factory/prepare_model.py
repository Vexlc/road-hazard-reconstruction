"""Fetch only the pinned road segmentation weights needed for uncached roads."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT=Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME',str(PROJECT/'work/hf_cache'))
path=snapshot_download('nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
                       revision='21b3847fae21ddee674abd31129307b6a1235bd9',
                       cache_dir=str(PROJECT/'work/hf_cache/hub'),
                       allow_patterns=['config.json','preprocessor_config.json','pytorch_model.bin','model.safetensors'])
print(path)
