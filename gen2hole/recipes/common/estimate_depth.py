"""Estimate relative inverse depth from square pavement references; scale is assigned downstream."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
RELEASE_WEIGHTS = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"


def infer(model, processor, image, device, resolution):
    data = processor(images=image, size={"height": resolution, "width": resolution}, return_tensors="pt")
    batch = data.pixel_values.to(device)
    prediction = model(torch.cat([batch, batch.flip(-1)], dim=0)).predicted_depth
    depth = (prediction[0] + prediction[1].flip(-1)) / 2
    depth = interpolate(depth[None, None], size=(image.height, image.width), mode="bicubic",
                        align_corners=False)[0, 0].cpu().numpy()
    low, high = np.percentile(depth, [1, 99])
    return (depth - low) / max(high - low, 1.e-6)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "calibration"
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = Path(args.model).is_dir()
    options = {"revision": REVISION} if not local and args.model == MODEL else {}
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=False, **options)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, **options).to(device).eval()
    model_name = args.model
    revision = getattr(model.config, "_commit_hash", None) or options.get("revision", "local")
    if local:
        weights = Path(args.model) / "model.safetensors"
        if weights.is_file():
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if digest == RELEASE_WEIGHTS:
                model_name, revision = MODEL, REVISION
            else:
                revision = "sha256:" + digest
    for name in args.names:
        src = root / "references" / (name + ".png")
        with Image.open(src) as opened:
            if opened.width != opened.height:
                raise ValueError(f"Expected a square nadir image: {src}")
            image = opened.convert("RGB")
            depth = .45 * infer(model, processor, image, device, 518) + .55 * infer(model, processor, image, device, 770)
        np.savez_compressed(out / (name + "_depth.npz"), inverse_depth=depth.astype("float16"),
                            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                            model=model_name, revision=revision)
        print(name, depth.shape, flush=True)


if __name__ == "__main__":
    main()
