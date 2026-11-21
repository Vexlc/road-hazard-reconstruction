"""Estimate relative inverse depth of the three pavement references; scale is assigned downstream."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


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
    options = {} if Path(args.model).is_dir() else {"revision": REVISION}
    processor = AutoImageProcessor.from_pretrained(args.model, **options)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, **options).to(device).eval()
    for name in args.names:
        src = root / "references" / (name + ".png")
        with Image.open(src) as opened:
            image = opened.convert("RGB")
            depth = .45 * infer(model, processor, image, device, 518) + .55 * infer(model, processor, image, device, 770)
        np.savez_compressed(out / (name + "_depth.npz"), inverse_depth=depth.astype("float16"),
                            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                            model=MODEL, revision=REVISION)
        print(name, depth.shape, flush=True)


if __name__ == "__main__":
    main()
