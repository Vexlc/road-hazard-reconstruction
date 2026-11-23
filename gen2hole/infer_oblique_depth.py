"""Cache independent original/flipped MoGe-2 predictions for a marked oblique photograph."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from moge.model.v2 import MoGeModel


def source_image(path):
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))


def overlay_mask(shape, spec):
    mask = np.zeros(shape, 'u1')
    for x0, y0, x1, y1, width in spec.get('overlay_segments', []):
        cv2.line(mask, (x0, y0), (x1, y1), 255, width)
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    rgb = source_image(args.image)
    assert list(rgb.shape[1::-1]) == spec['image_size']
    overlay = overlay_mask(rgb.shape[:2], spec)
    cleaned = cv2.inpaint(rgb, overlay, 3, cv2.INPAINT_TELEA) if overlay.any() else rgb
    x, y, r, b = spec['crop']
    cropped = cleaned[y:b, x:r]
    image = torch.from_numpy(cropped.copy()).permute(2, 0, 1).float().to(args.device) / 255
    model = MoGeModel.from_pretrained(args.weights).eval().to(args.device)
    maps = {}
    with torch.inference_mode():
        for k, view in [('direct', image), ('flip', image.flip(-1))]:
            result = model.infer(view, num_tokens=2500, use_fp16=args.device == 'cuda')
            depth = result['depth'].cpu().numpy()
            K = result['intrinsics'].cpu().numpy()
            if k == 'flip':
                depth = depth[:, ::-1]
                K[0, 2] = 1 - K[0, 2]
            K[:2] *= np.array([[r - x], [b - y]])
            K[:2, 2] += [x - .5, y - .5]
            full = np.full(rgb.shape[:2], np.nan, 'f4')
            full[y:b, x:r] = np.where(np.isfinite(depth), depth, np.nan)
            maps['depth_' + k] = full
            maps['K_' + k] = K
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **maps, image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                        spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(), tokens=2500)


if __name__ == '__main__':
    main()
