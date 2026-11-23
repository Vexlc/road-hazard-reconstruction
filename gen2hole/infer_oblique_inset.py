"""Infer two detailed crops with the full-image camera calibration and return optical depths in its frame."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from moge.model.v2 import MoGeModel

from infer_oblique_depth import overlay_mask


def infer(model, rgb, K, centre, size, flip, device):
    # Rotate a virtual camera to look at the pit. A rectangular crop alone changes
    # the principal point, which the network assumes to be at its image centre.
    forward = np.linalg.solve(K, [*centre, 1])
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(forward, right), forward])
    w, h = size
    inner = K.copy()
    inner[0, 2], inner[1, 2] = (w-1)/2, (h-1)/2
    H = K @ rotation @ np.linalg.inv(inner)
    crop = cv2.warpPerspective(rgb, H, (w,h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT_101)
    view = torch.from_numpy(crop).permute(2,0,1).to(device).float()/255
    if flip:
        view = view.flip([-1])
    fov = float(np.rad2deg(2*np.arctan(w/(2*inner[0,0]))))
    result = model.infer(view, fov_x=fov, num_tokens=2500, use_fp16=device=='cuda')
    depth = result['depth'].float().cpu().numpy()
    if flip:
        depth = depth[:, ::-1]
    y, x = np.indices((h,w), dtype='f4')
    rays = np.stack([(x-inner[0,2])/inner[0,0], (y-inner[1,2])/inner[1,1], np.ones_like(x)],-1)
    z = np.float32(depth * (rays @ rotation[2]))
    return cv2.warpPerspective(z, H, rgb.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=float('nan'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args()
    profile=json.loads(a.profile.read_text())
    key=profile['id']
    spec=json.loads((a.bundle/'calibration'/(key+'-outline.json')).read_text())
    src=a.bundle/'references'/(key+'.jpg')
    rgb=np.array(ImageOps.exif_transpose(Image.open(src)).convert('RGB'))
    overlay=overlay_mask(rgb.shape[:2],spec)
    clean=cv2.inpaint(rgb,overlay,3,cv2.INPAINT_TELEA)
    with np.load(a.bundle/'calibration'/(key+'-depth.npz')) as base:
        K=base['K_direct'].astype('f8')
        weights_hash=sha256(a.weights.read_bytes()).hexdigest()
        if str(base['weights_sha256']) != weights_hash:
            raise ValueError('Inset and global depth must use the same checkpoint')
    model=MoGeModel.from_pretrained(a.weights).eval().to(a.device)
    maps={}
    with torch.inference_mode():
        for i, view in enumerate(profile['insets']):
            maps['z'+str(i)]=infer(model,clean,K,view['centre'],view['size'],view['flip'],a.device)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**maps,K=K,source_sha256=sha256(src.read_bytes()).hexdigest(),
                        weights_sha256=weights_hash,insets=json.dumps(profile['insets'],sort_keys=True))


if __name__=='__main__':
    main()
