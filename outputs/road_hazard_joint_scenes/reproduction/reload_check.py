"""Render the standard StreetCrafter checkpoint entry point and compare its picture."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from create_scene import create_scene
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer


@torch.no_grad()
def main():
    scene=create_scene()
    camera=next(c for c in scene.getTrainCameras()+scene.getTestCameras() if c.meta["frame"]==44)
    rgb=StreetGaussianRenderer().render(camera,scene.gaussians)["rgb"]
    pixels=np.round(rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255)
    folder=Path(cfg.model_path)
    jpeg=np.asarray(Image.open(folder/"drive_44.jpg"),dtype="f4")
    assert jpeg.shape==pixels.shape
    mae=float(np.abs(pixels-jpeg).mean())
    assert mae < 2.,mae
    report={"scene":folder.name,"frame":44,"pixel_shape":list(pixels.shape),"mean_absolute_error_vs_saved_jpeg":mae}
    (folder/"reload_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print("RELOADED",report,flush=True)

if __name__=="__main__":
    main()
