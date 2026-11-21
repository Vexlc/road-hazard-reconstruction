"""Compare P90 reconstructions to the held-out PothRGBD annotation in source pixels."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "gen2hole/runs/real-rgb-check"


def metrics(path, reference):
    with Image.open(path) as image:
        predicted = np.asarray(image) > 127
    if predicted.shape != reference.shape:
        raise ValueError("The fitted mask must match the source crop")
    hits = int(np.sum(predicted & reference))
    union = int(np.sum(predicted | reference))
    return dict(intersection=hits, union=union, iou=hits/union,
                precision=hits/max(1,int(predicted.sum())), recall=hits/int(reference.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/rgb-box-P90")
    parser.add_argument("--out", type=Path, default=ROOT / "validation/rgb-box-P90.json")
    args = parser.parse_args()
    bundle = args.run / "series"
    previous = json.loads((BASELINE / "rgb-evaluation.json").read_text())
    rgb = "references/P90.png"
    if (bundle / rgb).read_bytes() != (BASELINE / "series" / rgb).read_bytes():
        raise ValueError("The RGB sources differ")
    annotation = previous["posthoc_reference"]["annotation"]
    with ZipFile(ROOT / "work/reconstruction-inputs/rgbd/labels.zip") as z:
        raw = z.read(annotation["filename"])
    if hashlib.sha256(raw).hexdigest() != annotation["sha256"]:
        raise ValueError("Annotation checksum mismatch")
    width, height = previous["raw_image_wh"]
    canvas = Image.new("L",(width,height))
    brush = ImageDraw.Draw(canvas)
    for line in raw.decode().splitlines():
        xy = np.array(line.split()[1:],dtype=float).reshape(-1,2)*[width-1,height-1]
        brush.polygon(list(map(tuple,xy)),fill=255)
    components, _ = ndimage.label(np.asarray(canvas)>0)
    foreground = components == (np.bincount(components.ravel())[1:].argmax()+1)
    x0,y0,x1,y1 = previous["crop_xyxy"]
    reference = foreground[y0:y1,x0:x1]
    mask = "calibration/P90_outline.png"
    meta = json.loads((bundle / "assets/pothole/P90/metadata.json").read_text())
    report = dict(asset="P90",source_asset=previous["source_asset"],crop_xyxy=previous["crop_xyxy"],
                  validation_mask=annotation,rgb_box_480=meta["rgb_box_480"],geometry=meta["depth_profile"],
                  design_depth_m=meta["depth_design_m"],unguided=metrics(BASELINE / "series" / mask, reference),
                  rgb_box=metrics(bundle / mask, reference),rebuilt=json.loads((bundle / "checks.json").read_text())["P90"])
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(f"Mask IoU: {report['unguided']['iou']:.4f} unguided, {report['rgb_box']['iou']:.4f} with RGB box")


if __name__ == "__main__":
    main()
