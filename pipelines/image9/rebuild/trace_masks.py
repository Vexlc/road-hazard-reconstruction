"""Refine image-space pavement outlines with a narrow-band seeded GrabCut."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes


ROOT = Path(__file__).resolve().parents[1]


def refine(rgb, polygon, band):
    n = rgb.shape[0]
    xy = np.rint(np.array(polygon) * ((n - 1) / 479)).astype("i4")
    initial = np.zeros(rgb.shape[:2], "u1")
    cv2.fillPoly(initial, [xy], 1)
    radius = int(round(band * n/480))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    inner = cv2.erode(initial, kernel) > 0
    outer = cv2.dilate(initial, kernel) > 0
    trimap = np.where(initial, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('u1')
    trimap[inner] = cv2.GC_FGD
    trimap[~outer] = cv2.GC_BGD
    cv2.setRNGSeed(27)
    cv2.grabCut(rgb, trimap, None, np.zeros((1,65)), np.zeros((1,65)), 5, cv2.GC_INIT_WITH_MASK)
    classes = (trimap & 1).astype('u1')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(classes)
    dense = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.uint8(binary_fill_holes(dense)) * 255


if __name__ == "__main__":
    layout = json.loads((ROOT/'calibration/layout.json').read_text())
    for name, spec in layout.items():
        if 'outline' not in spec:
            continue
        img = np.asarray(Image.open(ROOT/'references'/(name+'.png')).convert('RGB'))
        for key, band in [('outline',8),('water',6)]:
            if key in spec:
                mask = refine(img, spec[key], band)
                Image.fromarray(mask).save(ROOT/'calibration'/(name+'_'+key+'.png'))
