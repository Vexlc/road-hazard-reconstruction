"""Extract the CC0 ambientCG scan and segment its drainage slots from the registered displacement."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image


def prepare(archive, dest):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != '87374fd38410e87f5e239a5191fc780b6d58dbdad18aebe012c627af4e96943b':
        raise ValueError('Expected the original ManholeCover010_1K-PNG.zip from ambientCG')
    with ZipFile(archive) as source:
        for layer, filename in [('Color', 'photo.png'), ('Displacement', 'relief.png'), ('Opacity', 'surround.png')]:
            raw = source.read('ManholeCover010_1K-PNG_' + layer + '.png')
            (dest / filename).write_bytes(raw)
    elevation = np.asarray(Image.open(dest / 'relief.png'))
    y, x = np.indices(elevation.shape)
    radius = np.hypot(x - 508, y - 511)
    dark_slots = ((elevation < 27000) & (radius > 261) & (radius < 306)).astype('u1')
    contours, _ = cv2.findContours(dark_slots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slots = [loop for loop in contours if 300 < cv2.contourArea(loop) < 900]
    if len(slots) != 24:
        raise ValueError('The scan must contain 24 separate slot mouths')
    mask = np.zeros_like(dark_slots)
    cv2.drawContours(mask, slots, -1, 255, cv2.FILLED)
    Image.fromarray(mask).save(dest / 'openings.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='https://ambientcg.com/get?file=ManholeCover010_1K-PNG.zip')
    prepare(parser.parse_args().archive, Path(__file__).resolve().parent)
