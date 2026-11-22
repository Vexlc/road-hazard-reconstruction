"""Prepare the registered reference, opening mask and pavement sample from the archived 4K maps."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image


def prepare(inputs, output):
    hashes = json.loads(Path(__file__).resolve().parents[1].joinpath('sources.json').read_text())
    maps = {}
    for role, material, channel in [('color', 'ManholeCover009', 'Color'),
                                    ('displacement', 'ManholeCover009', 'Displacement'),
                                    ('4k_pavement', 'Asphalt031', 'Color')]:
        image = inputs / material / f'{material}_4K-PNG_{channel}.png'
        if sha256(image.read_bytes()).hexdigest() != hashes['split_iron_' + role + '_sha256']:
            raise ValueError('Unexpected source map ' + image.name)
        maps[role] = image
    output.mkdir()
    with Image.open(maps['color']) as raw:
        raw.resize((1024, 1024), Image.Resampling.LANCZOS).save(output / 'photo.jpg', quality=96, subsampling=0)
    height = cv2.resize(cv2.imread(str(maps['displacement']), cv2.IMREAD_UNCHANGED),
                        (1024, 1024), interpolation=cv2.INTER_AREA)
    if height.dtype != np.uint16:
        raise ValueError('Expected 16-bit displacement')
    cv2.imwrite(str(output / 'relief.png'), height)
    # The fourteen dark slots in the RGB have a near-zero plateau in the registered displacement.
    n, labels, stats, centers = cv2.connectedComponentsWithStats(np.uint8(height < 5500))
    holes = [i for i in range(1, n) if stats[i, 4] > 500 and np.linalg.norm(centers[i] - [512, 512]) < 350]
    if len(holes) != 14:
        raise ValueError('Expected fourteen distinct perforations in Manhole Cover 009')
    Image.fromarray(np.uint8(np.isin(labels, holes)) * 255).save(output / 'openings.png')
    with Image.open(maps['4k_pavement']) as raw:
        surface = raw.resize((1024, 1024), Image.Resampling.LANCZOS)
        surface.crop((165, 423, 549, 807)).save(output / 'pavement.jpg', quality=94)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='Directory with the two ambientCG material folders')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error('Choose a new output directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / 'source'
        prepare(args.inputs, stage)
        stage.rename(args.out)
