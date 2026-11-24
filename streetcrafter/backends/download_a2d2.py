"""Download synchronized front-left fisheye images and LiDAR from Audi's A2D2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from requests.adapters import HTTPAdapter, Retry


BASE = 'https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/'
DRIVE = '20180810_150607'
FRAMES = range(1500, 1564)


def download(root):
    files = [('cams_lidars.json', root / 'cams_lidars.json')]
    for frame in FRAMES:
        for kind, ext in [('camera', 'png'), ('camera', 'json'), ('lidar', 'npz')]:
            name = f'{DRIVE.replace("_", "")}_{kind}_frontleft_{frame:09d}.{ext}'
            key = f'camera_lidar/{DRIVE}/{kind}/cam_front_left/{name}'
            files.append((key, root / f'{frame:09d}.{ext}'))
    root.mkdir(parents=True, exist_ok=True)

    def get(entry):
        key, path = entry
        if path.is_file():
            return
        with requests.Session() as session:
            session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=.5,
                          status_forcelist=[429, 500, 502, 503, 504])))
            with session.get(BASE + key, timeout=60) as response:
                response.raise_for_status()
                data = response.content
            with NamedTemporaryFile(dir=root, delete=False) as tmp:
                tmp.write(data)
            Path(tmp.name).replace(path)
    with ThreadPoolExecutor(10) as pool:
        list(pool.map(get, files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    download(parser.parse_args().root)
