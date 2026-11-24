"""Fetch only the KITTI-360 ZIP members used by the driving clip."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

import requests
from requests.adapters import HTTPAdapter, Retry


BASE = 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/'


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.client = requests.Session()
        self.client.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5,
                          status_forcelist=[429, 500, 502, 503, 504])))
        response = self.client.get(url, headers={'Range': 'bytes=0-0'}, timeout=45, stream=True)
        with response:
            response.raise_for_status()
            self.size = int(response.headers['Content-Range'].split('/')[-1])
        self.pos = 0

    def seek(self, offset, whence=0):
        self.pos = max(0, [0, self.pos, self.size][whence] + offset)
        return self.pos

    def tell(self):
        return self.pos

    def range(self, start, end):
        with self.client.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=90) as r:
            r.raise_for_status()
            if r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise IOError(f'Incorrect server range at byte {start}')
            return r.content

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.pos + size)
        if self.pos >= end:
            return b''
        block = self.range(self.pos, end - 1)
        self.pos = end
        return block


def fetch(root, archive, members):
    members = {name: root / dest for name, dest in members.items() if not (root / dest).is_file()}
    if not members:
        return
    remote = RangeFile(BASE + archive)
    with zipfile.ZipFile(remote) as zipped:
        entries = [(zipped.getinfo(name), dest) for name, dest in members.items()]

    def extract(entry):
        info, path = entry
        start = info.header_offset
        # Include the variable-length local header and validate the exact payload by CRC.
        head = remote.range(start, start + 29)
        length = sum(struct.unpack_from('<HH', head, 26))
        offset = start + 30 + length
        payload = remote.range(offset, offset + info.compress_size - 1)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            payload = zlib.decompress(payload, -15)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(info.compress_type)
        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
            raise IOError(f'Corrupt member {info.filename}')
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
            fp.write(payload)
        Path(fp.name).replace(path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(extract, entries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--drive', default='0000')
    parser.add_argument('--camera', choices=['image_02', 'image_03'], default='image_02')
    parser.add_argument('--start', type=int, default=200)
    parser.add_argument('--count', type=int, default=48)
    args = parser.parse_args()
    root, cam = args.root, args.camera
    drive = f'2013_05_28_drive_{args.drive}_sync'
    frames = range(args.start, args.start + args.count)
    for archive, folder, suffix, dest in [
        ('data_2d_raw/' + drive + '_' + cam + '.zip', f'{drive}/{cam}/data_rgb', 'png', cam),
        ('data_3d_raw/' + drive + '_velodyne.zip', drive + '/velodyne_points/data', 'bin', 'velodyne'),
    ]:
        fetch(root, archive, {f'{folder}/{i:010d}.{suffix}': f'{dest}/{i:010d}.{suffix}' for i in frames})
    fetch(root, '89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip',
          {f'{drive}/poses.txt': 'poses.txt'})
    fetch(root, 'data_2d_raw/data_timestamps_fisheye.zip',
          {f'{drive}/{cam}/timestamps.txt': f'{cam}/timestamps.txt'})
    fetch(root, '384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip', {f'calibration/{name}': f'calibration/{name}' for name in
          [cam+'.yaml', 'calib_cam_to_pose.txt', 'calib_cam_to_velo.txt']})
