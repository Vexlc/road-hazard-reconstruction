"""Write paired fish/perspective COLMAP models with a shared held-out set."""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def held_out(index):
    return index > 0 and index % 8 == 0


class FisheyeBundle:
    def __init__(self, dest, cameras, meta):
        self.dest, self.meta = dest, meta
        self.rows = {}
        for view, calibration in cameras.items():
            for key in (view, view.replace('all', 'train')):
                sparse = dest / key / 'sparse/0'
                sparse.mkdir(parents=True)
                (dest / key / 'images').mkdir()
                (sparse / 'cameras.txt').write_text(calibration + '\n')
                self.rows[key] = []
        meta.update(train=[], test=[], times={}, source_frames={})

    def frame(self, i, frame, time, pose, views):
        name = f'{i:04d}.png'
        w2c = np.linalg.inv(pose)
        q = Rotation.from_matrix(w2c[:3, :3]).as_quat()[[3, 0, 1, 2]]
        row = ' '.join(map(str, [i+1, *q, *w2c[:3, 3], 1, name])) + '\n\n'
        test = held_out(i)
        self.meta['test' if test else 'train'].append(name)
        self.meta['times'][name] = float(time)
        self.meta['source_frames'][name] = int(frame)
        for part, image in views.items():
            path = self.dest / part / 'images' / name
            if not cv2.imwrite(str(path), image):
                raise IOError(path)
            self.rows[part].append(row)
            if not test:
                key = part.replace('all', 'train')
                os.link(path, self.dest / key / 'images' / name)
                self.rows[key].append(row)

    def finish(self, samples):
        cloud = np.concatenate(samples)
        _, unique = np.unique(np.floor(cloud[:, :3]/.10), axis=0, return_index=True)
        cloud = cloud[unique]
        point_path = self.dest / 'all/sparse/0/points3D.txt'
        np.savetxt(point_path, np.column_stack((np.arange(1, len(cloud)+1), cloud)),
                   fmt='%d %.7f %.7f %.7f %d %d %d 0')
        for part, rows in self.rows.items():
            (self.dest / part / 'sparse/0/images.txt').write_text(''.join(rows))
            if part != 'all':
                os.link(point_path, self.dest / part / 'sparse/0/points3D.txt')
        self.meta['seed_points'] = len(cloud)
        (self.dest / 'all/split.json').write_text(json.dumps(self.meta, indent=2))
