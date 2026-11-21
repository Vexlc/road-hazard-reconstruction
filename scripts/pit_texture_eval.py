"""Inspect the delivered GLB UVs and compare its embedded base color with held-out road photographs."""

from io import BytesIO
import json
import struct

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def read_texture(path, width):
    data = path.read_bytes()
    (n,) = struct.unpack_from('<I', data, 12)
    doc = json.loads(data[20 : 20 + n])
    binary = memoryview(data)[28 + n :]

    def buffer(index):
        v = doc['bufferViews'][index]
        start = v.get('byteOffset', 0)
        return binary[start : start + v['byteLength']]

    def accessor(index, dim):
        a = doc['accessors'][index]
        if a['componentType'] != 5126 or 'byteStride' in doc['bufferViews'][a['bufferView']]:
            raise ValueError('Unexpected UV or vertex storage')
        return np.frombuffer(
            buffer(a['bufferView']), dtype='<f4', offset=a.get('byteOffset', 0), count=a['count'] * dim
        ).reshape(-1, dim)

    mats = set()
    for mesh in doc['meshes']:
        for prim in mesh['primitives']:
            mats.add(prim['material'])
            pos = accessor(prim['attributes']['POSITION'], 3)
            uv = accessor(prim['attributes']['TEXCOORD_0'], 2)
            # glTF is Y-up and the exported image-space texture origin is at its top left.
            if not np.allclose(uv, 0.5 + pos[:, [0, 2]] / width, atol=2e-6):
                raise ValueError('UV map and metric surface disagree in ' + str(path))
    if len(mats) != 1:
        raise ValueError('Expected one photographed material')
    texture = doc['materials'][mats.pop()]['pbrMetallicRoughness']['baseColorTexture']['index']
    view = doc['images'][doc['textures'][texture]['source']]['bufferView']
    return np.array(Image.open(BytesIO(buffer(view))).convert('RGB'))


def correlate(a, b):
    a = a - a.mean()
    b = b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1.0e-8 else None


def compare(view, pose, geometry, width, texture, depth, pit, crop, output):
    basis = np.asarray(geometry['local_axes_in_anchor_camera'])
    center = np.asarray(geometry['local_origin_in_anchor_m'])
    y, x = np.nonzero(pit)
    xyz = view.xyz(np.c_[x, y], depth[y, x])
    local = (xyz @ pose[:3, :3].T + pose[:3, 3] - center) @ basis
    h, w = texture.shape[:2]
    u = (0.5 + local[:, 0] / width) * (w - 1)
    v = (0.5 - local[:, 1] / width) * (h - 1)
    # Integrate the projected texel footprint of each camera pixel. Point sampling
    # a high-resolution photograph aliases fine asphalt texture in distant views.
    result = view.rgb.copy()
    ymin, ymax = y.min(), y.max() + 1
    xmin, xmax = x.min(), x.max() + 1
    inside = pit[ymin:ymax, xmin:xmax]
    coordinates = np.zeros((*inside.shape, 2), 'f4')
    coordinates[y - ymin, x - xmin] = np.c_[u, v]
    nearest = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    maps = coordinates[tuple(nearest)]
    ddy = np.gradient(maps, axis=0) if maps.shape[0] > 1 else np.zeros_like(maps)
    ddx = np.gradient(maps, axis=1) if maps.shape[1] > 1 else np.zeros_like(maps)
    center = coordinates[y - ymin, x - xmin]
    dx = ddx[y - ymin, x - xmin]
    dy = ddy[y - ymin, x - xmin]
    for i in range(0, len(x), 20000):
        sl = slice(i, i + 20000)
        total = np.zeros((len(x[sl]), 3), 'f4')
        for oy in (-0.375, -0.125, 0.125, 0.375):
            for ox in (-0.375, -0.125, 0.125, 0.375):
                uv = (center[sl] + ox * dx[sl] + oy * dy[sl])[:, None, :]
                total += cv2.remap(texture, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[
                    :, 0
                ]
        result[y[sl], x[sl]] = np.uint8(np.rint(total / 16))
    Image.fromarray(result[crop]).save(output, quality=95)
    valid = view.mask & pit & (view.depth > 0) & (abs(view.depth - depth) < 0.035)
    valid = cv2.erode(valid.astype('u1'), np.ones((7, 7), 'u1')) > 0
    if not valid.any():
        return dict(pixels=0)
    a = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype('f4')
    b = cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY).astype('f4')
    da = a - cv2.GaussianBlur(a, (0, 0), 3)
    db = b - cv2.GaussianBlur(b, (0, 0), 3)
    return dict(
        pixels=int(valid.sum()),
        base_color_size_px=[w, h],
        raw_rgb_mae_0_255=float(abs(result[valid].astype('f4') - view.rgb[valid]).mean()),
        gray_correlation=correlate(a[valid], b[valid]),
        highpass_gray_correlation=correlate(da[valid], db[valid]),
    )
