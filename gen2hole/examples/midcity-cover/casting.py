"""Rectify the photographed casting and register the two surviving tread fields."""

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box

from calibration import SIZE, THICKNESS

SIDE = 640
SPAN = 0.62
CENTER = (SIDE - 1) / 2
UPRIGHT = cv2.getRotationMatrix2D((CENTER, CENTER), 40, 1)
BACK = cv2.invertAffineTransform(UPRIGHT)


def transform(p, affine):
    return np.asarray(p) @ affine[:, :2].T + affine[:, 2]


def model_xy(p, cam):
    return (transform(p, BACK) - CENTER) * (np.array([1, -1]) * SPAN / SIDE) + cam.lid_xy


def texture(src, cam):
    yy, xx = np.indices((SIDE, SIDE))
    xy = model_xy(np.stack([xx, yy], axis=-1), cam)
    px = cam.project(np.dstack([xy, np.full(xx.shape, THICKNESS)]))
    px = (px + 0.5) * np.array(src.shape[1::-1]) / SIZE - 0.5
    return cv2.remap(src, px[:, :, 0].astype("f4"), px[:, :, 1].astype("f4"), cv2.INTER_LINEAR)


def cast_shapes(rgb):
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("f4"), (5, 5), 1.0)
    gy, gx = np.gradient(g)
    # Clip the diamond field to the observed circular border and the blank lettering panel.
    rim = Point(320, 311).buffer(251, resolution=128)
    tread = []
    for upper in (True, False):
        field = rim.intersection(box(0, 0, 640, 258) if upper else box(0, 368, 640, 640))
        base, step = (250, -56) if upper else (376, 55)
        for row in range(5):
            cy = base + row * step
            for col in range(-5, 7):
                cx = 296 + 51 * col + 25.5 * (row % 2)
                nominal = np.array(
                    [[cx, cy - 48], [cx + 22, cy], [cx, cy + 48], [cx - 22, cy]], float
                )
                # Look along edge normals for the actual bevel near the estimated lattice.
                # A median across each intact edge suppresses isolated rust and photographs of grit.
                lines = []
                for start, end in zip(nominal, np.roll(nominal, -1, axis=0)):
                    vec = end - start
                    normal = np.array([vec[1], -vec[0]]) / np.linalg.norm(vec)
                    s = np.linspace(0.12, 0.88, 21)
                    edge = start + s[:, None] * vec
                    clear = np.array([field.contains(Point(p)) for p in edge])
                    delta = 0.0
                    if clear.sum() >= 7:
                        offsets = np.arange(-5, 6)
                        q = edge[clear, None, :] + offsets[None, :, None] * normal
                        q = np.rint(q).astype(int)
                        gradient = np.abs(
                            gx[q[..., 1], q[..., 0]] * normal[0]
                            + gy[q[..., 1], q[..., 0]] * normal[1]
                        )
                        score = np.median(gradient, axis=0) - abs(offsets) * 0.42
                        delta = float(offsets[np.argmax(score)])
                    lines.append((normal, normal @ start + delta))
                vertices = [
                    np.linalg.solve(
                        np.array([lines[j - 1][0], lines[j][0]]), [lines[j - 1][1], lines[j][1]]
                    )
                    for j in range(4)
                ]
                p = Polygon(vertices).intersection(field)
                if not p.is_empty and p.area > 120:
                    tread.append(p.simplify(0.65))
    # Traces of raised iron faces in the upright 640-pixel casting atlas.
    l = Polygon([(153, 270), (165, 269), (166, 337), (204, 337), (204, 350), (153, 351)])
    d = Polygon(
        [(297, 270), (343, 269), (350, 273), (352, 279), (352, 340), (345, 347), (297, 348)],
        [[(315, 282), (315, 334), (331, 333), (336, 327), (335, 286), (331, 280)]],
    )
    h = Polygon(
        [
            (450, 269),
            (462, 269),
            (462, 300),
            (489, 300),
            (489, 268),
            (501, 268),
            (501, 349),
            (488, 349),
            (488, 315),
            (462, 315),
            (462, 349),
            (449, 349),
        ]
    )
    dots = [box(x, 335, x + 13, 348) for x in (232, 380, 533)]
    return tread, [l, d, h, *dots]


def perimeter(cam, n=480):
    phase = np.arange(n) * 2 * np.pi / n
    r = np.full(n, cam.lid_radius * SIDE / SPAN)
    x = CENTER + r * np.cos(phase)
    # Follow the locally missing material on the south edge of the rectified photograph.
    marks = np.array(
        [
            [282, 625],
            [288, 623],
            [292, 620],
            [297, 619],
            [301, 613],
            [306, 612],
            [311, 615],
            [320, 615],
            [325, 613],
            [330, 616],
            [339, 615],
            [347, 613],
            [351, 616],
            [357, 624],
            [363, 626],
        ]
    )
    use = (x >= 282) & (x <= 363) & (np.sin(phase) > 0.9)
    y = np.interp(x[use], marks[:, 0], marks[:, 1])
    r[use] = np.minimum(r[use], (y - CENTER) / np.sin(phase[use]))
    return model_xy(CENTER + r[:, None] * np.stack([np.cos(phase), np.sin(phase)], axis=-1), cam)[
        ::-1
    ]
