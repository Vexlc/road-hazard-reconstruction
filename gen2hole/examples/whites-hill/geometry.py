"""Textured planar castings, perforated panels and swept small hardware."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lane-damage"))
from photo_mesh import Mesh, grid_faces, refine, rings_faces, xyz


def unit(v):
    return v / np.linalg.norm(v)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def rounded(x0, y0, x1, y1, r):
    pts = []
    for (x, y), angle in zip(
        [(x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r), (x0 + r, y0 + r)], [-90, 0, 90, 180]
    ):
        a = np.deg2rad(angle + np.linspace(0, 90, 7))
        pts.extend(np.column_stack([x + r * np.cos(a), y + r * np.sin(a)]))
    return np.array(pts)


def sample(photo, pixels, mode=cv2.BORDER_REPLICATE):
    return cv2.remap(photo, np.float32(pixels), None, cv2.INTER_LINEAR, borderMode=mode)


class Parts:
    def __init__(self):
        self.mesh = Mesh()
        self.groups = {}

    def add(self, name, verts, faces, uv, mat):
        self.groups.setdefault((name, mat), []).append(
            (np.asarray(verts), np.asarray(faces), np.asarray(uv))
        )

    def finish(self):
        for (name, mat), items in self.groups.items():
            v = []
            f = []
            uv = []
            offset = 0
            for points, faces, coords in items:
                v.append(points)
                f.append(faces + offset)
                uv.append(coords)
                offset += len(points)
            self.mesh.add(
                name,
                np.vstack(v),
                np.vstack(f),
                np.vstack(uv),
                mat,
                name == "Road around the aperture",
            )
        return self.mesh

    def face(self, name, rings, place, mat, uv, normal, pitch=0.1):
        q = np.vstack(rings)
        f = rings_faces(rings)
        # Refine with the planar parameters attached for non-linear texture projection.
        if pitch:
            flat, tri = refine(q, f, pitch)
            verts = place(flat)
            f = tri
        else:
            verts = place(q)
            flat = q
        a, b, c = verts[f].transpose(1, 0, 2)
        flip = (np.cross(b - a, c - a) @ normal) < 0
        f[flip] = f[flip, ::-1]
        self.add(name, verts, f, uv(flat, verts), mat)

    def tube(self, name, path, radii, mat, segments=24, inward=False):
        path = np.asarray(path)
        radii = np.broadcast_to(radii, (len(path),))
        t = np.gradient(path, axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None]
        initial = np.eye(3)[np.argmin(abs(t[0]))]
        axis = unit(np.cross(t[0], initial))
        rings = []
        coords = []
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        a = np.linspace(0, 2 * np.pi, segments + 1)
        for p, r, w, length in zip(path, radii, t, lengths):
            axis = unit(axis - w * (w @ axis))
            cross = np.cross(w, axis)
            rings.extend(p + r * (np.cos(a)[:, None] * axis + np.sin(a)[:, None] * cross))
            coords.extend(np.c_[a / (2 * np.pi), np.full(len(a), length / max(lengths[-1], 1e-8))])
        verts = np.asarray(rings)
        f = grid_faces(len(path), len(a))
        centroid = verts[f].mean(1)
        # grid_faces uses one winding. Compare a normal to the local radial direction.
        a0, b0, c0 = verts[f[0]]
        if np.cross(b0 - a0, c0 - a0) @ (centroid[0] - path[:2].mean(0)) < 0:
            f = f[:, ::-1]
        self.add(name, verts, f[:, ::-1] if inward else f, coords, mat)


def strip(parts, name, ring, bottom, top, mat):
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        v = np.array([[*a, bottom], [*b, bottom], [*b, top], [*a, top]])
        parts.add(name, v, [[0, 1, 2], [0, 2, 3]], rect(0, 0, 1, 1), mat)
