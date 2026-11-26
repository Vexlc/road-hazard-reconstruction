"""Surfaces visible at the edges of the near exposure, without extrapolating its guard arrangement."""

import cv2
import numpy as np
from geometry import Parts, rings_faces, refine, unit, xyz


def observed_context(cam, photo, material, slab_z):
    parts = Parts()
    parts.mesh.materials["Near exposure"] = material
    red = (
        (photo[:, :, 0].astype(float) > 1.30 * photo[:, :, 1])
        & (photo[:, :, 0].astype(float) > 1.45 * photo[:, :, 2])
        & (photo[:, :, 0] > 105)
    )
    boxes = [(35, 0, 151, 94), (373, 0, 480, 58), (254, 394, 480, 544)]
    # Ground contact and a second bottom-edge point establish the vertical plane.
    contacts = [[[60, 88], [116, 82]], [[382, 30], [442, 53]], [[267, 538], [478, 463]]]
    for i, ((x0, y0, x1, y1), uv) in enumerate(zip(boxes, contacts)):
        local = np.uint8(red[y0:y1, x0:x1]) * 255
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((3, 3), "u1"))
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.85, True)[:, 0] + [x0, y0]
        a, b = cam.plane(uv, 0)
        n = unit(np.cross(b - a, [0.0, 0.0, 1.0]))
        poly, faces = refine(poly.astype(float), rings_faces([poly]), 10.0)
        q = cam.rays(poly)
        t = ((a - cam.eye) @ n) / (q @ n)
        if t.min() <= 0:
            raise ValueError("Unsupported panel fold")
        v = cam.eye + t[:, None] * q
        if np.cross(*(v[faces[0, 1:]] - v[faces[0, 0]])) @ (cam.eye - v[faces[0, 0]]) < 0:
            faces = faces[:, ::-1]
        parts.add(
            "Folding manhole guard panel detail " + str(i + 1),
            v,
            faces,
            (poly + 0.5) / [480, 640] * [1, -1] + [0, 1],
            "Near exposure",
        )
    # In this exposure the corner of a mottled slab enters at the lower right.
    poly = np.array([[276, 571], [479, 499], [479, 639], [288, 639]])
    v = cam.plane(poly, slab_z)
    parts.face(
        "Lid face",
        [v[:, :2]],
        lambda p: xyz(p, slab_z),
        "Near exposure",
        lambda p, v: (cam.project(v) + 0.5) / [480, 640] * [1, -1] + [0, 1],
        np.array([0.0, 0.0, 1.0]),
        0.04,
    )
    return parts.finish().parts


def coupling(cam):
    from scipy.optimize import minimize_scalar

    end = cam.plane([[171.6, 216.5]], -0.32)[0]
    rim = [
        [164.5, 216],
        [165.4, 212.9],
        [167.4, 210.3],
        [170, 209.5],
        [173.4, 210.5],
        [176.5, 212.7],
        [178.3, 216.7],
        [178.6, 219.7],
        [177, 222.3],
        [174.2, 223.5],
        [170.8, 223.2],
        [167.5, 221.4],
        [165.6, 219.2],
    ]
    rays = cam.rays(rim)

    def fit(z):
        axial = unit(end - cam.plane([[187, 204]], z)[0])
        t = ((end - cam.eye) @ axial) / (rays @ axial)
        radii = np.linalg.norm(cam.eye + t[:, None] * rays - end, axis=1)
        return axial, radii

    def loss(z):
        axial, radii = fit(z)
        return radii.std() / radii.mean() if axial @ (cam.eye - end) > 0 else 1.0

    fit_z = minimize_scalar(loss, bounds=(-0.52, -0.32), method="bounded").x
    axial, r = fit(fit_z)
    return end, axial, float(np.median(r))
