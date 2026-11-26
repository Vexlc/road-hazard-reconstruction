"""Fit directional colour on visible water splats using the two registered exposures."""

import cv2
import numpy as np
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt
from photo_mesh import Surface, normalize


def fit_water_radiance(folder, pair, shift, photos):
    sidecar = np.load(folder / "splat_material.npz")
    water = sidecar["part_names"][sidecar["part"]] == "Water surface"
    ply = PlyData.read(folder / "asset.ply", mmap="c")
    vertices = ply["vertex"].data
    points = np.column_stack([vertices[i][water] for i in "xyz"])
    geometry = Surface(folder / "asset.glb")
    directions, colours, confidences = [], [], []
    for k, name in enumerate(("west", "northwest")):
        photo = photos[k]
        mask = np.zeros(photo.shape[:2], "u1")
        for polygon in pair.s["water_sampling"][name]:
            cv2.fillPoly(mask, [np.asarray(polygon, "i4")], 1)
        # Fill outside the hand-drawn open-water regions only to feather the boundary.
        # Normalised convolution keeps pipe/shore pixels out of the smoothed colour.
        blur = lambda x: cv2.GaussianBlur(x, (0, 0), 2.2)
        blurred = blur(photo.astype("f4") * mask[:, :, None]) / np.maximum(
            blur(mask.astype("f4"))[:, :, None], 1.0e-6
        )
        distance, indices = distance_transform_edt(1 - mask, return_indices=True)
        uv = pair.project(points + shift, k)
        x, y = np.rint(uv).astype(int).T
        h, w = mask.shape
        in_frame = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x.clip(0, w - 1), y.clip(0, h - 1)
        rgb = blurred[indices[0, y, x], indices[1, y, x]] / 255
        eye = pair.eye(k) - shift
        delta = points - eye
        # An unobstructed ray reaches its target at parameter 1.
        depth, part, _ = geometry.cast(np.broadcast_to(eye, points.shape), delta)
        visible = (part == 4) & (abs(depth - 1) < 0.004) & in_frame
        weight = np.exp(-distance[y, x] / 23) * visible
        directions.append(normalize(delta))
        colours.append(rgb)
        confidences.append(weight)
    weights = np.stack(confidences, axis=1)
    w0, w1 = weights.T
    total = weights.sum(1)
    neutral = np.array([100, 95, 73]) / 255
    blend = np.minimum(total, 1)[:, None]
    mean = (
        (w0[:, None] * colours[0] + w1[:, None] * colours[1]) / np.maximum(total[:, None], 1.0e-8)
    ) * blend + neutral * (1 - blend)
    delta = directions[0] - directions[1]
    weight = np.minimum(w0, w1)[:, None]
    gradient = (
        weight[:, :, None]
        * (colours[0] - colours[1])[:, :, None]
        * delta[:, None, :]
        / (0.14 + (delta * delta).sum(1)[:, None, None])
    )
    # Clamp extrapolation outside the photographed viewing angles.
    gradient = gradient.clip(-0.32, 0.32)
    mean_dir = (w0[:, None] * directions[0] + w1[:, None] * directions[1]) / np.maximum(
        total[:, None], 1.0e-8
    )
    base = mean - np.einsum("nci,ni->nc", gradient, mean_dir)
    c0, c1 = 0.28209479177387814, 0.4886025119029199
    for channel in range(3):
        vertices["f_dc_" + str(channel)][water] = (base[:, channel] - 0.5) / c0
        for j, (axis, sign) in enumerate(((1, -1), (2, 1), (0, -1))):
            vertices["f_rest_" + str(channel * 3 + j)][water] = (
                sign * gradient[:, channel, axis] / c1
            )
    # Use a second inode because input may be memory mapped.
    tmp = folder / "water-colour.ply"
    ply.write(tmp)
    tmp.replace(folder / "asset.ply")
    return dict(
        method="Masked two-view directional SH1; regularised, neutral fallback",
        water_splats=int(water.sum()),
        west_confident=int((w0 > 0.5).sum()),
        northwest_confident=int((w1 > 0.5).sum()),
        both_confident=int((np.min(weights, axis=1) > 0.5).sum()),
    )
