"""Split road Gaussians at a measured opening and retain their original radiance outside the lip."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def retained_ground(layer, world, world_rotation, placements):
    from pit_pairs import carve

    keep = carve(layer, world, placements)
    ids = np.flatnonzero(~keep)
    if not len(ids):
        return layer, 0
    parent = ids.copy()
    centers = world[ids].copy()
    sizes = layer["_scaling"][ids].exp().numpy()
    axes = world_rotation @ Rotation.from_quat(layer["_rotation"][ids].numpy()[:, [1, 2, 3, 0]]).as_matrix()
    alpha = layer["_opacity"][ids].sigmoid().numpy().ravel().clip(1e-6, 1 - 1e-6)
    optical = -np.log1p(-alpha)
    finished = []
    for step in range(9):
        clear = np.ones(len(parent), bool)
        interior = np.zeros(len(parent), bool)
        for cut, origin, basis in placements:
            xy = ((centers - origin) @ basis)[:, :2]
            distance = cut.signed_distance(xy)
            # Largest horizontal standard deviation of each oriented particle in the local road frame.
            xy_axes = (basis[:, :2].T @ axes) * sizes[:, None, :]
            cov = xy_axes @ xy_axes.transpose(0, 2, 1)
            radius = np.sqrt(np.linalg.eigvalsh(cov)[:, -1])
            guard = 2.1 * radius + 0.015
            clear &= distance > guard
            interior |= distance < -guard
        axis = np.argmax(sizes, axis=1)
        split = ~clear & ~interior & (sizes.max(1) > 0.033) & (step < 8)
        if clear.any():
            finished.append((parent[clear], centers[clear], sizes[clear], optical[clear]))
        if not split.any():
            break
        k = np.flatnonzero(split)
        j = axis[k]
        displacements = axes[k, :, j] * sizes[k, j, None] * (1 - 0.62**2) ** 0.5
        centers = np.concatenate([centers[k] - displacements, centers[k] + displacements])
        sizes = sizes[k].copy()
        sizes[np.arange(len(k)), j] *= 0.62
        # Match integrated optical density and the parent's first two spatial moments.
        optical = np.tile(optical[k] / (2 * 0.62), 2)
        sizes = np.tile(sizes, (2, 1))
        axes = np.tile(axes[k], (2, 1, 1))
        parent = np.tile(parent[k], 2)
    base = {k: v[keep] for k, v in layer.items()}
    if finished:
        parent, locations, sizes, optical = (np.concatenate(items) for items in zip(*finished))
        children = {k: v[parent] for k, v in layer.items()}
        children["_xyz"] = children["_xyz"] + torch.tensor(
            (locations - world[parent]) @ world_rotation, dtype=torch.float32
        )
        children["_scaling"] = torch.tensor(np.log(sizes), dtype=torch.float32)
        children["_opacity"] = torch.logit(
            torch.tensor(-np.expm1(-optical), dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        )[:, None]
        base = {k: torch.cat([v, children[k]]) for k, v in base.items()}
    return base, len(ids)
