"""Integrate measured fissure reflectance over the camera pixel on the host's own painted pavement."""

import cv2
import numpy as np
import open3d as o3d


def sampled_hits(mesh, origins, directions, selected, samples, pixel_width):
    """Cast stratified subrays across each native camera pixel, preserving depth occlusion at each lip."""
    orig = origins[selected]
    ray = directions[selected]

    def steps(grid):
        return [
            np.gradient(grid, axis=i)[selected]
            if grid.shape[i] > 1
            else np.zeros_like(grid[selected])
            for i in (0, 1)
        ]

    dy, dx = steps(directions)
    oy, ox = steps(origins)
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * pixel_width
    lookup = np.zeros(max(mesh.surfaces) + 1, "u1")
    for gid, (part, *_) in mesh.surfaces.items():
        lookup[gid] = part
    raster = mesh.marked_crack["marked_crack"]
    for u in offsets:
        for v in offsets:
            o = orig + u * ox + v * oy
            d = ray + u * dx + v * dy
            hit = mesh.scene.cast_rays(
                o3d.core.Tensor(
                    np.ascontiguousarray(np.column_stack([o, d]), dtype="f4")
                )
            )
            t = hit["t_hit"].numpy()
            gid = hit["geometry_ids"].numpy()
            part = lookup[np.minimum(gid, len(lookup) - 1)]
            bed = np.isfinite(t) & np.isin(part, [2, 139])
            indices = np.flatnonzero(bed)
            xyz = o[bed] + d[bed] * t[bed, None]
            rc = np.floor(
                (0.5 + xyz[:, :2] / mesh.extent * [1, -1]) * raster.shape[::-1]
            ).astype(int)
            x, y = np.clip(rc, 0, np.array(raster.shape[::-1]) - 1).T
            yield indices, x, y


def compose(mesh, rgb, stripe, origins, directions, visible, pixel_width, samples=4):
    color = rgb.astype("f4")
    if stripe is not None and stripe.any():
        keep = (~stripe).astype("f4")
        count = cv2.GaussianBlur(keep, (0, 0), 11)
        road = cv2.GaussianBlur(color * keep[:, :, None], (0, 0), 11) / np.maximum(
            0.001, count[:, :, None]
        )
        asphalt = np.where(stripe[:, :, None], np.minimum(road, color), color)[visible]
    else:
        asphalt = color[visible]
    source = color[visible]
    paint = source - asphalt
    shift = np.zeros_like(source)
    for i, x, y in sampled_hits(
        mesh, origins, directions, visible, samples, pixel_width
    ):
        ratio = mesh.marked_crack["bed_intensity_ratio"][y, x, None]
        remain = mesh.marked_crack["paint_survival"][y, x, None]
        shift[i] += asphalt[i] * (ratio - 1) + paint[i] * (remain - 1)
    return np.uint8(np.rint(np.clip(source + shift / (samples * samples), 0, 255)))
