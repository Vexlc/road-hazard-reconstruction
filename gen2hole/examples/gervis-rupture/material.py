"""Sample the source road at a fixed physical grain scale for novel viewpoints."""

import cv2
import numpy as np


def portable_asphalt(photo, cam, span):
    # Six clear centres in the near carriageway, below the painted line and ahead
    # of the tape. Rotations are restricted to 90-degree turns to preserve scale.
    centers = cam.plane([[54, 423], [75, 428], [111, 431], [59, 444], [85, 451], [113, 446]])[:, :2]
    pitch = 0.006
    width, height = np.ceil(span / pitch).astype(int)
    iy, ix = np.indices((height, width), dtype="f4")
    xy = np.stack(
        [((ix + 0.5) / width - 0.5) * span[0], (0.5 - (iy + 0.5) / height) * span[1]], axis=-1
    )
    cell = 0.12
    grid = xy / cell
    index = np.floor(grid).astype(int)
    fraction = grid - index
    rgb = np.zeros((*iy.shape, 3), float)
    weight = np.zeros(iy.shape, float)
    for di, dj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        a = index[..., 0] + di
        b = index[..., 1] + dj
        k = (a * 23 + b * 37 + a * b * 11) % len(centers)
        offset = xy - np.stack([a, b], -1) * cell
        turn = (a + 3 * b) % 4
        u, v = offset[..., 0], offset[..., 1]
        sample = centers[k] + np.stack(
            [np.choose(turn, [u, -v, -u, v]), np.choose(turn, [v, u, -v, -u])], -1
        )
        p = cam.project(np.dstack([sample, np.zeros(iy.shape)]))
        # No unseen texels are synthesized from outside the photograph.
        if (
            p[..., 0].min() < 0
            or p[..., 1].min() < 390
            or p[..., 0].max() > 162
            or p[..., 1].max() > 478
        ):
            raise ValueError("Donor window crosses the inspected asphalt swatch")
        values = cv2.remap(photo, p[..., 0].astype("f4"), p[..., 1].astype("f4"), cv2.INTER_LINEAR)
        w = (1 - abs(fraction[..., 0] - di)) * (1 - abs(fraction[..., 1] - dj))
        rgb += values * w[..., None]
        weight += w
    return np.uint8(np.rint(rgb / weight[..., None]).clip(0, 255))


def face_bands(photo, cam, edges, road_rgb):
    # A narrow source face has too few rows to carry a new high-resolution material.
    # Keep its along-edge colour and chips, reducing the baked source lighting.
    bands = []
    uvs = []
    for j, edge in enumerate(edges):
        bottom, top = edge
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(bottom[:, :2], axis=0), axis=1))]
        x = np.linspace(0, arc[-1], 480)
        top = np.column_stack([np.interp(x, arc, a) for a in top.T])
        bottom = np.column_stack([np.interp(x, arc, a) for a in bottom.T])
        t = (np.arange(40) + 0.5) / 40
        pos = bottom + (top - bottom) * t[::-1, None, None]
        uv = cam.project(pos)
        raw = cv2.remap(
            photo, uv[..., 0].astype("f4"), uv[..., 1].astype("f4"), cv2.INTER_LINEAR
        ).astype(float)
        median = np.maximum(np.median(raw[8:-8], axis=(0, 1)), 10.0)
        gain = np.clip(raw / median, 0.5, 1.75)
        bands.append(np.uint8(np.clip(road_rgb * 0.64 * gain**0.70, 0, 255)))
        n = len(edge[0])
        u = (0.5 + 479 * arc / arc[-1]) / 480
        v = np.r_[np.full(n, 1 - (j * 40 + 39.5) / 120), np.full(n, 1 - (j * 40 + 0.5) / 120)]
        uvs.append(np.c_[np.tile(u, 2), v])
    return np.vstack(bands), uvs


def roof_bands(photo, cam, roofs, atlas, extent):
    """Preserve observed chips where the source resolves the road in both directions."""
    grey = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype("f4")
    smooth = cv2.GaussianBlur(grey, (0, 0), 3.2)
    ratio = np.clip((grey + 10) / (smooth + 10), 0.45, 1.28)
    bands = []
    uvs = []
    W, H = 640, 256
    for i, roof in enumerate(roofs):
        rows, cols = roof.shape[:2]
        u, v = np.meshgrid(
            np.linspace(0, cols - 1, W, dtype="f4"), np.linspace(0, rows - 1, H, dtype="f4")
        )
        pos = cv2.remap(roof.astype("f4"), u, v, cv2.INTER_LINEAR)
        pixel = cam.project(pos)
        base_uv = (0.5 + pos[..., :2] / extent * [1, -1]) * atlas.shape[1::-1] - 0.5
        base = cv2.remap(
            atlas, base_uv[..., 0].astype("f4"), base_uv[..., 1].astype("f4"), cv2.INTER_LINEAR
        )
        detail = cv2.remap(
            ratio, pixel[..., 0].astype("f4"), pixel[..., 1].astype("f4"), cv2.INTER_LINEAR
        )
        pu, pv = np.gradient(pixel, axis=(1, 0))
        wu, wv = np.gradient(pos, axis=(1, 0))
        # Minimum image resolution (pixels per metre) over tangent directions.
        # The ratio of projected area to maximal stretch is the smaller singular value.
        area = np.linalg.norm(np.cross(wu, wv), axis=2).clip(1.0e-10)
        det = abs(pu[..., 0] * pv[..., 1] - pu[..., 1] * pv[..., 0]) / area
        metric = (
            np.sum(pu * pu, axis=-1) * np.sum(wv * wv, axis=-1)
            + np.sum(pv * pv, axis=-1) * np.sum(wu * wu, axis=-1)
            - 2 * np.sum(pu * pv, axis=-1) * np.sum(wu * wv, axis=-1)
        ) / area**2
        largest = np.sqrt(
            0.5 * (metric + np.sqrt(np.maximum(metric**2 - 4 * det**2, 0))).clip(1.0e-8)
        )
        weight = 0.65 * np.clip((det / largest - 10) / 28, 0, 1)
        bands.append(np.uint8(np.rint(base * (1 + weight * (detail - 1))[..., None]).clip(0, 255)))
        y, x = np.indices((rows, cols))
        uvs.append(
            np.stack(
                [
                    (0.5 + x * (W - 1) / (cols - 1)) / W,
                    1 - (i * H + 0.5 + y * (H - 1) / (rows - 1)) / (H * len(roofs)),
                ],
                axis=-1,
            ).reshape(-1, 2)
        )
    return np.vstack(bands), uvs
