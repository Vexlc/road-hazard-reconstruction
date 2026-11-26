"""Map monochrome laser returns to exposure and grain contrast observed around a camera patch."""

import numpy as np
import cv2


def camera_color(color, rgb, context, visible, part, stripe):
    # The laser's display range varies between scans. Fit the response on adjoining road only.
    gray = color[..., 0].astype("f4")
    luma = rgb.astype("f4") @ np.float32([0.2126, 0.7152, 0.0722])
    road = context if stripe is None else context & ~stripe
    if road.sum() < 8:
        road = context
    src = gray[road]
    src_mid = np.median(src)
    base = cv2.GaussianBlur(rgb.astype("f4"), (0, 0), 2.5)
    if stripe is not None:
        # Reconstruct road illumination on both sides independently of the host paint position.
        unpainted = cv2.inpaint(rgb, stripe.astype("u1"), 4, cv2.INPAINT_TELEA)
        base = cv2.GaussianBlur(unpainted.astype("f4"), (0, 0), 2.5)
    host_residual = luma - (base @ np.float32([0.2126, 0.7152, 0.0722]))
    src_iqr = np.diff(np.percentile(src, [20, 80]))[0]
    host_iqr = np.diff(np.percentile(host_residual[road], [20, 80]))[0]
    gain = np.clip(host_iqr / max(src_iqr, 1.0), 0.1, 2.0)
    contrast = gray[visible] - src_mid
    # Returns below the normal range on intact pavement often belong to a recessed fissure.
    # Keep its dark tail while matching the much weaker contrast of ordinary road aggregate.
    lower = np.percentile(src, 10) - src_mid
    tail = np.minimum(contrast - lower, 0.0)
    delta = gain * contrast + (part[visible] == 2) * max(0.0, 0.65 - gain) * tail
    values = base[visible] + delta[:, None]
    paint = part[visible] == 114
    if paint.any():
        # Keep the camera's actual paint exposure and antialiased edge pixels.
        paint_values = gray[visible][paint]
        values[paint] = (
            rgb[visible][paint].astype("f4")
            + (paint_values - np.median(paint_values))[:, None] * gain
        )
    return np.rint(values).clip(0, 255).astype("u1")
