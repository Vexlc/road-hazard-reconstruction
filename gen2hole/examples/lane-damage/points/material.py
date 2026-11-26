"""Metric swatches of exposed aggregate and adjacent pavement from photographed donor scans."""

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent


def texture(spec, rec, maps, base, wall_projection=None):
    rng = np.random.default_rng(19000 + int(rec["id"][1:]))
    donor = list(spec["materials"])[int(rec["id"][1:]) % 2]
    desc = spec["materials"][donor]
    path = base / donor / (donor + "_diff_2k.jpg")
    if hashlib.sha256(path.read_bytes()).hexdigest() != desc["diffuse"]["sha256"]:
        raise ValueError(path)
    image = np.asarray(Image.open(path).convert("RGB"))
    swatches = []
    surveys = json.loads((HERE / "swatches.json").read_text())
    block = 80
    for prior in surveys["states"]:
        if prior["material"] != donor:
            continue
        x, y = prior["origin_col_row"]
        n = prior["intervals"]
        src = image[y : y + n + 1, x : x + n + 1]
        mask = cv2.fillPoly(
            np.zeros(src.shape[:2], "u1"),
            [np.array(prior["photographed_rim_xy"], "i4")],
            1,
        )
        valid = (
            distance_transform_edt(np.pad(mask > 0, 1))[1:-1, 1:-1] > block * 0.72 + 5
        )
        ys, xs = np.nonzero(valid)
        for i in rng.choice(len(ys), min(80, len(ys)), replace=False):
            a, b = ys[i] - block // 2, xs[i] - block // 2
            swatches.append(src[a : a + block, b : b + block])
    h, w = (
        np.round(
            np.ptp(maps["xy"], axis=(0, 1))[::-1] * image.shape[0] / desc["scan_span_m"]
        ).astype(int)
        + 1
    )
    # Overlap adjacent photographed swatches, keeping the original grain scale and varied orientation.
    ramp = (
        np.minimum(np.arange(1, block + 1), np.arange(block, 0, -1)).clip(1, 12) / 12.0
    )
    window = (ramp[:, None] * ramp[None, :])[:, :, None].astype("f4")

    def quilt(pool):
        out = np.zeros((h + block, w + block, 3), "f4")
        weight = np.zeros((h + block, w + block, 1), "f4")
        for y in range(0, h, block - 12):
            for x in range(0, w, block - 12):
                patch = np.rot90(pool[rng.integers(len(pool))], rng.integers(4))
                out[y : y + block, x : x + block] += patch * window
                weight[y : y + block, x : x + block] += window
        return out[:h, :w] / weight[:h, :w]

    alpha = cv2.resize(
        maps["confidence"].astype("f4"), (w, h), interpolation=cv2.INTER_LINEAR
    )
    # Carry the wearing course down the upper wall. Only the deeper exposed zone takes the aggregate scan.
    depth = cv2.resize(-maps["z"], (w, h))
    depth_scale = max(0.018, 0.60 * np.percentile(-maps["z"][maps["mask"]], 85))
    exposure = np.clip((depth - 0.004) / depth_scale, 0, 1)
    exposure = exposure * exposure * (3 - 2 * exposure)
    alpha = (alpha * exposure)[:, :, None]
    road = spec["road_material"]
    path = base / road["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != road["sha256"]:
        raise ValueError(path)
    asphalt = np.asarray(Image.open(path).convert("RGB"))
    ratio = (
        asphalt.shape[0] / road["nominal_span_m"] * desc["scan_span_m"] / image.shape[0]
    )
    height, width = np.ceil(np.array([h, w]) * ratio).astype(int)
    y, x = [rng.integers(n - s + 1) for n, s in zip(asphalt.shape[:2], [height, width])]
    intact = cv2.resize(
        asphalt[y : y + height, x : x + width], (w, h), interpolation=cv2.INTER_AREA
    )
    # Transfer only the observed material contrast; photographed relief and shadows are not copied.
    samples = json.loads((HERE / "colors.json").read_text())
    reference = samples[(int(rec["id"][1:]) - 418) % len(samples)]
    bed = quilt(swatches)
    ratio = np.asarray(reference["bed_to_shoulder_srgb_p65"])
    bed *= (
        ratio
        * np.percentile(intact, 65, axis=(0, 1))
        / np.maximum(np.percentile(bed, 65, axis=(0, 1)), 1)
    )
    color = (intact * (1 - alpha) + bed * alpha if wall_projection is None else
             wall_projection(intact, bed, alpha, maps, desc["scan_span_m"] / image.shape[0]))
    return (
        np.uint8(np.clip(color, 0, 255)),
        donor,
        reference["capture"],
    )
