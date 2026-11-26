"""Rectify annotated road photographs into scale-free RGBA road decals."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
COLORS = np.array([[0, 0, 0], [230, 169, 51], [36, 204, 209], [231, 98, 127]], np.uint8)


def transform(points, homography):
    xy = np.asarray(points, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(xy, homography)[:, 0]


def road_projection(width, height, focal, horizon):
    # Project [right, forward, 1] at camera height one with a pitched pinhole.
    cx, cy = (width - 1) / 2, (height - 1) / 2
    tilt = np.arctan((cy - horizon) / focal)
    c, s = np.cos(tilt), np.sin(tilt)
    return np.array(
        [[focal, cx * c, cx * s], [0, cy * c - focal * s, cy * s + focal * c], [0, c, s]]
    )


def source_labels(shape, row):
    labels = np.zeros(shape, np.uint8)
    for value, polygons in [(1, [row["extent"]]), (2, row["exposed"]), (3, row["retained"])]:
        for polygon in polygons:
            cv2.fillPoly(labels, [np.int32(polygon)], value)
    return labels


def panel(rgb, labels, crop):
    marks = labels > 0
    overlay = rgb.copy()
    overlay[marks] = np.rint(0.65 * rgb[marks] + 0.35 * COLORS[labels[marks]]).astype("u1")
    tile = Image.fromarray(overlay).crop(crop)
    tile.thumbnail((438, 310), Image.Resampling.LANCZOS)
    return tile


def build(name, row, dest):
    folder = dest / name
    folder.mkdir(parents=True, exist_ok=True)
    source = HERE / "references" / (row["photo"] + ".jpg")
    if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"{name}: source checksum mismatch")
    rgb = np.asarray(Image.open(source).convert("RGB"))
    height, width = rgb.shape[:2]
    labels = source_labels((height, width), row)
    yy, xx = np.nonzero(labels)
    uv = np.stack((xx, yy), axis=-1)
    pose = road_projection(width, height, row["focal"], row["horizon"])
    xy = transform(uv, np.linalg.inv(pose))
    lo, hi = xy.min(0) - 0.08, xy.max(0) + 0.08
    density = min(320, 920 / (hi - lo).max())
    size = tuple((np.ceil((hi - lo) * density).astype(int) + 1).tolist())
    plane_to_atlas = np.array(
        [[density, 0, -density * lo[0]], [0, -density, density * hi[1]], [0, 0, 1]]
    )
    image_to_atlas = plane_to_atlas @ np.linalg.inv(pose)

    def warp(arr, interp):
        return cv2.warpPerspective(arr, image_to_atlas, size, flags=interp)

    color = warp(rgb, cv2.INTER_LINEAR)
    regions = warp(labels, cv2.INTER_NEAREST)
    distance = cv2.distanceTransform(np.uint8(regions == 0), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.rint(255 * np.clip(1 - distance / (0.065 * density), 0, 1)).astype("u1")
    valid = warp(np.full(labels.shape, 255, np.uint8), cv2.INTER_LINEAR)
    if np.any((alpha > 0) & (valid != 255)):
        raise ValueError(f"{name}: decal contains unsupported source texels")
    color[alpha == 0] = 0
    Image.fromarray(np.dstack((color, alpha))).save(folder / "decal.png")
    Image.fromarray(regions).save(folder / "regions.png")
    restored = cv2.warpPerspective(
        regions, np.linalg.inv(image_to_atlas), (width, height), flags=cv2.INTER_NEAREST
    )
    iou = {}
    for k in (1, 2, 3):
        mask, back = labels == k, restored == k
        iou[k] = float((mask & back).sum() / (mask | back).sum())
    for u, v, wanted in row["probes"]:
        if labels[v, u] != wanted or restored[v, u] != wanted:
            raise ValueError(f"{name}: failed label probe {(u,v,wanted)}")
    if min(iou.values()) < 0.95:
        raise ValueError(f"{name}: round-trip masks {iou}")

    outlines = np.asarray(row["extent"])
    sweep = []
    for horizon_offset in (-8, 0, 8):
        for focal_factor in (0.9, 1.0, 1.1):
            horizon = row["horizon"] + horizon_offset
            focal = row["focal"] * focal_factor
            trial = road_projection(width, height, focal, horizon)
            span = np.ptp(transform(outlines, np.linalg.inv(trial)), axis=0)
            sweep.append(
                dict(horizon_px=horizon, focal_px=focal, span_camera_heights=span.tolist())
            )

    meta = dict(
        photo=row["photo"],
        source_sha256=row["sha256"],
        size_px=list(size),
        camera_height_units=1,
        units="one assumed source camera height",
        source_xy_to_atlas_xy=image_to_atlas.tolist(),
        atlas_xy_to_road_xy=np.linalg.inv(plane_to_atlas).tolist(),
        source_mask_roundtrip_iou=iou,
        projection_sensitivity=sweep,
        probe_count=len(row["probes"]),
    )
    (folder / "mapping.json").write_text(json.dumps(meta, indent=2) + "\n")
    raw_box = (int(xx.min() - 17), int(yy.min() - 12), int(xx.max() + 18), int(yy.max() + 13))
    annotated = panel(rgb, labels, raw_box)
    original = Image.fromarray(rgb).crop(raw_box)
    original.thumbnail((438, 310), Image.Resampling.LANCZOS)
    view = Image.new("RGB", (930, max(365, 53 + original.height + annotated.height)), (34, 36, 38))
    view.paste(original, (10, 36))
    view.paste(annotated, (10, 42 + original.height))
    texture = Image.fromarray(np.dstack((color, alpha))).convert("RGBA")
    texture.thumbnail((445, view.height - 43), Image.Resampling.LANCZOS)
    view.paste(texture, (470 + (445 - texture.width) // 2, 31), texture)
    ImageDraw.Draw(view).text((10, 9), name + " | photograph and traced regions", fill="white")
    ImageDraw.Draw(view).text((480, 9), "Road-plane decal | forward is up", fill="white")
    view.save(folder / "inspection.jpg", quality=94)
    print(name, size, "minimum round-trip IoU", round(min(iou.values()), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for key, value in json.loads((HERE / "traces.json").read_text()).items():
        build(key, value, args.output)
