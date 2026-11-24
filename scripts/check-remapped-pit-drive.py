"""Replay the Audi rear exposure using its four subrays after rotating and cropping the raster."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

rig = import_module("check-cone-cycles")
from road_geometry import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("views", type=Path)
    p.add_argument("models", type=Path)
    a = p.parse_args()
    site = a.views / "audi"
    raw = site / "P250-P251/rear_center"
    result = a.views / "audi-remapped-rear"
    result.mkdir(parents=True, exist_ok=True)
    expected = np.asarray(Image.open(raw / "instance.png"))
    yy, xx = np.nonzero(expected)
    # The crop surrounds both assets. Rotate its raster clockwise about the pixel centres.
    y, x = np.mgrid[max(yy.min()-24, 0):min(yy.max()+25, expected.shape[0]),
                    max(xx.min()-24, 0):min(xx.max()+25, expected.shape[1])]
    y, x = (np.rot90(arr, k=-1) for arr in (y, x))
    np.save(result / "source_xy.npy", np.stack((x, y), -1))
    file = result / "camera.npz"
    rig.cli("remap-camera", raw / "view.npz", result / "source_xy.npy", file)
    rig.cli("render", rig.ROOT / "gen2hole/examples/remapped-pits/layout.json", site / "anchor.json", file, result, "--asset-root", a.models)
    for name in ("part", "instance"):
        original, rotated = (np.asarray(Image.open(folder / (name + ".png"))) for folder in (raw, result))
        np.testing.assert_array_equal(rotated, original[y, x])
    np.testing.assert_array_equal(np.load(result / "range.npy"), np.load(raw / "range.npy")[y, x])
    before = np.array(Image.open(raw / "after.png"))[y, x].astype(int)
    delta = abs(np.asarray(Image.open(result / "after.png")).astype(int) - before)
    asset = expected[y, x] > 0
    if np.any(delta[~asset]) or np.percentile(delta[asset], 99) > 2:
        raise ValueError("Unexpected change to the remapped exposure")
    save_json(result / "check.json", dict(pixels=int(asset.size), asset_pixels=int(asset.sum()),
              matched_instance_part_range=True, max_rgb_delta=int(delta.max()),
              asset_p99_rgb_delta=float(np.percentile(delta[asset], 99))))


if __name__ == "__main__":
    main()
