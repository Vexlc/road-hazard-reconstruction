"""Compare two road-depth policies at manually labelled asphalt and post pixels in an original exposure."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, resolve_samples
from prop_registration import check_geometry
from road_geometry import save_json


def polygon(shape, vertices):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(vertices)], 1)
    return mask.astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/road-debris")
    parser.add_argument("--assets", type=Path, default=ROOT / "runs/road-debris")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/road-debris/visibility.json")
    args = parser.parse_args()
    recipe = json.loads((ROOT / "gen2hole/examples/road-debris/visibility.json").read_text())
    pair = recipe["pair"]
    site = args.views / pair / recipe["site"]
    folder = site / pair / recipe["camera"]
    registered = json.loads((site / "index.json").read_text())[pair]["views"][recipe["camera"]]
    if any(registered[key] != recipe[key] for key in ("frame", "crop_xy", "source_rgb_sha256")):
        raise ValueError("Labelled KITTI exposure or crop differs")
    if hashlib.sha256((folder / "before.png").read_bytes()).hexdigest() != recipe["before_png_sha256"]:
        raise ValueError("Labelled original crop differs")
    anchor = json.loads((site / "anchor.json").read_text())
    mesh = RoadMesh(args.assets / pair)
    check_geometry(anchor, args.assets / pair)
    with np.load(folder / "view.npz", allow_pickle=False) as view:
        image = view["rgb"]
        if not np.array_equal(image, np.asarray(Image.open(folder / "before.png"))):
            raise ValueError("Camera packet RGB differs from the labelled photograph")
        shape = image.shape[:2]
        road = polygon(shape, recipe["asphalt_xy"])
        post = polygon(shape, recipe["post_xy"])
        controls = {}
        for name, host in [("unobstructed", np.full(shape, np.inf, "f4")), ("radial", view["host_range"])]:
            samples = [
                mesh.render(
                    image,
                    view["camera_to_world"],
                    r,
                    anchor,
                    host,
                    protected,
                    with_shadow=True,
                    pixel_width=len(view["rays_camera"]) ** -0.5,
                )
                for r, protected in zip(view["rays_camera"], view["protected"])
            ]
            controls[name] = resolve_samples(samples)
    _, ref_ids, _, ref_loss = controls["unobstructed"]
    strong = road & (ref_ids == 0) & (ref_loss > 0.3)
    post_cross = post & ((ref_ids > 0) | (ref_loss >= 0.004))
    after = np.array(Image.open(folder / "after.png"))
    final_loss = np.load(folder / "shadow.npy")
    missing = lambda loss: int(np.count_nonzero(strong & (loss < ref_loss * 0.5)))
    out = {
        "road_polygon_pixels": int(road.sum()),
        "strong_shadow_pixels": int(strong.sum()),
        "radial_missing": missing(controls["radial"][3]),
        "surveyed_missing": missing(final_loss),
        "post_pixels": int(post.sum()),
        "projection_crossing_post": int(post_cross.sum()),
        "changed_post_pixels": int((post & np.any(after != image, axis=2)).sum()),
        "post_max_srgb_delta": int(abs(after.astype(int) - image.astype(int))[post].max()),
        "post_asset_pixels": int(np.count_nonzero(post & (np.asarray(Image.open(folder / "part.png")) > 0))),
    }
    print(out)
    if (
        out["strong_shadow_pixels"] < 50
        or out["surveyed_missing"] >= out["radial_missing"]
        or out["projection_crossing_post"] < 1
        or out["post_max_srgb_delta"] > 2
        or out["post_asset_pixels"] != 0
    ):
        raise ValueError("Registered visibility check failed")
    save_json(args.out, out)


if __name__ == "__main__":
    main()
