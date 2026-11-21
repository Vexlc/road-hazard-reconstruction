"""Construct the textured mesh and insertion field from fitted image coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

import chambers
import pavement
from mesh_io import Mesh, ROOT, intact_pavement, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--edition", choices=["pilot", "series"], required=True)
    parser.add_argument("keys", nargs="+")
    args = parser.parse_args()
    layout = json.loads((ROOT / "calibration/layout.json").read_text())
    if set(args.keys) - layout.keys():
        raise ValueError(f"Missing final fits: {set(args.keys) - layout.keys()}")
    args.cache.mkdir(parents=True, exist_ok=True)
    for ordinal, (key, cfg) in enumerate(layout.items()):
        if key not in args.keys:
            continue
        rgb = source(key)
        if rgb.shape[0] != rgb.shape[1]:
            raise ValueError(f"Expected a square nadir image: {key}")
        mesh = Mesh()
        grain = intact_pavement(rgb, key, cfg)
        model_id = cfg["id"]
        Image.fromarray(grain).save(args.cache / (model_id + "_ground.jpg"), quality=96, subsampling=0)
        if cfg.get("depth_profile") == "masked_groove":
            from crack_surface import build as build_crack
            field, fit = build_crack(mesh,key,cfg)
        elif cfg["category"] == "pothole":
            # The pilot used the source filename as its material/model subtype.
            field, fit = pavement.build(mesh, key, dict(model=key) | cfg, grain)
        else:
            field, fit = chambers.build(mesh, key, cfg, layout, grain)
        if args.edition == "pilot":
            recipe, seed = "image-fit-v1", 311720 + ordinal
        else:
            recipe, seed = "image-fit-100-v2", 811720 + ord(key[0]) * 100 + int(key[1:])
        ref = Path("references") / (key + ".png")
        meta = dict(id=model_id, category=cfg["category"], label=cfg["label"],
                    family=cfg.get("family", cfg.get("shape")), source_image=str(ref),
                    source_sha256=hashlib.sha256((ROOT / ref).read_bytes()).hexdigest(),
                    recipe_version=recipe, scale_kind="designed_metric", seed=seed, **fit)
        mesh.save(ROOT / "assets" / cfg["category"] / model_id, args.cache, meta, field)
        print(f"{model_id}: {sum(meta['geometry_parts'].values()):,} triangles", flush=True)


if __name__ == "__main__":
    main()
