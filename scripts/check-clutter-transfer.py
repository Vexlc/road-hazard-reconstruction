"""Package and verify road, fisheye and full-longitude paired-object exposures."""

import argparse
from importlib import import_module
from pathlib import Path

records = import_module("check-mei-pits")
PAIRS = ("R282-R283", "R284-R285")
PRODUCTS = (*records.PRODUCTS, "shadow.npy")
SITES = {
    "urban": ("image_00", "image_03"),
    "sun": ("image_00", "image_03"),
    "audi": ("front_left", "rear_center"),
}


def roads(models, root, packets, replay):
    cli = import_module("check-cone-cycles").cli
    report = {}
    if not replay:
        scenes = {
            "urban": "kitti360-0000",
            "sun": "kitti360-0000",
            "audi": "a2d2-frontleft",
        }
        records.pack(packets, root, PAIRS, scenes, PRODUCTS)
        return
    for pair in PAIRS:
        for site, sensors in SITES.items():
            base = packets / pair / site
            for sensor in sensors:
                d = base / sensor
                dst = root / "replay" / pair / site / sensor
                known = records.read(d / "record.json")
                cli(
                    "render",
                    models / pair,
                    base / "appearance.json",
                    d / "view.npz",
                    dst,
                    "--calibration",
                    d / known["calibration"],
                )
                actual = {key: records.digest(dst / key) for key in PRODUCTS}
                if actual != known["sha256"]:
                    raise ValueError(
                        f"Registered {site} {sensor} differs from the original"
                    )
                report[f"{pair}/{site}/{sensor}"] = actual
    records.save(root / "replay/check.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("pano-inputs", "pano-check", "pack", "replay", "seat-check")
    )
    for key in ("models", "root", "packets", "panoramas"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    if a.stage in ("pack", "replay"):
        roads(a.models, a.root, a.packets, a.stage == "replay")
    elif a.stage == "seat-check":
        from check_clutter_seating import verify

        dest = a.root / "seating"
        records.save(dest / "check.json", verify(a.models, dest, PAIRS))
    else:
        import clutter_panorama

        if a.stage == "pano-inputs":
            clutter_panorama.prepare(a.root / "panorama", a.panoramas, PAIRS)
        else:
            clutter_panorama.check(a.panoramas, a.root / "pano-check", a.models, PAIRS)
