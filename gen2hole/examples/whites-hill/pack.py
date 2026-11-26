"""Bundle the two observation contexts and timestamped sensor replay packets."""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive


def package(models, views, release):
    release.mkdir(parents=True, exist_ok=True)
    folder = models / "model/series/assets/missing/M576"
    names = (
        "asset.glb",
        "inspection.glb",
        "asset.ply",
        "metadata.json",
        "surface.npz",
        "splat_material.npz",
    )
    assets = {"M576/" + n: folder / n for n in names}
    assets.update(
        {
            str(f.relative_to(HERE)): f
            for f in HERE.rglob("*")
            if f.is_file() and "__pycache__" not in f.parts
        }
    )
    assets.update({"views/" + f.name: f for f in (views / "views").iterdir() if f.is_file()})
    archive(release / "M576.zip", assets)
    packets = {"README.md": HERE / "README.md", "camera.json": views / "camera.json"}
    for side in ["front", "back"]:
        for name in ["drive.mp4", "placement.json", "part.png", "rays.npz"]:
            packets[side + "/" + name] = views / "drive" / side / "M576" / name
    packets.update({"lidar/" + p.name: p for p in (views / "lidar").iterdir() if p.is_file()})
    archive(release / "camera-lidar.zip", packets)
    (release / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256((release / n).read_bytes()).hexdigest() + "  " + n + "\n"
            for n in ["M576.zip", "camera-lidar.zip"]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["models", "views", "release"]:
        parser.add_argument(key, type=Path)
    package(**vars(parser.parse_args()))
