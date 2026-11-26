"""Collect the three models and their verified road exposures into independently usable ZIPs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive


def package(models, views, drive, output):
    output.mkdir(parents=True, exist_ok=True)
    keys = [row["id"] for row in json.loads((HERE / "survey.json").read_text())["states"]]
    films = {"README.md": HERE / "README.md", "camera.json": HERE / "validation/camera.json"}
    archives = []
    for key in keys:
        folder = models / "scans/series/assets/pothole" / key
        entries = {
            key + "/" + name: folder / name
            for name in (
                "asset.glb",
                "asset.ply",
                "surface.npz",
                "metadata.json",
                "splat_material.npz",
            )
        }
        for suffix in ("near.jpg", "opposite.jpg", "sections.npz", "wheel.npz"):
            name = key + "-" + suffix
            entries["views/" + name] = views / name
        for name in (
            "README.md",
            "survey.json",
            "build.py",
            "wheel.py",
            "validation/quality.json",
            "validation/wheel.json",
            "validation/rebuild.json",
            "figures/wheel.png",
        ):
            entries[name] = HERE / name
        entries["references/" + key + ".jpg"] = models / "scans/series/references" / (key + ".jpg")
        path = output / (key + ".zip")
        archive(path, entries)
        archives.append(path)
        for side in ("front", "back"):
            for name in ("drive.mp4", "rays.npz", "part.png", "placement.json"):
                file = Path(side) / key / name
                films[str(file)] = drive / file
    path = output / "road-camera.zip"
    archive(path, films)
    archives.append(path)
    (output / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in sorted(archives)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "views", "drive", "output"):
        parser.add_argument(name, type=Path)
    package(**vars(parser.parse_args()))
