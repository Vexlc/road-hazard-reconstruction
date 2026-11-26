"""Package the five assets, reproducible field checks and ten separate driving views."""

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVER = HERE.parent / "hensting-cover"
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive as write_archive


def files(base):
    return {
        str(p.relative_to(base)): p
        for p in base.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def pack(models, outputs, out):
    out.mkdir(parents=True, exist_ok=True)
    entries = files(COVER)
    run = models / "cover"
    for p in (run / "series/assets/raised/A565").iterdir():
        entries["A565/" + p.name] = p
    for name in ("top-source-weight.png", "top-atlas.png"):
        entries[name] = run / name
    for name in ("source-view.png", "A565-near.jpg", "A565-opposite.jpg"):
        entries["views/" + name] = outputs / "views" / name
    write_archive(out / "A565.zip", entries)
    entries = files(HERE)
    for key in ("P566", "P567", "P568", "P569"):
        folder = models / "scans/series/assets/pothole" / key
        for name in (
            "asset.glb",
            "asset.ply",
            "metadata.json",
            "surface.npz",
            "splat_material.npz",
        ):
            entries[key + "/" + name] = folder / name
        for suffix in ("near.jpg", "opposite.jpg", "sections.npz"):
            name = key + "-" + suffix
            entries[key + "/" + name] = outputs / "scans" / name
    write_archive(out / "wheel-track-pits.zip", entries)
    cameras = {
        "cover-guide.md": COVER / "README.md",
        "pits-guide.md": HERE / "README.md",
        "camera.json": outputs / "camera.json",
    }
    for side in ("front", "back"):
        for key in ("A565", "P566", "P567", "P568", "P569"):
            folder = outputs / "drive" / side / key
            for name in ("drive.mp4", "placement.json", "part.png", "rays.npz"):
                cameras[f"{side}/{key}/{name}"] = folder / name
    write_archive(out / "road-camera.zip", cameras)
    names = ("A565.zip", "wheel-track-pits.zip", "road-camera.zip")
    (out / "SHA256SUMS").write_text(
        "".join(hashlib.sha256((out / n).read_bytes()).hexdigest() + "  " + n + "\n" for n in names)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("models", "outputs", "out"):
        parser.add_argument(arg, type=Path)
    pack(**vars(parser.parse_args()))
