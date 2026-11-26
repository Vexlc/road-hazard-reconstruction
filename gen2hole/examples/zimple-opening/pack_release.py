"""Package the M582 meshes, splats and sensor replay with verified inner hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive


def pack(models, views, drive, lidar, out):
    out.mkdir(parents=True, exist_ok=True)
    folder = models / "model/series/assets/missing/M582"
    hashes = json.loads((HERE / "validation/rebuild.json").read_text())
    entries = {}
    for name, expected in hashes.items():
        path = folder / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Unexpected model fingerprint " + name)
        entries["M582/" + name] = path
    for f in sorted(views.glob("*.jpg")):
        entries["views/" + f.name] = f
    for f in HERE.rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts:
            entries[str(f.relative_to(HERE))] = f
    archive(out / "M582.zip", entries)
    clips = {"README.md": HERE / "README.md", "camera.json": HERE / "validation/camera.json"}
    for side in ("front", "back"):
        for name in ("drive.mp4", "placement.json", "rays.npz", "part.png"):
            clips[side + "/" + name] = drive / side / "M582" / name
    for f in sorted(lidar.glob("*.npz")):
        clips["lidar/" + f.name] = f
    clips["lidar/report.json"] = lidar / "report.json"
    archive(out / "camera-lidar.zip", clips)
    (out / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256((out / n).read_bytes()).hexdigest() + "  " + n + "\n"
            for n in ("M582.zip", "camera-lidar.zip")
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "views", "drive", "lidar", "out"):
        p.add_argument(name, type=Path)
    pack(**vars(p.parse_args()))
