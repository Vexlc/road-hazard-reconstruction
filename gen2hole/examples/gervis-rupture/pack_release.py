"""Package the paired geometry, field evidence and road replay with file digests."""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive


def pack(models, views, drive, lidar, out):
    out.mkdir(parents=True, exist_ok=True)
    folder = models / "model/series/assets/pothole/P581"
    entries = {
        "P581/" + name: folder / name
        for name in (
            "asset.glb",
            "inspection.glb",
            "asset.ply",
            "metadata.json",
            "splat_material.npz",
            "surface.npz",
        )
    }
    for name in (
        "source-rays.npz",
        "sections.npz",
        "P581-near.jpg",
        "P581-opposite.jpg",
        "source-view.png",
        "transfer-view.png",
    ):
        entries["views/" + name] = views / name
    for p in HERE.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            entries[str(p.relative_to(HERE))] = p
    archive(out / "P581.zip", entries)
    clips = {"README.md": HERE / "README.md", "camera.json": HERE / "validation/camera.json"}
    for side in ("front", "back"):
        for name in ("drive.mp4", "placement.json", "rays.npz", "part.png"):
            clips[side + "/" + name] = drive / side / "P581" / name
    for f in sorted(lidar.glob("*.npz")):
        clips["lidar/" + f.name] = f
    clips["lidar/report.json"] = lidar / "report.json"
    archive(out / "camera-lidar.zip", clips)
    (out / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256((out / n).read_bytes()).hexdigest() + "  " + n + "\n"
            for n in ("P581.zip", "camera-lidar.zip")
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "views", "drive", "lidar", "out"):
        p.add_argument(k, type=Path)
    pack(**vars(p.parse_args()))
