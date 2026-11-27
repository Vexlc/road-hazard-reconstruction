"""Package each verified state and the matched camera/LiDAR comparisons with checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive


def pack(models, views, drive, lidar, out):
    states = json.loads((HERE / "survey.json").read_text())["relative_grate_poses"]
    expected = json.loads((HERE / "validation/rebuild.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for name, sha in expected.items():
        if hashlib.sha256((models / name).read_bytes()).hexdigest() != sha:
            raise ValueError(name)
    guides = {
        str(p.relative_to(HERE)): p
        for p in HERE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    clips = {
        "README.md": HERE / "README.md",
        "camera.json": HERE / "validation/camera.json",
        "packet-check.json": HERE / "validation/packets.json",
    }
    bundles = []
    for state in states:
        files = dict(guides)
        for rel in expected:
            if "/" not in rel or rel.startswith(state + "/"):
                files["models/" + rel] = models / rel
        for manifest in ("model/run.json", "model/series/assets/asset_manifest.json"):
            files["models/" + state + "/" + manifest] = models / state / manifest
        for name in ("M589-near.jpg", "M589-opposite.jpg", "quality.json"):
            files["views/" + name] = views / state / name
        path = out / ("M589-" + state + ".zip")
        archive(path, files)
        bundles.append(path)
        for side in ("front", "back"):
            for name in (
                "drive.mp4",
                "placement.json",
                "before.png",
                "after.png",
                "rays.npz",
                "part.png",
            ):
                p = Path(state) / side / "M589" / name
                clips[str(p)] = drive / p
    for p in sorted(lidar.glob("*.npz")):
        clips["lidar/" + p.name] = p
    clips["lidar/report.json"] = lidar / "report.json"
    path = out / "camera-lidar.zip"
    archive(path, clips)
    bundles.append(path)
    (out / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in sorted(bundles)
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("models", "views", "drive", "lidar", "out"):
        p.add_argument(n, type=Path)
    pack(**vars(p.parse_args()))
