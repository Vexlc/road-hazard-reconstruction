"""Compare native replays inside the surveyed openings and beyond their projected support."""

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "streetcrafter/joint"))
from pit_pairs import Footprint, RIG, load_rig, rear_mount

sys.path.insert(0, str(ROOT / "gen2hole/downstream/sensors"))
from road_geometry import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=ROOT / "outputs/measured-pavement/joint")
    p.add_argument("--whole", type=Path, help="Optional replays with whole-particle removal")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/measured-pavement")
    p.add_argument("--anchor", type=Path, default=ROOT / "outputs/measured-pavement/kitti/anchor.json")
    p.add_argument("--pair", help="Rendered layout ID when it differs from the ordered asset IDs")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    seats = json.loads(a.anchor.read_text())["instances"]
    pair = a.pair or "-".join(s["asset"] for s in seats)
    edges = []
    for s in seats:
        k = s["asset"]
        matches = list(a.assets.glob(f"*/series/assets/*/{k}/surface.npz"))
        f = matches[0] if matches else a.assets / k / "surface.npz"
        foot = Footprint(f)
        b = np.array(s["anchor"]["basis_columns"])
        edges.append(foot.edge @ b[:, :2].T + s["anchor"]["center_world"])
    split, poses = load_rig(RIG)
    offset, rear = rear_mount()
    out = {}
    for camera in ["front", "rear"]:
        sources = {"split": a.views}
        if a.whole:
            sources["whole"] = a.whole
        stats = {
            k: dict(inside_samples=0, inside_changed=0, outside_changed=0, outside_samples=0, frames=0)
            for k in sources
        }
        for pth in sorted((a.views / "original" / camera).glob("*.png")):
            b = np.asarray(Image.open(pth), dtype=int)
            h, w = b.shape[:2]
            tr = poses[pth.name]
            cal = split["projections"]["eval"][split["views"][pth.name]["projection"]]
            if camera == "rear":
                tr = np.linalg.inv(offset) @ tr
                cal = rear
            _, width, height, fx, fy, cx, cy = cal
            assert (width, height) == (w, h)
            mask = np.zeros((h, w), "u1")
            # Skip partially near-clipped hulls, whose ordinary pinhole projection is undefined.
            points = [edge @ tr[:3, :3].T + tr[:3, 3] for edge in edges]
            if min(x[:, 2].min() for x in points) < 0.3:
                continue
            for xyz in points:
                uv = xyz[:, :2] / xyz[:, 2:] * [fx, fy] + [cx, cy]
                cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype("i4")), 1)
            if mask.sum() < 60:
                continue
            exterior = cv2.dilate(mask, np.ones((21, 21), "u1")) == 0
            for k, s in sources.items():
                img = np.asarray(Image.open(s / pair / camera / pth.name), dtype=int)
                changed = np.max(abs(img - b), axis=2) > 8
                stats[k]["outside_changed"] += int((exterior & changed).sum())
                stats[k]["outside_samples"] += int(exterior.sum())
                stats[k]["inside_changed"] += int((mask.astype(bool) & changed).sum())
                stats[k]["inside_samples"] += int(mask.sum())
                stats[k]["frames"] += 1
        for value in stats.values():
            if not value["frames"] or value["inside_changed"] < 500:
                raise ValueError("Missing visible repair")
        if a.whole and stats["split"]["outside_changed"] >= stats["whole"]["outside_changed"]:
            raise ValueError(f"Splitting failed to preserve more of the original {camera} view")
        out[camera] = stats
    print(json.dumps(out))
    save_json(a.out, out)


if __name__ == "__main__":
    main()
