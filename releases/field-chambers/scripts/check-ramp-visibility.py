"""Track the same pit surface before and after passing, including banks hidden by the opposite rim."""

from importlib import import_module
from pathlib import Path
import argparse
import numpy as np
from road_geometry import Surface, save_json, sha256

checker = import_module("check-pavement-pass")


def check(run, raw, views, out):
    results = {}
    for placement in sorted((views / "front").glob("*/placement.json")):
        key = placement.parent.name
        (glb,) = (run / "series/assets").glob(f"*/{key}/asset.glb")
        h = sha256(glb)
        mesh = Surface(glb)
        a, b = [
            checker.exposure(views / side / key, raw, mesh, h)
            for side in ("front", "back")
        ]
        if not np.array_equal(a["matrix"], b["matrix"]) or a["time"] >= b["time"]:
            raise ValueError("Camera placement mismatch")
        entry = dict(
            glb_sha256=h,
            scene=raw.name,
            front_frame=a["frame"],
            back_frame=b["frame"],
            front_range_error_mm=1000 * a["error_m"],
            back_range_error_mm=1000 * b["error_m"],
        )
        dest = out / key
        dest.mkdir(exist_ok=True, parents=True)
        for name, src, target in [("front_to_back", a, b), ("back_to_front", b, a)]:
            flow = checker.motion.flow(src, target, mesh, target["lens"])
            uv = flow["target_uv"]
            xyz = src["xyz"]
            M = target["matrix"]
            eye = (target["pose"][:3, 3] - M[:3, 3]) @ M[:3, :3]
            delta = xyz - eye
            length = np.linalg.norm(delta, axis=1)
            first, block_part, _ = mesh.cast(
                np.broadcast_to(eye, xyz.shape), delta / length[:, None]
            )
            gap = length - first
            # A reverse cast tangent to a bank can pass just behind a known
            # source hit. Keep those samples separate from proved occlusions.
            grazing = (gap <= -0.0002) | ~np.isfinite(first)
            rc = np.rint(uv - target["offset"]).astype(int)
            part = target["parts"]
            inside = (rc >= 0).all(1) & (rc < part.shape[::-1]).all(1)
            is_foreground = np.zeros(len(rc), bool)
            is_foreground[inside] = part[rc[inside, 1], rc[inside, 0]] == 0
            reason = np.zeros(len(xyz), "u1")
            reason[is_foreground] = 1
            reason[~inside] = 2
            reason[gap >= 0.0002] = 3
            reason[grazing] = 4
            np.testing.assert_array_equal(reason == 0, flow["target_visible"])
            fn = dest / (name + ".npz")
            np.savez_compressed(
                fn,
                **flow,
                visibility_code=reason,
                intervening_distance_m=np.where(grazing, np.nan, np.maximum(0, gap)),
                first_part=block_part,
            )
            counts = np.bincount(reason, minlength=5)
            entry[name] = dict(
                samples=len(reason),
                visible=int(counts[0]),
                target_background=int(counts[1]),
                outside_crop=int(counts[2]),
                self_occluded=int(counts[3]),
                unresolved_grazing=int(counts[4]),
                packet_sha256=sha256(fn),
            )
            if counts[3]:
                entry[name]["occluded_gap_median_mm"] = float(
                    1000 * np.median(gap[reason == 3])
                )
        results[key] = entry
        print(
            key,
            entry["front_to_back"]["visible"],
            entry["front_to_back"]["self_occluded"],
            flush=True,
        )
    save_json(out / "summary.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("run", "raw", "views", "out"):
        p.add_argument(n, type=Path)
    check(**vars(p.parse_args()))
