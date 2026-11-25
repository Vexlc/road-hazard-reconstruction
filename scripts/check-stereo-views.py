"""Compare native Gaussian renders of the historical stereo cases from three identical cameras."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from atlas_gaussians import ROOT, digest, geometry_frame, load, render


def compare(row, reference, rebuilt):
    original = dict(row, glb=reference / row["glb"], ply=reference / row["ply"])
    if (
        digest(original["ply"]) != row["ply_sha256"]
        or digest(original["glb"]) != row["glb_sha256"]
    ):
        raise ValueError(
            f"Reference model differs from the published manifest: {row['id']}"
        )
    # Each independent load uses its own count. An insertion at the rim must
    # be visible to the comparison rather than silently truncating either PLY.
    candidate = dict(row, ply=rebuilt / row["ply"])
    metadata = json.loads((rebuilt / row["metadata"]).read_text())
    candidate["gaussians"] = metadata["gaussians"]
    before, after = load(original), load(candidate)
    width, height = 640, 400
    base = geometry_frame(dict(original, category="坑洞"), width / height)
    views = []
    for azimuth, tilt in [(-18, 42), (115, 68), (218, 54)]:
        camera = dict(
            base,
            theta=np.deg2rad(azimuth),
            phi=np.deg2rad(tilt),
            radius=base["radius"] * 1.15,
        )
        first, alpha, depth = render(before, camera, width, height)
        second, other_alpha, other_depth = render(after, camera, width, height)
        solid = (alpha > 0.5) & (other_alpha > 0.5)
        if solid.sum() < 1000:
            raise ValueError(f"Insufficient visible foreground: {row['id']}")
        rgb_delta = np.abs(first.astype(np.int16) - second)
        views.append(
            dict(
                azimuth_deg=azimuth,
                polar_deg=tilt,
                common_foreground_pixels=int(solid.sum()),
                rgb_max_u8=int(rgb_delta.max()),
                changed_rgb_pixels=int(np.any(rgb_delta, axis=-1).sum()),
                alpha_max=float(np.abs(alpha - other_alpha).max()),
                foreground_depth_max_m=float(
                    np.abs(depth[solid] - other_depth[solid]).max()
                ),
            )
        )
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        help="Check a published stereo ID; repeat to select several",
    )
    args = parser.parse_args()
    cases = json.loads(
        (ROOT / "validation/stereo-reconstruction-differences.json").read_text()
    )
    rows = {
        r["id"]: r
        for r in json.loads(
            (ROOT / "outputs/road_hazard_expansion/asset_manifest.json").read_text()
        )
    }
    selected = args.asset or [item["id"] for item in cases]
    if any(key not in rows or rows[key]["category"] != "stereo" for key in selected):
        parser.error("Expected a published stereo asset ID")
    torch.set_num_threads(2)
    result = {
        key: compare(rows[key], args.reference.resolve(), args.rebuilt.resolve())
        for key in selected
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(
        r["changed_rgb_pixels"] or r["alpha_max"] or r["foreground_depth_max_m"]
        for views in result.values()
        for r in views
    ):
        raise ValueError(f"Replayed samples differ; see {args.report}")
    print(
        f"{len(result)} stereo assets, {sum(map(len, result.values()))} matching native Gaussian views"
    )


if __name__ == "__main__":
    main()
