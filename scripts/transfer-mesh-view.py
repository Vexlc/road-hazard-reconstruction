"""Fit a local road placement or render a mesh into a metric camera packet from any host renderer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, fit_road, pixel_rays, resolve_samples
from road_geometry import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    fit = tasks.add_parser("fit", help="Fit an anchor from metric world-space road samples")
    fit.add_argument("points", type=Path, help="N × 3 .npy")
    fit.add_argument("out", type=Path)
    for name in ("seed", "heading", "up"):
        fit.add_argument("--" + name, nargs=3, type=float, required=True)
    fit.add_argument("--radius", type=float, required=True, help="metres")
    render = tasks.add_parser(
        "render", help="Return matched RGB, visible part IDs and asset hit range in metres"
    )
    render.add_argument("asset", type=Path, help="asset folder containing asset.glb and surface.npz")
    render.add_argument("anchor", type=Path)
    render.add_argument(
        "view", type=Path, help="npz: rgb, camera_to_world, host_range, protected and optionally rays_camera"
    )
    render.add_argument("out", type=Path)
    render.add_argument("--calibration", type=Path, help="JSON COLMAP camera row when rays_camera is omitted")
    args = parser.parse_args()
    if args.task == "fit":
        anchor = fit_road(
            np.load(args.points, allow_pickle=False),
            np.asarray(args.seed),
            np.asarray(args.heading),
            up_world=args.up,
            radius_m=args.radius,
        )
        save_json(args.out, anchor)
        return
    mesh = RoadMesh(args.asset)
    with np.load(args.view, allow_pickle=False) as view:
        if ("rays_camera" in view) == bool(args.calibration):
            parser.error("Provide exactly one lens definition: calibrated rays in the packet or --calibration")
        rays = (
            view["rays_camera"]
            if "rays_camera" in view
            else pixel_rays(json.loads(args.calibration.read_text()))
        )
        anchor = json.loads(args.anchor.read_text())
        if rays.ndim == 3:
            rays = rays[None]
        protected = view["protected"]
        if protected.ndim == 2:
            protected = np.broadcast_to(protected, rays.shape[:-1])
        if rays.ndim != 4 or not len(rays) or protected.shape != rays.shape[:-1]:
            parser.error("Expected H×W×3 rays or S×H×W×3 subpixel rays with compatible foreground masks")
        rgb, pose, host = view["rgb"], view["camera_to_world"], view["host_range"]
        after, ids, ranges = resolve_samples(
            [mesh.render(rgb, pose, subray, anchor, host, mask) for subray, mask in zip(rays, protected)]
        )
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(after).save(args.out / "after.png")
    Image.fromarray(ids).save(args.out / "part.png")
    np.save(args.out / "range.npy", ranges)
    save_json(args.out / "result.json", dict(mesh_sha256=mesh.digest, visible_pixels=int((ids > 0).sum())))


if __name__ == "__main__":
    main()
