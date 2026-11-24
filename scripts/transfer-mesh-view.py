"""Fit a local road placement or render a mesh into a metric camera packet from any host renderer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, fit_footprint, fit_road, footprint_size, pixel_rays, resolve_samples
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
    fit.add_argument("--asset", type=Path, help="Check ground support across this asset's entire contact footprint")
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
    render.add_argument("--road-tolerance-m", type=float, default=0.10,
                        help="Radial tolerance for uncertain host pavement; labelled occluders use their own range")
    args = parser.parse_args()
    if args.task == "fit":
        points = np.load(args.points, allow_pickle=False)
        anchor = fit_road(
            points,
            np.asarray(args.seed),
            np.asarray(args.heading),
            up_world=args.up,
            radius_m=args.radius,
        )
        if args.asset:
            anchor = fit_footprint(anchor, points, footprint_size(args.asset))
        save_json(args.out, anchor)
        return
    mesh = RoadMesh(args.asset)
    with np.load(args.view, allow_pickle=False) as view:
        if ("rays_camera" in view) == bool(args.calibration):
            parser.error("Provide exactly one lens definition: calibrated rays in the packet or --calibration")
        protected = view["protected"]
        if "rays_camera" in view:
            rays = view["rays_camera"]
        else:
            rays, valid = pixel_rays(json.loads(args.calibration.read_text()), return_valid=True)
            protected = protected | ~valid
        anchor = json.loads(args.anchor.read_text())
        if rays.ndim == 3:
            rays = rays[None]
        if protected.ndim == 2:
            protected = np.broadcast_to(protected, rays.shape[:-1])
        if rays.ndim != 4 or not len(rays) or protected.shape != rays.shape[:-1]:
            parser.error("Expected H×W×3 rays or S×H×W×3 subpixel rays with compatible foreground masks")
        rgb, pose, host = view["rgb"], view["camera_to_world"], view["host_range"]
        foreground = view.get("occluder_range")
        if foreground is not None:
            if foreground.shape == rgb.shape[:2]:
                foreground = np.broadcast_to(foreground, rays.shape[:-1])
            if foreground.shape != rays.shape[:-1]:
                parser.error("Expected H×W or S×H×W radial metres for occluder_range")
        after, ids, ranges = resolve_samples(
            [mesh.render(rgb, pose, subray, anchor, host, mask,
                         occluder_range=None if foreground is None else foreground[i],
                         road_tolerance_m=args.road_tolerance_m)
             for i, (subray, mask) in enumerate(zip(rays, protected))]
        )
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(after).save(args.out / "after.png")
    Image.fromarray(ids).save(args.out / "part.png")
    np.save(args.out / "range.npy", ranges)
    save_json(args.out / "result.json", dict(mesh_sha256=mesh.digest, visible_pixels=int((ids > 0).sum())))


if __name__ == "__main__":
    main()
