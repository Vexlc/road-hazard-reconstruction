"""Fit a local road placement or render a mesh into a metric camera packet from any host renderer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, fit_footprint, fit_road, footprint_size, resolve_samples
from road_geometry import save_json
from prop_lighting import fit_receivers
from prop_registration import check_geometry, driving_clearance, registered_geometry
from host_depth import geometry, road_samples
from pavement_layout import PavementLayout, resolve_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    readout = tasks.add_parser("register-readout", help="Register scanline mid-exposures against a timed sensor trajectory")
    readout.add_argument("view", type=Path, help="Host packet with int64 row_time_ns")
    readout.add_argument("trajectory", type=Path, help="NPZ with time_ns and camera_to_world at each knot")
    readout.add_argument("out", type=Path)
    remap = tasks.add_parser("remap-camera", help="Transport a registered exposure through a nearest-pixel inverse map")
    remap.add_argument("view", type=Path)
    remap.add_argument("source_xy", type=Path, help="Hout×Wout×2 .npy map into original pixel centres (x, y)")
    remap.add_argument("out", type=Path)
    remap.add_argument("--calibration", type=Path, help="Original lens JSON if the input omits explicit rays")
    reflection = tasks.add_parser("reflection-mesh", help="Recover visible reflecting surfaces from labelled metric RGB-D")
    reflection.add_argument("view", type=Path)
    reflection.add_argument("out", type=Path)
    reflection.add_argument("--additional-view", type=Path, action="append", default=[], help="Another registered observation in the same world coordinates")
    reflection.add_argument("--calibration", type=Path)
    reflection.add_argument("--stride", type=int, default=2)
    reflection.add_argument("--max-edge-m", type=float, default=.35)
    road = tasks.add_parser("road-points", help="Unproject labelled metric road depth from calibrated host views")
    road.add_argument("out", type=Path, help="World-space XYZ .npy")
    road.add_argument("views", type=Path, nargs="+")
    road.add_argument("--calibration", type=Path, help="Shared lens calibration if packets omit explicit rays")
    road.add_argument("--stride", type=int, default=2, help="Pixel subsampling interval")
    fit = tasks.add_parser("fit", help="Fit an anchor from metric world-space road samples")
    fit.add_argument("points", type=Path, help="N × 3 .npy")
    fit.add_argument("out", type=Path)
    for name in ("seed", "heading", "up"):
        fit.add_argument("--" + name, nargs=3, type=float, required=True)
    fit.add_argument("--gravity-up", type=float, nargs=3, help="Gravity-opposing world direction for site-dependent water")
    fit.add_argument("--radius", type=float, required=True, help="metres")
    fit.add_argument("--asset", type=Path, help="Check ground support across this asset's entire contact footprint")
    fit.add_argument("--lighting", type=Path, help="Registered road RGB reference and optional world-space sun")
    fit.add_argument("--trajectory", type=Path, help="N × 3 .npy, measured vehicle centre trajectory in world metres")
    fit.add_argument("--vehicle-width", type=float, default=2.06, help="Full vehicle width in metres")
    fit.add_argument("--vehicle-length", type=float, default=5.10, help="Full vehicle length in metres")
    fit.add_argument("--clearance", type=float, default=.12, help="Minimum lateral clearance in metres")
    render = tasks.add_parser(
        "render", help="Return matched RGB, visible part IDs and asset hit range in metres"
    )
    render.add_argument("asset", type=Path, help="asset folder containing asset.glb and surface.npz")
    render.add_argument("anchor", type=Path)
    render.add_argument(
        "view", type=Path, help="npz: rgb, camera_to_world, host_range, protected and optionally rays_camera"
    )
    render.add_argument("out", type=Path)
    render.add_argument("--reflection-mesh", type=Path, help="Finite-distance host reflecting surfaces in world coordinates")
    render.add_argument("--calibration", type=Path, help="JSON COLMAP camera row when rays_camera is omitted")
    render.add_argument("--measured-rim", action="store_true", help="Use the scanned lip and adjoining continuous RGB-D pavement to resolve occlusion")
    render.add_argument("--road-tolerance-m", type=float, default=0.10,
                        help="Radial tolerance for uncertain host pavement; labelled occluders use their own range")
    color = tasks.add_parser(
        "calibrate-pavement",
        help="Measure pavement exposure outside the openings and store it in a fitted layout anchor",
    )
    color.add_argument("asset", type=Path, help="Pavement layout JSON")
    color.add_argument("anchor", type=Path)
    color.add_argument("out", type=Path)
    color.add_argument(
        "views",
        type=Path,
        nargs="+",
        help="Registered host exposures sharing one lighting and camera response",
    )
    color.add_argument("--calibration", type=Path)
    for command in (fit, render, color):
        command.add_argument("--asset-root", type=Path, help="Built members of a pavement layout JSON")
    args = parser.parse_args()
    if args.task == "reflection-mesh":
        from metric_reflection import MetricReflectors, from_depth
        members = []
        for exposure in (args.view, *args.additional_view):
            with np.load(exposure, allow_pickle=False) as view:
                members.append(from_depth(view, args.stride, args.max_edge_m, args.calibration))
        packet = MetricReflectors.merge(members).packet()
    if args.task == "register-readout":
        from camera_readout import interpolate_rows

        with np.load(args.view, allow_pickle=False) as data:
            packet = dict(data)
        with np.load(args.trajectory, allow_pickle=False) as traj:
            rows = interpolate_rows(traj["time_ns"], traj["camera_to_world"], packet["row_time_ns"])
        if rows.shape[-3] != packet["rgb"].shape[0]:
            parser.error("row_time_ns must match the image height")
        packet["camera_to_world_rows"] = rows
        packet.setdefault("camera_to_world", rows.reshape(-1, *rows.shape[-3:])[0, rows.shape[-3] // 2])
    if args.task == "remap-camera":
        from camera_remap import remap_packet

        with np.load(args.view, allow_pickle=False) as source:
            packet = remap_packet(source, np.load(args.source_xy, allow_pickle=False), args.calibration)
    if args.task in ("register-readout", "remap-camera", "reflection-mesh"):
        from tempfile import NamedTemporaryFile

        args.out.parent.mkdir(exist_ok=True, parents=True)
        with NamedTemporaryFile(dir=args.out.parent, delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with path.open("wb") as target:
                np.savez_compressed(target, **packet)
            path.replace(args.out)
        finally:
            path.unlink(missing_ok=True)
        return
    grouped = bool(getattr(args, "asset", None) and args.asset.is_file())
    if grouped and args.asset_root is None:
        parser.error("Provide --asset-root with a pavement layout recipe")
    layout = PavementLayout(args.asset, args.asset_root) if grouped else None
    if args.task == "calibrate-pavement":
        from pavement_appearance import apply_models, register_views

        if layout is None:
            parser.error("calibrate-pavement requires a layout recipe and its models")
        anchor = json.loads(args.anchor.read_text())
        models = register_views(layout, anchor, args.views, args.calibration)
        anchor.pop("pavement_appearance_by_camera", None)
        save_json(args.out, apply_models(anchor, models))
        return
    if args.task == "road-points":
        clouds = []
        for path in args.views:
            with np.load(path, allow_pickle=False) as packet:
                clouds.append(road_samples(packet, args.calibration, args.stride))
        points = np.concatenate(clouds)
        if not len(points):
            parser.error("No valid measured road samples in the supplied views")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb") as dest:
            np.save(dest, points)
        return
    if args.task == "fit":
        points = np.load(args.points, allow_pickle=False)
        anchor = fit_road(
            points,
            np.asarray(args.seed),
            np.asarray(args.heading),
            up_world=args.up,
            radius_m=args.radius,
        )
        if layout:
            if args.trajectory:
                parser.error("Vehicle envelope clearance is only supported for standing props")
            anchor = layout.fit(anchor, points, args.up, args.radius)
            if args.gravity_up or layout.water_states:
                from mesh_transfer import unit
                anchor["gravity_up_world"] = unit(args.gravity_up or args.up).tolist()
                layout.bind(anchor)
        elif args.asset:
            anchor = fit_footprint(anchor, points, footprint_size(args.asset))
            from cargo_support import check_members
            support = check_members(args.asset, anchor, points)
            if support:
                anchor["member_support"] = support
            with np.load(args.asset / "surface.npz", allow_pickle=False) as field:
                if "contact_mask" in field:
                    anchor["registered_geometry"] = [registered_geometry(args.asset)]
                    anchor["ground_receivers"] = fit_receivers(points, anchor)
        if args.trajectory:
            if not args.asset:
                parser.error("--trajectory requires the full geometry of --asset")
            mesh = RoadMesh(args.asset)
            anchor["vehicle_clearance"] = driving_clearance(
                anchor, mesh.bounds[:, :2], np.load(args.trajectory, allow_pickle=False),
                half_width_m=args.vehicle_width / 2, half_length_m=args.vehicle_length / 2,
                margin_m=args.clearance,
            )
        if args.lighting:
            anchor["lighting"] = json.loads(args.lighting.read_text())
        save_json(args.out, anchor)
        return
    mesh = layout if layout else RoadMesh(args.asset, measured_rim=args.measured_rim)
    with np.load(args.view, allow_pickle=False) as view:
        rays, host, protected, pose, origins = geometry(view, args.calibration, with_origins=True)
        from host_exposure import exposure
        rgb = view["rgb"]
        colors, weights = exposure(view, host.shape)
        anchor = json.loads(args.anchor.read_text())
        if "pavement_appearance_by_camera" in anchor:
            from pavement_appearance import camera_anchor

            anchor = camera_anchor(anchor, str(view["sensor_name"].item()) if "sensor_name" in view else None)
        if not layout:
            check_geometry(anchor, args.asset)
        foreground = view.get("occluder_range")
        if foreground is not None:
            if foreground.shape == rgb.shape[:2]:
                foreground = np.broadcast_to(foreground, rays.shape[:-1])
            if foreground.shape != rays.shape[:-1]:
                parser.error("Expected H×W or S×H×W radial metres for occluder_range")
        receivers = view.get("receiver_mask")
        if receivers is not None and receivers.shape == rgb.shape[:2]:
            receivers = np.broadcast_to(receivers, rays.shape[:-1])
        if receivers is not None and (receivers.shape != rays.shape[:-1] or receivers.dtype != bool):
            parser.error("Expected boolean H×W or S×H×W registered receiver_mask")
        if layout:
            layout.bind(anchor)
            from pavement_water import DirectionalProbe
            from metric_reflection import MetricReflectors
            reflection_probe = DirectionalProbe.from_packet(view)
            if args.reflection_mesh:
                if any(k in view for k in MetricReflectors.fields):
                    parser.error("Choose embedded or separate reflection geometry")
                with np.load(args.reflection_mesh, allow_pickle=False) as source:
                    reflectors = MetricReflectors.from_packet(source)
                    if reflectors is None:
                        parser.error("Missing metric reflector geometry in the external file")
            else:
                reflectors = MetricReflectors.from_packet(view)
        elif args.reflection_mesh:
            parser.error("A reflection mesh requires a pavement layout")
        receiver_band = view.get("receiver_band_m")
        if receiver_band is not None:
            if receiver_band.shape != () or layout or not mesh.standing:
                parser.error("receiver_band_m must be a scalar for an above-road asset")
            receiver_band = float(receiver_band)
        samples = []
        for i, (subray, mask) in enumerate(zip(rays, protected)):
            incident = rgb if colors is None else colors[i]
            options = dict(ray_origins_camera=None if origins is None else origins[i],
                           occluder_range=None if foreground is None else foreground[i],
                           road_tolerance_m=args.road_tolerance_m,
                           receiver_mask=None if receivers is None else receivers[i])
            if layout:
                sample = layout.sample(incident, pose, subray, anchor, host[i], mask, reflection_probe=reflection_probe, reflectors=reflectors, pixel_width=len(rays)**-0.5, **options)
            else:
                options["receiver_band_m"] = receiver_band
                options["instance_ids"] = anchor.get("member_support", {}).get("geometry_instances")
                if mesh.normal_textures:
                    options["pixel_width"] = len(rays)**-.5
                sample = mesh.render(incident, pose, subray, anchor, host[i], mask, with_shadow=mesh.standing, **options)
            samples.append(sample)
        after, ids, ranges, *shadow = resolve_samples([s[:3] for s in samples] if layout else [s[:4] for s in samples],
            host_rgb=None if colors is None else rgb, weights=weights)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(after).save(args.out / "after.png")
    Image.fromarray(ids).save(args.out / "part.png")
    np.save(args.out / "range.npy", ranges)
    if layout:
        Image.fromarray(resolve_instances(samples)).save(args.out / "instance.png")
    elif "member_support" in anchor:
        instance = resolve_instances([(s[0], s[1], s[2], s[4]) for s in samples])
        Image.fromarray(instance).save(args.out / "instance.png")
    if shadow:
        np.save(args.out / "shadow.npy", shadow[0])
    save_json(args.out / "result.json", dict(mesh_sha256=mesh.digest, visible_pixels=int((ids > 0).sum())))


if __name__ == "__main__":
    main()
