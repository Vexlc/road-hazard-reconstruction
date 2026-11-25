"""Check directional illumination in two ported openings against an independent Blender BVH."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))


EXAMPLES = ROOT / "gen2hole/examples/sun-openings"
PAIRS = ("P276-A277", "P278-M277")
SITES = {"kitti": "kitti360-0000", "audi": "a2d2-frontleft"}


def asset_path(root, key):
    direct = root / key / "asset.glb"
    if direct.is_file():
        return direct
    (path,) = root.glob(f"*/series/assets/*/{key}/asset.glb")
    return path


def oracle(path, dest, sun):
    import bpy  # Initializes Blender's mathutils runtime.
    from mathutils.bvhtree import BVHTree
    import trimesh

    scene = trimesh.load_scene(path, process=False)
    verts, faces = [], []
    for node in sorted(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        p = trimesh.transform_points(mesh.vertices, transform)
        faces.extend(np.asarray(mesh.faces) + len(verts))
        verts.extend(np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1))
    bvh = BVHTree.FromPolygons(verts, faces, all_triangles=True)
    # A horizontal opaque canopy at z=0.8 shades half the road at grade.
    canopy_x = 0.8 * sun[0] / sun[2]
    east = np.cross(sun, [0, 1, 0])
    east /= np.linalg.norm(east)
    north = np.cross(sun, east)
    disk = [
        sun,
        sun + 0.012 * east,
        sun - 0.012 * east,
        sun + 0.012 * north,
        sun - 0.012 * north,
    ]
    samples = []
    axis = np.linspace(-0.36, 0.36, 83)
    for dx in (0.0, 0.48):
        for x in axis:
            for y in axis:
                look = np.array([dx, 0.12, -1.0])
                pos, normal, _, _ = bvh.ray_cast([x - dx, y - 0.12, 1.0], look)
                if pos is None:
                    continue
                q, n = np.array(pos), np.array(normal)
                if np.linalg.norm(q[:2]) > 0.34:
                    continue
                if n @ look > 0:
                    n = -n
                crossing_x = q[0] + (0.8 - q[2]) * sun[0] / sun[2]
                if abs(crossing_x - canopy_x) < 0.17:
                    continue  # The measured atlas deliberately has a finite edge width.
                start = q + n * 0.0004
                rays_clear = [
                    bvh.ray_cast(start, direction)[0] is None for direction in disk
                ]
                external = crossing_x >= canopy_x
                samples.append([*q, *n, np.mean(rays_clear), external])
    if not samples or bpy.app.version[:2] < (4, 0):
        raise ValueError("Expected Blender 4 and visible well geometry")
    np.savez_compressed(dest, samples=np.array(samples), sun=sun)


def check_transport():
    from camera_remap import remap_packet
    from host_position import register_positions

    rays = np.zeros((2, 3, 4, 3))
    rays[..., 2] = 1
    origins = rays.copy() * 0
    origins[1, ..., 0] = 0.025
    pose = np.eye(4)
    pose[:3, 3] = [50, -70, 5]
    view = dict(
        rgb=np.zeros((3, 4, 3), "u1"),
        camera_to_world=pose,
        rays_camera=rays,
        ray_origins_camera=origins,
        surface_position=pose[:3, 3] + origins + rays * 3,
        position_valid=np.ones(rays.shape[:-1], bool),
        protected=np.zeros((3, 4), bool),
    )
    xy = np.array([[[2, 0], [0, 1], [-2, 0]]])
    values = np.arange(24).reshape(2, 3, 4) / 23
    for field in (values, values[0]):
        view["sun_visibility"] = field
        packet = register_positions(view)
        np.testing.assert_array_equal(packet["sun_visibility"], field)
        result = remap_packet(packet, xy)
        expected = field[..., [0, 1, 0], [2, 0, 0]].copy()
        expected[..., -1] = 0
        np.testing.assert_array_equal(
            result["sun_visibility"], np.expand_dims(expected, -2)
        )
        assert result["protected"][:, 0, -1].all()
    view["sun_visibility"] = np.full((3, 4), 2.0)
    try:
        register_positions(view)
    except ValueError as exc:
        if "sun_visibility" not in str(exc):
            raise
    else:
        raise AssertionError("Accepted invalid host sunlight visibility")


def check_control(path, control):
    from mesh_transfer import RoadMesh
    from pavement_sun import fit_sun, shade_hits

    mesh = RoadMesh(path.parent, filter_pavement=True, measured_rim=True)
    with np.load(control) as data:
        rows, sun = data["samples"], data["sun"]
    xyz, normals, opening, external = rows[:, :3], rows[:, 3:6], rows[:, 6], rows[:, 7]
    # These separate linear light levels and diffuse albedo are fixed for the synthetic control.
    source = np.full((len(rows), 3), 128, "u1")
    shade, lit = [64, 72, 81], [192, 185, 170]
    axis = np.arange(-1.58, 1.59, 0.02)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    xy = xy[(abs(xy) <= mesh.extent / 2 + 1.2).all(1)]
    visibility = np.float32(xy[:, 0] >= 0)
    model = fit_sun(
        dict(sun_world=sun, shade_srgb=shade, lit_srgb=lit),
        dict(basis_columns=np.eye(3)),
        mesh.extent,
        source,
        [(xy, np.zeros((len(xy), 3), "u1"), visibility)],
    )
    sky = np.full(len(rows), 0.7)
    actual = shade_hits(mesh, source, xyz, normals, sky, model, np.zeros_like(source))

    # Compute the reference RGB independently, in linear light then IEC sRGB.
    def decode(rgb):
        encoded = np.asarray(rgb) / 255
        return np.where(
            encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
        )

    cosine = np.clip(normals @ sun, 0, None) / sun[2]
    linear = (
        decode(shade) * sky[:, None]
        + (decode(lit) - decode(shade)) * (cosine * opening * external)[:, None]
    )
    v = np.clip(linear, 0, 1)
    expected = np.rint(
        np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255
    )
    errors = np.max(abs(actual - expected), axis=1)
    lit_count = int(np.sum((cosine > 0.1) & (opening > 0.99) & (external > 0)))
    shadow_count = int(np.sum((cosine > 0.1) & (opening < 0.01) & (external > 0)))
    if not lit_count or not shadow_count or (errors > 2).mean() > 0.002:
        raise ValueError(
            f"Directional light disagrees with Blender: {lit_count}, {shadow_count}, {np.percentile(errors, [95, 99.9, 100])}"
        )
    return dict(
        samples=len(rows),
        below_grade=int(np.sum(xyz[:, 2] < -0.01)),
        directly_lit=lit_count,
        shadowed_by_opening=shadow_count,
        canopy_shaded=int(np.sum(external == 0)),
        channel_error_p99=float(np.percentile(errors, 99)),
        max_channel_error=float(errors.max()),
        fraction_within_two_levels=float(np.mean(errors <= 2)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pack", "replay", "oracle", "control"])
    parser.add_argument("--models", type=Path, default=ROOT / "runs/sun-openings")
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/sun-openings")
    parser.add_argument("--packets", type=Path, default=ROOT / "work/opening-exposures")
    parser.add_argument("--controls", type=Path, default=ROOT / "work/opening-controls")
    args = parser.parse_args()
    if args.stage != "oracle":
        camera_packets = import_module("check-mei-pits")
    if args.stage == "pack":
        camera_packets.pack(args.packets, args.views, PAIRS, SITES)
    elif args.stage == "replay":
        camera_packets.replay(
            args.packets, args.models, args.views / "replay", EXAMPLES, PAIRS, SITES
        )
    else:
        args.controls.mkdir(parents=True, exist_ok=True)
        report = {}
        for key in ("A277", "M277"):
            path = asset_path(args.models, key)
            for light, direction in [
                ("high", [0.22, -0.29, 0.93]),
                ("low", [0.87, -0.048, 0.482]),
            ]:
                target = args.controls / f"{key}-{light}.npz"
                if args.stage == "oracle":
                    sun = np.array(direction) / np.linalg.norm(direction)
                    oracle(path, target, sun)
                else:
                    report[target.stem] = check_control(path, target)
        if args.stage == "control":
            check_transport()
            camera_packets.save(args.views / "control/check.json", report)


if __name__ == "__main__":
    main()
