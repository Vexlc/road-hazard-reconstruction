"""Check source geometry, independently rebuilt files, fitted driving clearance and saved camera transfers."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, resolve_samples
from prop_registration import driving_clearance
from road_geometry import glb_triangles, sha256, save_json

IDS = ("B232", "B233", "H234", "H235")


def read(path):
    return json.loads(path.read_text())


def geometry(folder, other):
    key = folder.name
    mesh = RoadMesh(folder)
    meta = read(folder / "metadata.json")
    assert mesh.standing and not mesh.cut.any()
    assert meta["base_p90_abs_m"] < 0.02 and meta["base_sample_count"] >= 100
    assert np.allclose(np.ptp(mesh.bounds, axis=0), meta["size_xyz_m"], atol=0.00001)
    for name, digest in meta["files"].items():
        assert sha256(folder / name) == digest
    if other:
        for name in [*meta["files"], "metadata.json"]:
            assert sha256(folder / name) == sha256(other / key / name), (key, name)
    with np.load(folder / "samples.npz") as samples, np.load(folder / "surface.npz") as fields:
        assert fields["contact_mask"].any() and np.max(fields["sky_blocked"]) > 0.4
        assert np.allclose(np.linalg.norm(samples["normal"], axis=1), 1, atol=1e-5)
        assert np.all(
            (samples["xyz"] >= mesh.bounds[0] - 0.0001) & (samples["xyz"] <= mesh.bounds[1] + 0.0001)
        )
        assert len(samples["xyz"]) == meta["surface_samples"]
    labels = sorted({value[0] for value in mesh.surfaces.values()})
    assert labels == ([21] if key.startswith("B") else [22, 23, 24])
    return dict(
        files={**meta["files"], "metadata.json": sha256(folder / "metadata.json")},
        labels=labels,
        triangles=meta["triangles"],
        surface_samples=meta["surface_samples"],
        base_p90_m=meta["base_p90_abs_m"],
    )


def physical(views, assets):
    stats = {}
    for site, key in [("hydrant", "H234"), ("barrier", "B232"), ("audi", "B233")]:
        folder = views / site
        anchor = read(folder / "anchor.json")
        clearance = anchor["vehicle_clearance"]
        assert clearance["minimum_clearance_m"] >= clearance["required_margin_m"]
        assert clearance["sample_spacing_m"] <= 0.07501
        assert anchor["ground_receivers"]["measured_cells"] > 20
        mesh = RoadMesh(assets / key)
        index = read(folder / "index.json")
        photos = {}
        for asset, record in index.items():
            for sensor, v in record["views"].items():
                path = folder / asset / sensor
                before = np.asarray(Image.open(path / "before.png"))
                after = np.asarray(Image.open(path / "after.png"))
                labels = np.asarray(Image.open(path / "part.png"))
                dist = np.load(path / "range.npy")
                shade = np.load(path / "shadow.npy")
                hit = labels > 0
                assert np.array_equal(hit, np.isfinite(dist)) and hit.sum() == v["visible"]
                assert np.isfinite(shade).all() and shade.min() >= 0 and shade.max() <= 1
                assert np.any(~hit & (shade > 0))
                outside = ~hit & (shade == 0)
                assert np.array_equal(before[outside], after[outside]) and v["protected_edits"] == 0
                photos[asset + "/" + sensor] = dict(
                    frame=v["frame"],
                    object_pixels=int(hit.sum()),
                    shadow_only_pixels=int((~hit & (shade > 0)).sum()),
                )
                if asset != key:
                    continue
                with np.load(path / "view.npz") as p:
                    rgb, pose, rays, depth, protected = [
                        p[k] for k in ("rgb", "camera_to_world", "rays_camera", "host_range", "protected")
                    ]

                def replay():
                    return resolve_samples(
                        [
                            mesh.render(rgb, pose, r, anchor, depth, m, with_shadow=True)
                            for r, m in zip(rays, protected)
                        ]
                    )

                a, l, d, s = replay()
                assert np.array_equal(a, after) and np.array_equal(l, labels)
                assert np.array_equal(d, dist, equal_nan=True) and np.array_equal(s, shade)
                guard = np.all(protected, axis=0)
                assert not np.any(hit[guard]) and not np.any(shade[guard])
                # One subpixel is sufficient to isolate receiver gating and metric foreground.
                ray = rays[0]
                mask = protected[0]
                opts = dict(with_shadow=True)
                a, l, d, s = mesh.render(rgb, pose, ray, anchor, depth, mask, **opts)
                _, l2, d2, s2 = mesh.render(
                    rgb, pose, ray, anchor, depth, mask, receiver_mask=np.zeros_like(mask), **opts
                )
                assert np.array_equal(l, l2) and np.array_equal(d, d2, equal_nan=True) and not s2.any()
                front = np.full_like(depth, 0.02)
                a2, l2, d2, s2 = mesh.render(
                    rgb, pose, ray, anchor, depth, mask, occluder_range=front, **opts
                )
                assert np.array_equal(rgb, a2) and not l2.any() and not np.isfinite(d2).any() and not s2.any()
        stats[site] = dict(clearance=clearance, photos=photos)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/roadside-props")
    p.add_argument("--second", type=Path, help="Fresh independent build of the same four assets")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/roadside-props")
    p.add_argument("--assets-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    results = dict(assets={key: geometry(args.assets / key, args.second) for key in IDS})
    if args.second:
        results["identical_rebuilt_files"] = 16
    with np.load(args.assets / "H234/samples.npz") as a, np.load(args.assets / "H235/samples.npz") as b:
        assert not np.array_equal(a["albedo"], b["albedo"])
    fresh, aged = [list(glb_triangles(args.assets / key / "asset.glb")) for key in ("H234", "H235")]
    assert len(fresh) == len(aged)
    for (part, vertices, faces), (other_part, other_vertices, other_faces) in zip(fresh, aged):
        assert part == other_part and np.array_equal(faces, other_faces)
        assert np.allclose(vertices, other_vertices, rtol=0, atol=1e-6)
    if not args.assets_only:
        results["physical"] = physical(args.views, args.assets)
        scene = ROOT / "runs/streetgs-joint/kitti360-0000/30000"
        source = sha256(scene / "joint.pt")
        results["native"] = {}
        for pair in ["H234-B232", "H235-B233"]:
            directory = scene / "roadside-props" / pair
            placed = read(directory / "placement.json")
            assert placed["source_sha256"] == source and placed["checkpoint_sha256"] == sha256(
                directory / "joint.pt"
            )
            assert placed["actors_exact"] and placed["original_geometry_exact"]
            replay = read(args.views / "joint" / pair / "replay.json")
            assert all(r["frames"] == 48 and r["peak_changed_pixels"] > 50 for r in replay.values())
            results["native"][pair] = dict(placement=placed, replay=replay)
        results["original_front_error"] = read(args.views / "joint/original/replay.json")["front"][
            "max_baseline_error"
        ]
        assert results["original_front_error"] <= 2
        results["scgs"] = {}
        for directory in ["scgs-hydrant", "scgs-barrier"]:
            data = read(args.views / directory / "index.json")
            for scene, row in data.items():
                assert all(
                    r["native_shadow_pixels"] and r["depth_matches_photo_exactly"]
                    for r in row["models"].values()
                )
                results["scgs"][scene] = row
        light = read(args.views / "hydrant/anchor.json")["lighting"]
        assert 0 < light["direct_fraction"] < 0.9 and light["max_reprojection_px"] < 8
        results["light"] = light
    # The clearance control must detect a footprint intersecting the surveyed vehicle path.
    anchor = dict(center_world=[0, 0, 0], basis_columns=np.eye(3))
    try:
        driving_clearance(
            anchor, [[-0.5, -0.2], [0.5, 0.2]], [[-5, 0, 0], [5, 0, 0]], half_length_m=2.55, half_width_m=1.03
        )
    except ValueError as e:
        assert "intersects" in str(e)
    else:
        raise AssertionError("Overlapping trajectory accepted")
    save_json(args.out, results)
    print(f"Verified {len(IDS)} assets" + (" and scene transfers" if not args.assets_only else "") + ".")


if __name__ == "__main__":
    main()
