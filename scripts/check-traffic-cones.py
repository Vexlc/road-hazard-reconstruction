"""Audit stable support, registered layouts, exported cameras, native pairs and the independent fisheye renderer."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mesh_transfer import RoadMesh, resolve_samples
from road_geometry import glb_triangles, save_json, sha256


def read(path):
    return json.loads(path.read_text())


def shapes(assets, second):
    folder = assets / "C236-C237"
    recipe = read(ROOT / "gen2hole/examples/traffic-cones/layout.json")
    combined = list(glb_triangles(folder / "asset.glb", names=True))
    singles = {}
    result = {}
    for key in ("C236", "C237", "C236-C237"):
        d = assets / key
        meta = read(d / "metadata.json")
        files = {f.name: sha256(f) for f in d.iterdir() if f.is_file()}
        for f, h in meta["files"].items():
            assert files[f] == h
        if second:
            for f, h in files.items():
                assert sha256(second / key / f) == h, (key, f)
        with np.load(d / "samples.npz") as s:
            v = s["sky_visibility"]
            assert np.isfinite(v).all() and v.min() >= 0 and v.max() < 1.00001
            assert (s["dielectric_f0"] > 0.15).all() and (s["dielectric_f0"] < 0.17).all()
            assert np.percentile(v, 5) < 0.3 and np.percentile(v, 90) > 0.98
        result[key] = dict(files=files, samples=meta["surface_samples"], triangles=meta["triangles"])
        if key in ("C236", "C237"):
            ((part, xyz, faces),) = glb_triangles(d / "asset.glb")
            assert part == 25 and abs(xyz[:, 2].min()) < 1e-6
            singles[key] = (xyz, faces)
    for row, (name, vertices, triangles) in zip(recipe["instances"], combined):
        coords, faces = singles[row["asset"]]
        rot = Rotation.from_euler("z", row["heading_deg"], degrees=True).as_matrix()
        expected = coords @ rot.T + [*row["xy_m"], 0]
        assert (
            name.endswith("@" + row["id"])
            and np.array_equal(triangles, faces)
            and np.allclose(vertices, expected, rtol=0, atol=2e-7)
        )
    a, face = singles["C236"]
    b, fallen_faces = singles["C237"]
    assert np.array_equal(face, fallen_faces)
    contact = b[:, 2] <= 0.001
    tri = b[face]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    com = np.average(tri.mean(1), axis=0, weights=area)
    eq = ConvexHull(b[contact, :2]).equations
    margin = float(-(eq[:, :2] @ com[:2] + eq[:, 2]).max())
    assert margin > 0.08 and a[contact, 2].max() > 0.51 and a[contact, 2].min() < 0.025
    result["C237"].update(
        contact_vertices=int(contact.sum()),
        support_margin_m=margin,
        tilt_deg=read(assets / "C237/metadata.json")["support_pose"]["tilt_from_vertical_deg"],
    )
    return result


def audits(a):
    out = dict(assets=shapes(a.assets, a.second))
    if a.second:
        out["byte_identical_rebuilt_files"] = 12
    if a.assets_only:
        return out
    mesh = RoadMesh(a.assets / "C236-C237")
    out["physical"] = {}
    for site in ("urban", "sun", "audi"):
        p = a.views / site
        anchor = read(p / "anchor.json")
        clearance = anchor["vehicle_clearance"]["minimum_clearance_m"]
        assert clearance >= anchor["vehicle_clearance"]["required_margin_m"]
        record = dict(minimum_vehicle_clearance_m=clearance, cameras={})
        for cam, r in read(p / "index.json")["C236-C237"]["views"].items():
            folder = p / "C236-C237" / cam
            with np.load(folder / "view.npz") as packet:
                rays, near, rgb, pose, protected = [
                    packet[k] for k in ("rays_camera", "host_range", "rgb", "camera_to_world", "protected")
                ]
            after, parts, dist, shadow = resolve_samples(
                [
                    mesh.render(rgb, pose, ray, anchor, near, mask, with_shadow=True)
                    for ray, mask in zip(rays, protected)
                ]
            )
            assert np.array_equal(after, np.array(Image.open(folder / "after.png"))) and np.array_equal(
                parts, np.array(Image.open(folder / "part.png"))
            )
            assert np.array_equal(dist, np.load(folder / "range.npy"), equal_nan=True) and np.array_equal(
                shadow, np.load(folder / "shadow.npy")
            )
            changed = (after != rgb).any(-1)
            assert not changed[protected.all(0)].any() and not changed[(parts == 0) & (shadow == 0)].any()
            note = dict(
                frame=r["frame"],
                pixels=int((parts > 0).sum()),
                pure_shadow_pixels=int(((parts == 0) & (shadow > 0)).sum()),
            )
            if site == "audi" and cam == "rear_center":
                prev = read(ROOT / "gen2hole/examples/mixed-road/ego-masks.json")["a2d2-frontleft"][cam]
                edge = np.asarray(prev["edge"]) * ([1920, 1208] / np.array(prev["grid"]))
                x, y = r["crop_xy"]
                h, w = parts.shape
                body = y + np.arange(h)[:, None] >= np.interp(x + np.arange(w), *edge.T)
                recovered = (parts > 0) & body
                assert recovered.sum() > 1000
                note["road_pixels_recovered_from_coarse_outline"] = int(recovered.sum())
            record["cameras"][cam] = note
        out["physical"][site] = record
    maskfile = ROOT / "gen2hole/examples/traffic-cones/audi-body.json"
    shape = read(maskfile)
    trained = np.array(shape["a2d2-frontleft"]["rear_center"]["edge"])
    images = sorted((ROOT / "work/data/a2d2-frontleft/rear_center").glob("*.png"))
    held = images[1::4]
    assert not {p.name for p in held} & set(shape["fit"]["frames"])
    coarse = read(ROOT / "gen2hole/examples/mixed-road/ego-masks.json")["a2d2-frontleft"]["rear_center"]
    independent, _ = import_module("fit-ego-edge").detect(held, coarse, 8, 72, 6, True)
    error = abs(trained[:, 1] - np.array(independent["edge"])[:, 1])
    assert np.percentile(error, 95) <= 3
    out["rig_mask"] = dict(
        held_out_exposures=len(held),
        p95_repeat_error_px=float(np.percentile(error, 95)),
        max_repeat_error_px=int(error.max()),
        mask_sha256=sha256(maskfile),
    )
    out["cycles"] = read(a.views / "cycles-check/check.json")
    scene = ROOT / "runs/streetgs-joint/kitti360-0000/30000"
    d = scene / "traffic-cones/C236-C237"
    placement = read(d / "placement.json")
    assert placement["source_sha256"] == sha256(scene / "joint.pt") and placement[
        "checkpoint_sha256"
    ] == sha256(d / "joint.pt")
    assert (
        placement["original_geometry_exact"]
        and placement["actors_exact"]
        and len(placement["placements"][0]["instances"]) == 2
    )
    replay = read(a.views / "joint/C236-C237/replay.json")
    assert all(v["frames"] == 48 and v["peak_changed_pixels"] > 70 for v in replay.values())
    out["native"] = dict(
        placement=placement, replay=replay, original=read(a.views / "joint/original/replay.json")
    )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "runs/traffic-cones")
    p.add_argument("--second", type=Path)
    p.add_argument("--assets-only", action="store_true")
    p.add_argument("--views", type=Path, default=ROOT / "outputs/traffic-cones")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save_json(a.out, audits(a))
    print("Verified cone support and registration.")
