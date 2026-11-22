"""Check the four angular cover exports against the source annotations and each other."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cover_observations import compare_cases

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream/sensors"))
from road_geometry import Surface, glb_triangles, read_json, sha256

topology = import_module("check-perforated-covers").topology
LID = {9, 10, 11}
PART_COLORS = {
    5: (106, 126, 134),
    6: (58, 185, 177),
    7: (45, 88, 156),
    9: (242, 192, 88),
    10: (202, 106, 134),
    11: (205, 132, 76),
}


def cast(scene, xy):
    rays = np.column_stack([xy, np.full(len(xy), 2.0)])
    distance, parts, _ = scene.cast(rays, np.broadcast_to([0.0, 0.0, -1.0], rays.shape))
    return 2 - distance, parts


def inspect_photo(bundle, trace, spec, path, dest):
    marks = trace["landmarks"]
    xy = np.float32(marks["image_xy"])
    lattice = np.float32(marks["lattice_xy"])
    fit = np.asarray(marks["fit"])
    homography, _ = cv2.findHomography(lattice[fit], xy[fit], 0)
    reprojection = cv2.perspectiveTransform(lattice[None], homography)[0]
    error = np.linalg.norm(xy - reprojection, axis=1)
    if np.percentile(error[~fit], 95) > 3:
        raise ValueError("Held-out tread junctions miss the source image")
    # Test interior source pixels of every marked notch against the actual exported plate.
    mask = cv2.imread(str(bundle / "calibration/perimeter-recesses.png"), cv2.IMREAD_GRAYSCALE)
    components, labels = cv2.connectedComponents((mask > 127).astype("u1"))
    if components - 1 != len(spec["recesses_xy"]):
        raise ValueError("Source notch labels have merged or disappeared")
    overlay = Image.open(bundle / "references/frame.png").convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for (u, v), trained in zip(xy, fit):
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline="#42efce" if trained else "#ffda53", width=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        outline = [tuple(map(int, p)) for p in c[:, 0]]
        draw.line(outline + outline[:1], fill="#ff886d", width=2)
    overlay.save(dest / "source-landmarks.png")
    with Image.open(bundle / "references/lid_ortho.png") as face:
        width, length = trace["size_m"]
        face.resize((face.width, round(face.width * length / width)), Image.Resampling.LANCZOS).save(
            dest / "orthophoto.png"
        )
    lid = Surface(path, include_names={"Lid face"})
    checks = []
    for i in range(1, components):
        core = cv2.erode((labels == i).astype("u1"), np.ones((3, 3), "u1"))
        y, x = np.nonzero(core)
        pixels = np.float32(np.column_stack([x, y]))
        matrix = np.asarray(trace["pixel_to_canonical"])
        uv = cv2.perspectiveTransform(pixels[None], matrix)[0] / (trace["resolution"] - 1)
        _, hits = cast(lid, (uv - 0.5) * trace["size_m"] * [1, -1])
        if len(hits) < 5 or np.any(hits):
            raise ValueError(f"Exported metal crosses the interior of notch {i}")
        checks.append(len(hits))
    return dict(
        fit_junctions=int(fit.sum()),
        heldout_junctions=int((~fit).sum()),
        heldout_p95_px=float(np.percentile(error[~fit], 95)),
        notch_core_pixels=checks,
    )


def check(run):
    bundle = run / "series"
    trace = read_json(bundle / "calibration/grate.json")
    survey = read_json(bundle / "calibration/survey.json")
    paths = {p.parent.name: p for p in (bundle / "assets").glob("*/*/asset.glb")}
    dest = run / "evaluation"
    dest.mkdir(exist_ok=True)
    seated = next(
        s["id"] for s in survey["states"] if not any(s.get(k) for k in ("absent", "angle_deg", "yaw_deg"))
    )
    missing = next(s["id"] for s in survey["states"] if s.get("absent"))
    ref = {k: (v, f, uv, tex) for k, v, f, uv, tex in glb_triangles(paths[seated], appearance=True)}
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, 513) * trace["road_span_m"],
        np.linspace(0.5, -0.5, 513) * trace["road_span_m"],
    )
    grid = np.column_stack([x.ravel(), y.ravel()])
    stock_z, stock_part = cast(Surface(paths[missing]), grid)
    aperture = stock_part == 7
    result = dict(
        seated=seated,
        missing=missing,
        photo=inspect_photo(bundle, trace, survey["perforations"], paths[seated], dest),
        aperture_span_m=np.ptp(ref[7][0][:, :2], axis=0).tolist(),
        panel_span_m=trace["size_m"],
        sample_pitch_m=trace["road_span_m"] / 512,
        aperture_reference_rays=int(aperture.sum()),
        states={},
    )
    result["photo"]["source_sha256"] = sha256(
        bundle / "references" / survey["frame"].get("original_reference", "frame.png")
    )
    lid_index = bundle / "assets/lids.json"
    lids = read_json(lid_index) if lid_index.exists() else {}
    for state in survey["states"]:
        key = state["id"]
        path = paths[key]
        folder = dest / key
        folder.mkdir(exist_ok=True)
        meta = read_json(path.parent / "metadata.json")
        parts = list(glb_triangles(path, appearance=True))
        _, visible = cast(Surface(path), grid)
        # First surface seen from above. Every state uses the same orthographic field.
        palette = np.zeros((256, 3), "u1")
        palette[[0, 1, 5, 6, 7, 9, 10, 11]] = [
            [247, 245, 236],
            [227, 226, 215],
            [94, 113, 116],
            [82, 135, 142],
            [37, 76, 98],
            [220, 165, 90],
            [160, 101, 67],
            [170, 110, 69],
        ]
        Image.fromarray(palette[visible.reshape(x.shape)]).save(folder / "plan.png")
        record = dict(
            glb_sha256=sha256(path), visible_floor_fraction=float((visible[aperture] == 7).mean())
        )
        plate = [(k, v, f) for k, v, f, uv, tex in parts if k in LID]
        expected = set(ref) - (LID if state.get("absent") else set())
        if {p[0] for p in parts} != expected:
            raise ValueError(f"Unexpected component list in {key}")
        for k, vertices, faces, uv, texture in parts:
            v0, f0, uv0, texture0 = ref[k]
            if k in LID:
                pose = np.asarray(meta["lid_transform"])
                expected_vertices = v0 @ pose[:3, :3].T + pose[:3, 3]
            else:
                expected_vertices = v0
            if not (
                np.allclose(vertices, expected_vertices, atol=1e-7, rtol=0)
                and np.array_equal(faces, f0)
                and np.array_equal(uv, uv0)
                and texture == texture0
            ):
                raise ValueError(f"Geometry or appearance changed unexpectedly on part {k} of {key}")
        if plate:
            if lids:
                export = lids[key]
                lid_path = bundle / export["file"]
                if export["parent_sha256"] != record["glb_sha256"] or export["sha256"] != sha256(
                    lid_path
                ):
                    raise ValueError("Standalone lid does not match its assembled model")
                isolated = list(glb_triangles(lid_path))
                if {k for k, _, _ in isolated} != LID:
                    raise ValueError("Standalone lid lacks one of its three surfaces")
                originals = {k: v[f] for k, v, f in plate}
                for k, v, f in isolated:
                    if not np.array_equal(v[f], originals[k]):
                        raise ValueError("Detached cover triangles have moved")
                record["detached_lid"] = export
            pose = np.asarray(meta["lid_transform"])
            # Verify the requested motion from the exported transform independently of the pose builder.
            if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7) or not np.isclose(
                np.linalg.det(pose[:3, :3]), 1
            ):
                raise ValueError("The cover shape changed between rigid states")
            if state.get("angle_deg"):
                pivot = state["pivot_edge"]
                edge = np.asarray(trace["rim_uv"])[[pivot, (pivot + 1) % len(trace["rim_uv"])]]
                axis = np.c_[(edge - 0.5) * trace["size_m"] * [1, -1], np.zeros(2)]
                moved = axis @ pose[:3, :3].T + pose[:3, 3]
                angle = np.degrees(np.arccos(np.clip(pose[2, 2], -1, 1)))
                if not np.isclose(angle, state["angle_deg"]) or not np.allclose(moved, axis, atol=1e-7):
                    raise ValueError("The raised lid does not rotate about the traced edge")
                record["verified_edge_tilt_degrees"] = float(angle)
            elif state.get("yaw_deg"):
                angle = np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))
                if not np.isclose((angle - state["yaw_deg"] + 180) % 360 - 180, 0):
                    raise ValueError("Planar lid angle differs from the requested state")
                record["verified_yaw_degrees"] = float(angle)
            record.update(topology(plate))
            if record["through_holes"] != 0:
                raise ValueError("An open edge notch became a closed hole")
            underside, hit = cast(Surface(path, include_names={"Lid underside"}), grid)
            valid = (hit == 10) & np.isfinite(stock_z)
            gaps = underside[valid] - stock_z[valid]
            record["minimum_underside_clearance_mm"] = float(gaps.min() * 1000)
            record["seat_contact_samples"] = int(
                ((abs(gaps) <= 0.00001) & (stock_part[valid] == 5)).sum()
            )
            if gaps.min() < -0.0001 or (
                not state.get("angle_deg") and not record["seat_contact_samples"]
            ):
                raise ValueError(f"Unsupported or intersecting cover in {key}")
        result["states"][key] = record
    return result


def check_transfer(run, report):
    road = import_module("check-pit-transfer").check(run, report["states"])
    sites = {}
    for case, scene in road["scenes"].items():
        sites.setdefault(case.split("_")[0], {})[scene["asset"]] = case
    baseline = report["seated"]
    matched = {}
    visibility = {}
    for site, cases in sites.items():
        if set(cases) != set(report["states"]):
            raise ValueError(f"Incomplete state family on road {site}")
        matched[site] = {}
        for key, case in cases.items():
            folder = run / "sensors/matched_camera" / case
            for shot in read_json(folder / "views.json"):
                if not shot["available"]:
                    continue
                with np.load(folder / shot["geometry"]) as layers:
                    visible = layers["part"]
                    if np.any(visible[layers["protected"] != 0]):
                        raise ValueError("A protected vehicle or foreground object received a part label")
                with Image.open(folder / shot["after_stem"]) as im:
                    overlay = np.array(im.convert("RGB"))
                counts = {}
                for part, color in PART_COLORS.items():
                    selected = visible == part
                    counts[part] = int(selected.sum())
                    overlay[selected] = np.rint(0.30 * overlay[selected] + 0.70 * np.array(color))
                if sum(counts.values()) != shot["visible_target_pixels"]:
                    raise ValueError("Part mask disagrees with the calibrated camera record")
                name = f"{site}-{shot['camera']}"
                png = run / "evaluation" / key / (name + "-parts.png")
                Image.fromarray(overlay).save(png)
                visibility.setdefault(key, {})[name] = dict(
                    frame=shot["frame"], pixels=counts, image=str(png.relative_to(run))
                )
            if key == baseline:
                continue
            comparison = compare_cases(run, site, cases[baseline], case, camera_folder="matched_camera")
            if set(comparison["cameras"]) != {"front_camera", "back_camera"}:
                raise ValueError(f"Missing matched views for {case}")
            matched[site][key] = comparison
    road["matched_to_seated"] = matched
    road["visible_parts"] = visibility
    return road


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--transfer-out", type=Path)
    args = parser.parse_args()
    report = check(args.run)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.transfer_out:
        args.transfer_out.write_text(
            json.dumps(check_transfer(args.run, report), indent=2, allow_nan=False) + "\n"
        )
