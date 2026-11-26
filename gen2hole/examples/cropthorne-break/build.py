"""Recover only the surviving road and the truncated kerb from the Cropthorne photograph."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "releases/field-chambers/gen2hole/recipes/series"))
from mesh_io import Mesh, grid_faces, write_json


def rays(p, centre, focal):
    p = np.asarray(p)
    return np.concatenate([(p - centre) / focal, np.ones((*p.shape[:-1], 1))], axis=-1)


def unit(v):
    return v / np.linalg.norm(v)


def paint_points(gray, windows):
    tracks = []
    for y0, y1, x0, x1, half in windows:
        points = []
        for y in range(y0, y1 + 1, 2):
            if y < 377 and x0 > 400:
                continue  # Utility spray crosses the centre markings here.
            expected = round(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            x = np.arange(max(0, expected - half - 4), min(gray.shape[1], expected + half + 5))
            values = gray[y, x]
            weights = np.maximum(values - np.percentile(values, 20) - 9, 0) ** 1.3
            if weights.sum() > 55 and np.percentile(values, 85) - np.percentile(values, 20) > 13:
                points.append([np.average(x, weights=weights), y])
        tracks.append(np.array(points))
    return tracks


def calibrate(tracks, survey, centre, focal):
    def residual(params, groups):
        vx, vy, *slopes = params
        return np.concatenate(
            [(p[:, 0] - vx - m * (p[:, 1] - vy)) / np.hypot(1, m) for p, m in zip(groups, slopes)]
        )

    # Keep whole rows out of the fit, including the scale calculation.
    train = [p[np.arange(len(p)) % 3 != 1] for p in tracks]
    test = [p[1::3] for p in tracks]
    fit = least_squares(residual, [255, 110, -0.65, 1.0, 1.25], args=(train,), loss="soft_l1")
    fwd = unit(rays(fit.x[:2], centre, focal))
    normal = np.cross(*rays(survey["pole_endpoints_px"], centre, focal))
    up = unit(np.cross(fwd, normal))
    if up[1] > 0:
        up = -up
    basis = np.stack([unit(np.cross(fwd, up)), fwd, up], axis=-1)
    anchors = []
    for p in train:
        q = rays(p, centre, focal) @ basis
        anchors.append(np.median(-q[:, 0] / q[:, 2]))
    height = survey["road_mark_spacing_assumed_m"] / abs(anchors[1] - anchors[0])
    return basis, height, fit.x[:2], residual(fit.x, test)


def build(out):
    s = json.loads((HERE / "survey.json").read_text())
    for key, record in s["photographs"].items():
        path = HERE / "references" / f"{key}.jpg"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source checksum mismatch for {key}")
    rgb = np.array(Image.open(HERE / "references" / f"{s['primary']}.jpg").convert("RGB"))
    h, w = rgb.shape[:2]
    centre = np.array([w, h]) / 2
    focal = s["focal_mm"] * s["focal_plane_px_per_inch"] / 25.4 * w / s["sensor_image_width_px"]
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float), (5, 5), 0.85)
    tracks = paint_points(gray, s["paint_tracks_y0_y1_x0_x1_halfwidth"])
    basis, height, vp, test = calibrate(tracks, s, centre, focal)
    eye = np.array([0.0, 0.0, height])

    def ground(px):
        q = rays(px, centre, focal) @ basis
        return eye - height * q / q[..., 2, None]

    def project(world):
        q = (world - eye) @ basis.T
        return q[..., :2] / q[..., 2, None] * focal + centre

    guides = np.array(s["lip_guides_px"])
    x = np.arange(guides[0, 0], guides[-1, 0] + 1)
    prior = np.interp(x, *guides.T)
    candidates = np.rint(prior[:, None] + np.arange(-4, 5)).astype(int)
    edge = gray[candidates + 1, x[:, None]] - gray[candidates - 1, x[:, None]]
    best = np.argmax(edge - 1.8 * abs(candidates - prior[:, None]), axis=1)
    lip = np.column_stack([x, candidates[np.arange(len(x)), best]])
    lip_world = ground(lip)
    origin = np.median(lip_world, axis=0)
    toe = np.array(s["curb_toe_px"])
    across = np.arange(toe[1, 0], w)
    upper = np.interp(across, *np.vstack([toe[::-1], lip]).T)
    down = np.linspace(0, 1, 17)[:, None]
    road_px = np.stack(np.broadcast_arrays(across, upper + down * (h - 1 - upper)), axis=-1)

    # Sweep the visible terminal profile along the roadside. Its end section is
    # assumed perpendicular to the two photographed toe points in plan view.
    end, near = ground(toe)
    along = unit(near - end)
    q = rays(s["curb_end_profile_px"], centre, focal) @ basis
    profile = eye + q * (np.dot(end - eye, along) / (q @ along))[:, None]
    # Clip each longitudinal line at the bottom image row.
    base = (profile - eye) @ basis.T
    step = along @ basis.T
    bottom = (h - 1 - centre[1]) / focal
    distance = (bottom * base[:, 2] - base[:, 1]) / (step[1] - bottom * step[2])
    # Some of the crest leaves the image on the left before the bottom.
    left = -centre[0] / focal
    t_left = (left * base[:, 2] - base[:, 0]) / (step[0] - left * step[2])
    extent = np.where(t_left > 0, np.minimum(distance, t_left), distance)
    curb = profile + down[..., None] * extent[:, None] * along

    mesh = Mesh()
    mat = mesh.material("Cropthorne July 22", rgb, 0.94, 0.0, 0.0, lossless_color=True)

    def surface(name, world, px):
        vertices = (world - origin).reshape(-1, 3)
        tri = grid_faces(*px.shape[:2])
        a, b, c = vertices[tri].transpose(1, 0, 2)
        invert = np.cross(b - a, c - a)[:, 2] < 0
        tri[invert] = tri[invert, ::-1]
        uv = ((px.reshape(-1, 2) + 0.5) / [w, h]) * [1, -1] + [0, 1]
        mesh.add(name, vertices, tri, uv, mat)

    surface("Surviving carriageway", ground(road_px), road_px)
    surface("Truncated kerb", curb, project(curb))
    out.mkdir(parents=True, exist_ok=True)
    (out / "P572.glb").write_bytes(mesh.glb_bytes())
    # Semantic cores are partial positive annotations. Unlabelled pixels retain 0.
    masks = np.zeros((h, w), "u1")
    for i, polygon in enumerate(s["visible_cores_px"].values(), 1):
        cv2.fillPoly(masks, [np.int32(polygon)], i)
    Image.fromarray(masks).save(out / "context-cores.png")
    np.savez_compressed(
        out / "edge.npz", pixels=lip, xyz=lip_world - origin, kerb_profile_xyz=profile - origin
    )
    info = dict(
        id=s["id"],
        size_px=[w, h],
        principal_px=centre.tolist(),
        focal_px=focal,
        camera_xyz_m=(eye - origin).tolist(),
        camera_to_world_row_vectors=basis.tolist(),
        vanishing_point_px=vp.tolist(),
        paint_holdout_count=len(test),
        paint_holdout_rms_px=float(np.sqrt(np.mean(test**2))),
        paint_holdout_max_px=float(np.max(abs(test))),
        lip_columns=len(lip),
        lip_x_span_m=float(np.ptp(lip_world[:, 0])),
        kerb_relief_m=float(np.ptp(profile[:, 2])),
        visible_edge_forward_distance_m=np.percentile(lip_world[:, 1], [0, 50, 100]).tolist(),
        context_core_values={k: i for i, k in enumerate(s["visible_cores_px"], 1)},
        coordinates="Right, road-forward, up. glTF is (x,z,-y). Origin is the coordinatewise median of the traced lip.",
        depth_scope="Only road and visible kerb. Missing geometry has unknown range.",
    )
    write_json(out / "camera.json", info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    print(json.dumps(build(parser.parse_args().out), indent=2))
