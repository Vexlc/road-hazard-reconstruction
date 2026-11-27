"""Build the photographed depressed panel, its shadowed lip and a strictly tie-supported mesh."""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from cameras import HERE, S, SCALE, photo
from surface import reconstruct

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_up,
    flatten_scene,
    grid,
    make_one,
    texture_uv,
    write_json,
    xyz,
)

LABELS = [
    "Road adjoining the fracture",
    "Fracture lip and exposed bed",
    "Fracture sidewall in surface coordinates",
]


def build(out):
    s = reconstruct()
    a, b = s["cams"]
    top, toe = s["top"], s["toe"]
    lower = []
    upper = []
    zu = []
    zl = []
    for i in range(len(top)):
        j = (i + 1) % len(top)
        n = max(1, int(np.ceil(max(np.linalg.norm(r[j] - r[i]) for r in (toe, top)) / 2)))
        f = np.arange(n) / n
        lower.extend(toe[i] + f[:, None] * (toe[j] - toe[i]))
        upper.extend(top[i] + f[:, None] * (top[j] - top[i]))
        zl.extend(s["lower_z"][i] + f * (s["lower_z"][j] - s["lower_z"][i]))
        zu.extend(s["upper_z"][i] + f * (s["upper_z"][j] - s["upper_z"][i]))
    lower, upper = np.asarray(lower), np.asarray(upper)
    bedring = b.at(lower, np.array(zl))
    lip = b.at(upper, np.array(zu))
    # Keep a narrow surveyed road-side lip. A synthetic taper on the perimeter closes
    # the portable patch against grade; no inference is added to the stereo-only file.
    centre = (lip[:, :2].min(0) + lip[:, :2].max(0)) / 2
    vector = lip[:, :2] - centre
    vector /= np.linalg.norm(vector, axis=1)[:, None]
    outer = xyz(lip[:, :2] + 0.22 * vector, 0)
    offset = (outer[:, :2].min(0) + outer[:, :2].max(0)) / 2
    span = np.ptp(outer[:, :2], axis=0) + 0.10
    g = s["uv"].reshape(-1, 2)
    valid = s["mask"] & (cKDTree(lower).query(g)[0] > 1.2)
    uv = np.vstack([lower, g[valid]])
    verts = np.vstack([bedring, s["v"][valid]])
    f = Delaunay(uv).simplices
    # Delaunay with sampled boundary constraints. Remove exterior faces and verify
    # that every perimeter segment is an actual edge of the surviving panel.
    contour = np.float32(toe)
    ok = np.array([cv2.pointPolygonTest(contour, tuple(p), False) >= 0 for p in uv[f].mean(1)])
    a0, b0, c0 = uv[f].transpose(1, 0, 2)
    f = f[ok & (abs(np.cross(b0 - a0, c0 - a0)) > 1e-7)]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    pair, count = np.unique(edges, axis=0, return_counts=True)
    edge = pair[count == 1]
    missing = len(
        set(
            map(
                tuple,
                np.sort(np.c_[np.arange(len(lower)), np.roll(np.arange(len(lower)), -1)], axis=1),
            )
        )
        - set(map(tuple, edge))
    )
    # Short collinear boundary slivers from floating-point clipping must not enter
    # the triangulation. Reconcile the boundary before accepting an export.
    if missing:
        raise ValueError(f"{missing} unpaired boundary edges")
    f = face_up(verts, f)
    image0, image1 = photo(0), photo(1)
    # Do not paint equipment staged beside the break onto the collapsed road.
    part_uv = b.project(verts[f].mean(1))
    projected = part_uv.astype("f4")
    other = np.zeros(len(f), bool)
    for k in ("reverse_pipe_mask", "occluded_end_mask"):
        mask = cv2.fillPoly(np.zeros((598, 902), "u1"), [np.int32(S[k])], 1)
        px = np.rint(projected).astype(int)
        other |= mask[px[:, 1].clip(0, 597), px[:, 0].clip(0, 901)] > 0
    mesh = Mesh()
    off = np.r_[offset, 0]
    atlas = np.vstack([image1, image0])
    mat = mesh.material("Two USGS exposures", atlas, 0.93, 0, 0.35)
    tex1 = texture_uv(b.project(verts) * SCALE, atlas.shape[1::-1])
    tex0 = texture_uv(a.project(verts) * SCALE + [0, image1.shape[0]], atlas.shape[1::-1])
    mesh.add(
        LABELS[1],
        np.vstack([verts, verts]) - off,
        np.vstack([f[~other], f[other] + len(verts)]),
        np.vstack([tex1, tex0]),
        mat,
    )
    n = len(lip)
    i = np.arange(n)
    j = np.roll(i, -1)
    band = np.vstack([np.c_[i, i + n, j], np.c_[j, i + n, j + n]])
    pairs = np.vstack([lip, bedring])
    sidefaces = band[
        np.linalg.norm(
            np.cross(pairs[band[:, 1]] - pairs[band[:, 0]], pairs[band[:, 2]] - pairs[band[:, 0]]),
            axis=1,
        )
        > 1e-9
    ]
    mesh.add(
        LABELS[2],
        pairs - off,
        sidefaces,
        texture_uv(b.project(pairs) * SCALE, atlas.shape[1::-1]),
        mat,
    )
    # Source apron stops before staged pipes or grass. For the unseen underside and
    # receiving taper use a clean, photographed intact-asphalt swatch.
    apron = mesh.material("Adjoining pavement analogue", image0[1280:1460, 800:1340], 0.96, 0, 0.3)
    pv = np.vstack([outer, lip])
    mesh.add(LABELS[0], pv - off, face_up(pv, band), 0.5 + (pv[:, :2] - offset) / span, apron, True)
    series = out / "series"
    dest = series / "assets/pothole/P590"
    cache = series / "mesh_cache"
    field = grid(span, 0.012)
    span = np.ptp(field, axis=(0, 1))
    meta = dict(
        id="P590",
        label=S["label"],
        category="pothole",
        family="dover_collapse",
        state="subsided_carriageway_panel",
        source=[i["source"] for i in S["source"]],
        source_author=S["author"],
        source_license=S["rights"],
        seed=29590,
        nominal_mark_spacing_m=S["nominal_lane_mark_spacing_m"],
        surface_dimensions_m=span.tolist(),
        local_origin_xyz_m=off.tolist(),
        focal_in_annotation_pixels=[c.K[0, 0] for c in (a, b)],
        camera_centres_local_m=[(c.eye - off).tolist() for c in (a, b)],
        shadow_face_is_optical=True,
    )
    mesh.save(dest, cache, meta, dict(height=np.zeros(field.shape[:2])))
    flatten_scene(dest / "asset.glb")
    # A source-tie-only TIN supports observation-sensitive sensor work without extrapolation.
    use = s["use"]
    v = s["matches"]["xyz"][use]
    tri = Delaunay(v[:, :2]).simplices
    longest = np.linalg.norm(v[tri] - np.roll(v[tri], 1, axis=1), axis=2).max(1)
    tri = tri[longest < 0.40]
    supported = Mesh()
    paint = supported.material("USGS reverse exposure", image1, 0.93, 0, 0.35)
    supported.add(
        LABELS[1],
        v - off,
        face_up(v, tri),
        texture_uv(b.project(v) * SCALE, image1.shape[1::-1]),
        paint,
    )
    (dest / "observed.glb").write_bytes(supported.glb_bytes())
    flatten_scene(dest / "observed.glb")
    origin = xyz(field, 1)
    dirs = np.broadcast_to([0.0, 0.0, -1.0], origin.shape)
    dist, part, _ = Surface(dest / "asset.glb").cast(origin, dirs)
    measured, _, _ = Surface(dest / "observed.glb").cast(origin, dirs)
    hit = np.isfinite(dist)
    alpha = np.asarray(part > 1, "f4")
    np.savez_compressed(
        dest / "surface.npz",
        width=span[0],
        length=span[1],
        height=np.where(hit, 1 - dist, 0).astype("f4"),
        stereo_valid=np.isfinite(measured) & (part == 2),
        first_part=part,
        defect_mask=part > 1,
        installation_mask=hit,
        observed_footprint=hit,
        alpha=alpha,
        water_mask=np.zeros_like(hit),
        registered_relief=True,
        photographic_rgb=True,
    )
    row = make_one(dest, cache, point_limit=760000, kernel_factors=(0.89, 0.18))
    for k, name in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("metadata", "metadata.json"),
        ("surface", "surface.npz"),
    ]:
        row[k] = "pothole/P590/" + name
    write_json(series / "assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
