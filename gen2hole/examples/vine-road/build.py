"""Reconstruct the displaced pavement plates of Vine Road in their survey coordinates."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
from camera import HERE, S, Survey, tri
from plates import SCALE, estimate, inside

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from photo_mesh import (
    Mesh,
    Surface,
    face_up,
    flatten_scene,
    grid,
    make_one,
    refine,
    rings_faces,
    write_json,
    xyz,
)

ASSET = Path("series/assets/pothole/P592")


def atlas(vertices, index, name, paint=None):
    """Bake a metric tile per plate so splat and glTF samplers share identical boundaries."""
    source = np.asarray(Image.open(HERE / "inputs/asphalt.png"))
    lo = vertices[:, :2].min(0) - 0.15
    span = np.ptp(vertices[:, :2], axis=0) + 0.30
    w, h = np.maximum(8, np.ceil(span * 48).astype(int))
    yy, xx = np.mgrid[:h, :w]
    x = (xx + 0.5) / w * span[0]
    y = (1 - (yy + 0.5) / h) * span[1]
    theta = index * 2.39996
    u = (np.cos(theta) * x + np.sin(theta) * y) / 2.2
    v = (-np.sin(theta) * x + np.cos(theta) * y) / 2.2
    tile = cv2.remap(
        source,
        (u * 512).astype("f4"),
        (v * 512).astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    # Match only aggregate channel medians, retaining the independent CC0 texture.
    target = np.array(S["surface_rgb_median"][name])
    tile = np.clip(target + 0.78 * (tile.astype(float) - np.median(source, axis=(0, 1))), 0, 255)
    uv_map = lambda q: (q[:, :2] - lo) / span
    if paint is not None:
        coords = uv_map(paint) * [w, -h] + [0, h] - 0.5
        mask = np.zeros((h, w), "u1")
        cv2.polylines(mask, [np.int32(np.round(coords * 16))], False, 255, 4, cv2.LINE_AA, shift=4)
        tile += (mask.astype(float) / 255)[..., None] * np.array([28.0, 28.0, 26.0])
    return np.uint8(np.clip(tile, 0, 255)), uv_map


def support_faces(panel):
    """Keep short, locally supported facets entirely inside a single traced plate."""
    p = panel.pts
    uv = panel.cam.project(p)
    faces = Delaunay(uv).simplices
    vv = p[faces]
    edge = np.linalg.norm(vv - np.roll(vv, 1, axis=1), axis=2)
    faces = faces[edge.max(1) < 1.5]
    # Test against the whole polygon via geometric intersection, including concave notches.
    from shapely.geometry import Polygon

    polygon = Polygon(panel.poly)
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    unit = np.c_[np.cos(theta), np.sin(theta)]
    for x, y, rx, ry in S["exclusions_720px"]:
        polygon = polygon.difference(Polygon(([x, y] + unit * [rx, ry]) * SCALE))
    faces = np.array([f for f in faces if polygon.covers(Polygon(uv[f]))], dtype="i4").reshape(
        -1, 3
    )
    return face_up(p, faces)


def percentiles(values):
    return np.percentile(values, [50, 90]).tolist() if len(values) else []


def build(out):
    survey = Survey()
    cameras = survey.fit()
    panels, counts, tests = estimate(survey, cameras)
    bounds = np.vstack([p.world(p.poly) for p in panels.values()])
    centre = (bounds.min(0) + bounds.max(0)) / 2
    centre[2] = survey.height("before", centre[None])[0]
    mesh, strict = Mesh(), Mesh()
    for index, (name, p) in enumerate(panels.items()):
        uv, f = refine(p.poly, rings_faces([p.poly]), 12.0)
        v = p.world(uv)
        trace = S["faint_edge_traces_720px"].get(name)
        paint = None if trace is None else p.world(np.asarray(trace) * SCALE)
        tile, uv_map = atlas(np.vstack([v, p.pts]), index, name, paint)
        label = "Vine Road plate " + name
        short = support_faces(p)
        for obj, verts, faces in [(mesh, v, face_up(v, f)), (strict, p.pts, short)]:
            if len(faces):
                material = obj.material("Aggregate " + name, tile, 0.94, 0, 0.32)
                obj.add(label, verts - centre, faces, uv_map(verts), material)
        errors = []
        for world, source, _, _ in tests:
            keep = inside(source, p.poly) > 1.5 * SCALE
            if keep.any():
                errors.extend(abs(world[keep, 2] - p.world(source[keep])[:, 2]))
        counts[name].update(
            strict_facets=len(short),
            withheld_rows=len(errors),
            withheld_abs_z_m_p50_p90=percentiles(errors),
        )
    dest = out / ASSET
    cache = out / "series/mesh_cache"
    field = grid(np.ptp(bounds[:, :2], axis=0) + 0.3, 0.10)
    span = np.ptp(field, axis=(0, 1))
    a, b, c = cameras
    t = survey.triples[survey.reserved]
    px = np.linalg.norm(c.project(tri(a, b, t[:, :2], t[:, 2:4])) - t[:, 4:], axis=1)
    cb = survey.pairs[2]
    held = survey.roads[cb["hold"][survey.roads]]
    road = tri(c, b, cb["u"][held], cb["v"][held])
    dz = abs(road[:, 2] - (survey.height("before", road) + survey.height("after", road)) / 2)
    meta = dict(
        id="P592",
        category="pothole",
        label=S["label"],
        family="vine_road",
        seed=29592,
        state="earthquake_displaced_road_plates",
        source=[s["url"] for s in S["early_sources"]],
        local_origin_utm6n_navd88_m=(np.array(S["local_origin_utm6n_z_m"]) + centre).tolist(),
        survey_to_model_subtract_m=centre.tolist(),
        surface_dimensions_m=span.tolist(),
        camera_parameters_survey=[c.q.tolist() for c in cameras],
        panels=counts,
        included_panels=list(panels),
        third_view_holdout=dict(n=len(px), pixel_p50_p90=percentiles(px)),
        intact_road_holdout=dict(n=len(dz), abs_z_m_p50_p90=percentiles(dz)),
        material=S["material"],
        unobserved="No faces in unobserved fissure interiors.",
    )
    mesh.save(dest, cache, meta, dict())
    (dest / "supported.glb").write_bytes(strict.glb_bytes())
    for n in ("asset.glb", "supported.glb"):
        flatten_scene(dest / n)
    ztop = float(bounds[:, 2].max() - centre[2] + 4)
    origins = xyz(field, ztop)
    directions = np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    full = Surface(dest / "asset.glb")
    limited = Surface(dest / "supported.glb")
    dist, part, _ = full.cast(origins, directions)
    near, _, _ = limited.cast(origins, directions)
    hit = np.isfinite(dist)
    measured = np.isfinite(near)
    height = np.where(hit, ztop - dist, 0).astype("f4")
    before = (survey.height("before", field + centre[:2]) - centre[2]).astype("f4")
    support_z = np.where(measured, ztop - near, np.nan).astype("f4")
    # Require correspondence to the same topmost plate in overlapping projections.
    import open3d as o3d

    rays = o3d.core.Tensor(np.concatenate([origins, directions], axis=2).astype("f4"))
    from road_geometry import glb_triangles

    def plate_ids(scene, path):
        ids = scene.scene.cast_rays(rays, nthreads=2)["geometry_ids"].numpy()
        lut = np.array(
            [
                list(panels).index(label.removeprefix("Vine Road plate ")) + 1
                for label, _, _ in glb_triangles(path, names=True)
            ],
            dtype="u1",
        )
        result = np.zeros(ids.shape, "u1")
        valid = ids < len(lut)
        result[valid] = lut[ids[valid]]
        return result

    ids = plate_ids(full, dest / "asset.glb")
    small_ids = plate_ids(limited, dest / "supported.glb")
    valid = hit & measured & (ids == small_ids)
    np.savez_compressed(
        dest / "surface.npz",
        x=field[0, :, 0],
        y=field[:, 0, 1],
        width=span[0],
        length=span[1],
        height=height,
        first_part=part,
        plate_id=ids,
        stereo_valid=valid,
        supported_height=support_z,
        before_grade=before,
        before_radius_m=survey.height("before_dist", field + centre[:2]).astype("f4"),
        drop_from_before=np.where(hit, before - height, np.nan),
        installation_mask=hit,
        defect_mask=hit,
        alpha=hit.astype("f4"),
        water_mask=np.zeros_like(hit),
        registered_relief=True,
        photographic_rgb=False,
    )
    row = make_one(dest, cache, point_limit=950000, kernel_factors=(0.87, 0.16))
    row["origin"] = "Horizontal footprint centre; Z zero is 2011 ground at that position."
    row["strict_area_fraction"] = float(valid.sum() / hit.sum())
    grade_support = valid & (survey.height("before_dist", field + centre[:2]) < 2.5)
    row["dated_comparison_cells"] = int(grade_support.sum())
    row["drop_from_before_m_p50_p95"] = np.percentile(
        (before - support_z)[grade_support], [50, 95]
    ).tolist()
    write_json(dest / "metadata.json", row)
    for k, n in [
        ("glb", "asset.glb"),
        ("ply", "asset.ply"),
        ("surface", "surface.npz"),
        ("metadata", "metadata.json"),
    ]:
        row[k] = "pothole/P592/" + n
    write_json(out / "series/assets/asset_manifest.json", [row])
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
