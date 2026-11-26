"""Raycast the distributed GLB; write range, source-view labels, orthophoto and validation."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import trimesh

HERE = Path(__file__).resolve().parent


class Renderer:
    def __init__(self, path):
        model = trimesh.load(path, force="scene", process=False)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.parts = {}
        for name in model.graph.nodes_geometry:
            transform, key = model.graph[name]
            m = model.geometry[key].copy()
            m.apply_transform(transform)
            v = np.asarray(m.vertices)[:, [0, 2, 1]] * [1, -1, 1]
            gid = self.scene.add_triangles(
                o3d.core.Tensor(v.astype("f4")), o3d.core.Tensor(m.faces.astype("uint32"))
            )
            self.parts[gid] = (name, m)

    def render(self, origins, directions):
        dirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        rays = np.concatenate(np.broadcast_arrays(origins, dirs), axis=-1).astype("f4")
        hit = self.scene.cast_rays(o3d.core.Tensor(rays))
        ranges = hit["t_hit"].numpy()
        ranges[~np.isfinite(ranges)] = np.nan
        ids = hit["geometry_ids"].numpy()
        rgb = np.zeros((*ids.shape, 3), "u1") + [36, 40, 39]
        parts = np.zeros_like(ids, "u1")
        for gid, (name, m) in self.parts.items():
            mask = ids == gid
            if not np.any(mask):
                continue
            parts[mask] = 1 if name == "Surviving carriageway" else 2
            faces = m.faces[hit["primitive_ids"].numpy()[mask]]
            weights = hit["primitive_uvs"].numpy()[mask]
            weights = np.column_stack((1 - weights.sum(axis=1), weights))
            uv = (m.visual.uv[faces] * weights[:, :, None]).sum(axis=1)
            texture = np.asarray(m.visual.material.baseColorTexture.convert("RGB"))
            uv = uv * [1, -1] + [0, 1]
            pixels = np.rint(uv * texture.shape[1::-1] - 0.5).astype(int)
            pixels = np.clip(pixels, 0, np.array(texture.shape[1::-1]) - 1)
            rgb[mask] = texture[pixels[:, 1], pixels[:, 0]]
        return ranges, parts, rgb.astype("u1")


def crossings(edge, x):
    a, b = edge[:-1], edge[1:]
    m = (a[:, 0] <= x) != (b[:, 0] <= x)
    if not m.any():
        raise ValueError("Vehicle trace is outside photographed break")
    t = (x - a[m, 0]) / (b[m, 0] - a[m, 0])
    return (a[m, 1] + t * (b[m, 1] - a[m, 1])).min()


def check(out):
    info = json.loads((out / "camera.json").read_text())
    s = json.loads((HERE / "survey.json").read_text())
    rgb = np.array(Image.open(HERE / "references/501704.jpg").convert("RGB"))
    eye = np.array(info["camera_xyz_m"])
    yy, xx = np.indices(rgb.shape[:2])
    dirs = np.stack(
        [
            (xx - info["principal_px"][0]) / info["focal_px"],
            (yy - info["principal_px"][1]) / info["focal_px"],
            np.ones_like(xx),
        ],
        axis=-1,
    )
    model = Renderer(out / "P572.glb")
    r, part, render = model.render(eye, dirs @ info["camera_to_world_row_vectors"])
    np.savez_compressed(out / "range.npz", range_m=r, part_id=part)
    context = np.asarray(Image.open(out / "context-cores.png"))
    counts = {name: int((context == i).sum()) for name, i in info["context_core_values"].items()}
    assert min(counts.values()) > 20 and np.all(np.isnan(r[context != 0]))
    assert set(np.unique(part)) == {0, 1, 2}
    color_error = np.abs(render.astype(int) - rgb).mean(axis=-1)
    e = np.load(out / "edge.npz")
    lip = e["xyz"]
    u = e["pixels"][4:-4:4, 0]
    # Check topmost rendered asphalt hit against a separately read GLB at integer centres.
    boundary = np.argmax(part[:, u] == 1, axis=0)
    boundary_err = abs(boundary - e["pixels"][4:-4:4, 1])
    assert max(boundary_err) <= 2 and info["paint_holdout_rms_px"] < 1
    err = float(color_error[part > 0].mean())
    assert err < 3, err
    # An overhead orthophoto also checks the direction of UVs in an independent view.
    pitch = 0.009
    ymax = 0.48
    xmin = -2.66
    xo = xmin + xx[:338, :580] * pitch
    yo = ymax - yy[:338, :580] * pitch
    origins = np.dstack([xo, yo, np.ones_like(xo)])
    _, _, ortho = model.render(origins, np.broadcast_to([0, 0, -1.0], origins.shape))
    Image.fromarray(ortho).save(out / "orthophoto.png")
    # A hypothetical aligned camera path, two contact traces separated by 1.60 m.
    paths = []
    for x in (eye[0] - 0.80, eye[0], eye[0] + 0.80):
        paths.append(dict(x_m=x, forward_from_camera_m=crossings(lip, x) - eye[1]))
    nominal = s["road_mark_spacing_assumed_m"]
    edges = [
        dict(
            mark_spacing_m=a,
            visible_span_m=a / nominal * info["lip_x_span_m"],
            median_forward_distance_m=a / nominal * info["visible_edge_forward_distance_m"][1],
        )
        for a in s["spacing_sensitivity_m"]
    ]
    metrics = dict(
        source_assumed_scale=True,
        range_is_camera_ray_distance=True,
        model_sha256=hashlib.sha256((out / "P572.glb").read_bytes()).hexdigest(),
        visible_asphalt_pixels=int((part == 1).sum()),
        visible_kerb_pixels=int((part == 2).sum()),
        context_core_pixels=counts,
        unknown_context_hits=int(np.isfinite(r[context != 0]).sum()),
        boundary_max_px=int(max(boundary_err)),
        boundary_rays=len(u),
        texture_mae_u8=err,
        assumed_1p60_m_track_paths=paths,
        spacing_sensitivity=edges,
    )
    (out / "quality.json").write_text(json.dumps(metrics, indent=2) + "\n")
    marked = rgb.copy()
    colors = [(254, 169, 52), (219, 137, 106), (56, 195, 126), (237, 89, 88)]
    for i, c in enumerate(colors, 1):
        contour, _ = cv2.findContours(
            np.uint8(context == i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(marked, contour, -1, c, 1, cv2.LINE_AA)
    cv2.polylines(marked, [e["pixels"].astype("i4")], False, (67, 229, 255), 2, cv2.LINE_AA)
    plate = Image.new("RGB", (1280, 687), (31, 36, 36))
    draw = ImageDraw.Draw(plate)
    plate.paste(Image.fromarray(rgb), (0, 27))
    plate.paste(Image.fromarray(marked), (640, 27))
    detail = Image.fromarray(render[325:427, :325]).resize((589, 185))
    plate.paste(detail, (12, 482))
    plate.paste(Image.fromarray(ortho).resize((332, 194)), (667, 480))
    for xy, label in [
        ((8, 8), "B4084 | Bob Embleton, 22 July 2007"),
        ((648, 8), "Visible fracture (cyan); pipe, soil, foliage and barrier cores"),
        ((10, 463), "Raycast of delivered mesh | near kerb and fractured road"),
        ((648, 463), "Orthophoto | road forward is up"),
    ]:
        draw.text(xy, label)
    draw.line((1030, 623, 1094, 623), fill="white", width=3)
    draw.text((1030, 630), "1 m (assumed scale)")
    draw.multiline_text(
        (1025, 512),
        "Dark field = outside\nreconstructed surface.\n\nNo inferred pit floor.",
        spacing=6,
    )
    plate.save(out / "overview.jpg", quality=94)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    check(parser.parse_args().out)
