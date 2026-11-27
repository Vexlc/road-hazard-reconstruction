"""Render a synthetic approach on the northern intact carriageway and retain metric ray data."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from build import ASSET, Mesh, atlas, face_up, flatten_scene, write_json
from camera import Survey, cams
from plates import SCALE
from photo_mesh import normalize
from road_geometry import glb_triangles


def approach(survey, cam, x, y):
    # Conservative inner pavement edges on photograph A, in its 720-wide annotation frame.
    left = np.interp(y, [404, 420, 501], [340, 344, 349])
    right = np.interp(y, [404, 420, 501], [420, 422, 435])
    uv = np.stack([left + (right - left) * x, y], axis=-1).reshape(-1, 2) * SCALE
    points = cam.at(uv, 4.0)
    for _ in range(6):
        points = cam.at(uv, survey.height("before", points))
    return points


def context(survey, cam, centre, out):
    y, x = np.mgrid[404:501:49j, 0:1:24j]
    points = approach(survey, cam, x, y)
    indices = np.arange(x.size).reshape(x.shape)
    a, b, c, d = [
        q.ravel() for q in (indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:])
    ]
    faces = np.vstack([np.c_[a, b, c], np.c_[b, c, d]])
    accepted = (
        (abs(survey.height("before", points) - survey.height("after", points)) < 0.4)
        & (survey.height("before_dist", points) < 2.5)
        & (survey.height("after_dist", points) < 2.5)
    )
    faces = faces[accepted[faces].all(1)]
    mesh = Mesh()
    tile, uv = atlas(points, 0, "approach-context")
    mat = mesh.material("Approach worn asphalt", tile, 0.94, 0, 0.32)
    mesh.add(
        "Road adjoining the fracture",
        points - centre,
        face_up(points, faces),
        uv(points),
        mat,
        True,
    )
    (out / "approach.glb").write_bytes(mesh.glb_bytes())
    flatten_scene(out / "approach.glb")


class View:
    def __init__(self, paths, ids):
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.faces = []
        self.palette = []
        self.ids = []
        for path in paths:
            for name, v, f, uv, buf in glb_triangles(path, names=True, appearance=True):
                self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
                self.ids.append(ids.get(name, 1))
                self.faces.append(uv[f])
                self.palette.append(
                    cv2.imdecode(np.frombuffer(buf, "u1"), cv2.IMREAD_COLOR)[:, :, ::-1]
                )

    def cast(self, eye, rays, colour=False):
        r = np.concatenate((np.broadcast_to(eye, rays.shape), rays), axis=-1).astype("f4")
        hit = self.scene.cast_rays(o3d.core.Tensor(r), nthreads=2)
        d = hit["t_hit"].numpy()
        g = hit["geometry_ids"].numpy()
        mask = np.isfinite(d)
        ids = np.zeros(d.shape, "u1")
        ids[mask] = np.array(self.ids)[g[mask]]
        if not colour:
            return d, ids
        rgb = np.empty(rays.shape, "u1")
        rgb[:] = [222, 228, 232]
        normal = hit["primitive_normals"].numpy()
        p = hit["primitive_ids"].numpy()
        bary = hit["primitive_uvs"].numpy()
        for j, (uv, tex) in enumerate(zip(self.faces, self.palette)):
            k = g == j
            if not k.any():
                continue
            b = bary[k]
            t = np.sum(uv[p[k]] * np.c_[1 - b.sum(1), b][..., None], axis=1)
            h, w = tex.shape[:2]
            col = tex[
                (t[:, 1] * h).astype(int).clip(0, h - 1), (t[:, 0] * w).astype(int).clip(0, w - 1)
            ]
            light = 0.49 + 0.58 * np.maximum(0, normal[k] @ normalize(np.array([-0.6, 0.15, 0.85])))
            rgb[k] = np.clip(col * light[:, None], 0, 255)
        return d, ids, rgb


def axes(direction, pitch):
    horizontal = normalize(np.r_[direction[:2], 0])
    right = np.cross(horizontal, [0, 0, 1.0])
    forward = np.cos(pitch) * horizontal + [0, 0, -np.sin(pitch)]
    down = np.cross(forward, right)
    return right, down, forward


def replay(model, out):
    out.mkdir(parents=True, exist_ok=True)
    asset = model / ASSET
    meta = json.loads((asset / "metadata.json").read_text())
    centre = meta["survey_to_model_subtract_m"]
    survey = Survey()
    cam = cams(meta["camera_parameters_survey"])[0]
    context(survey, cam, centre, out)
    ids = {"Vine Road plate " + name: i + 2 for i, name in enumerate(meta["included_panels"])}
    full = View([asset / "asset.glb", out / "approach.glb"], ids)
    strict = View([asset / "supported.glb", out / "approach.glb"], ids)
    h, w = 360, 640
    y, x = np.indices((h, w))
    f = 490.0
    cx = (w - 1) / 2
    cy = (h - 1) / 2
    az, el = np.meshgrid(np.deg2rad(np.linspace(-42, 42, 337)), np.deg2rad(np.linspace(-23, 3, 32)))
    frames = []
    depths = []
    labels = []
    scans = []
    good = []
    parts = []
    sids = []
    beam_dirs = []
    samples = []
    for i, row in enumerate(np.linspace(493, 418, 12)):
        pt = approach(survey, cam, np.array([0.67, 0.67]), np.array([row, row - 2]))
        eye = pt[0] - centre + [0, 0, 1.55]
        right, down, forward = axes(pt[1] - pt[0], np.deg2rad(4))
        rays = normalize(
            ((x - cx) / f)[..., None] * right + ((y - cy) / f)[..., None] * down + forward
        )
        dist, part, rgb = full.cast(eye, rays, True)
        assert (part >= 2).any() and (i > 0 or (part == 1).any())
        Image.fromarray(rgb).save(out / f"frame-{i:02}.jpg", quality=91)
        flat_right, _, flat_fwd = axes(pt[1] - pt[0], 0.0)
        lr = np.cos(el)[..., None] * (
            np.sin(az)[..., None] * flat_right + np.cos(az)[..., None] * flat_fwd
        )
        lr[..., 2] += np.sin(el)
        d1, id1 = full.cast(eye, lr)
        d2, id2 = strict.cast(eye, lr)
        depths.append(dist)
        labels.append(part)
        scans.append(d1)
        good.append(d2)
        parts.append(id1)
        sids.append(id2)
        beam_dirs.append(lr)
        frames.append(
            dict(
                eye_local_m=eye.tolist(),
                right_down_forward=np.array([right, down, forward]).tolist(),
            )
        )
        samples.append(
            dict(
                frame=i,
                visible_panel_pixels=int((part >= 2).sum()),
                model_lidar_hits=int((id1 >= 2).sum()),
                support_lidar_hits=int((id2 >= 2).sum()),
            )
        )
    np.savez_compressed(
        out / "rays.npz",
        camera_range_m=depths,
        camera_part=labels,
        beam_dirs=np.float32(beam_dirs),
        model_range_m=scans,
        model_part=parts,
        supported_range_m=good,
        supported_part=sids,
        frame_eye_local_m=[f["eye_local_m"] for f in frames],
        azimuth_rad=az,
        elevation_rad=el,
    )
    write_json(
        out / "camera.json",
        dict(
            size=[w, h],
            K=[[f, 0, cx], [0, f, cy], [0, 0, 1]],
            frames=frames,
            plate_codes=ids,
            no_hit=0,
            approach=1,
            camera_height_m=1.55,
            ray_metric="distance along unit ray",
            samples=samples,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "out"):
        p.add_argument(key, type=Path)
    replay(**vars(p.parse_args()))
