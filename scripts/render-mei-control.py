"""Forward-project the actual surveyed GLBs through OpenCV omnidir and rasterize them in raw KITTI pixels."""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import moderngl
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mei_camera import MeiCamera

VERT = """#version 330
in vec2 raw_pixel;
in vec3 camera_xyz;
uniform vec2 first_pixel;
uniform vec2 image_size;
out vec3 location;
void main() {
    vec2 pixel = raw_pixel - first_pixel + vec2(0.5);
    gl_Position = vec4(2.0*pixel/image_size - 1.0, 0.0, 1.0);
    location = camera_xyz;
}
"""
FRAG = """#version 330
in vec3 location;
uniform float instance_id;
uniform float min_cosine;
out vec2 id_range;
void main() {
    float radius = length(location);
    if (location.z <= radius*min_cosine || radius >= 100.0) discard;
    gl_FragDepth = radius/100.0;
    id_range = vec2(instance_id, radius);
}
"""


def read(path):
    return json.loads(path.read_text())


class NativeCalibration(NamedTuple):
    intrinsic: np.ndarray
    distortion: np.ndarray
    xi: float
    size: tuple


def native_calibration(path):
    # Read the vendor file independently of the inverse-camera adapter under test.
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        p = storage.getNode("projection_parameters")
        d = storage.getNode("distortion_parameters")
        matrix = np.eye(3)
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = [
            p.getNode(k).real() for k in ("gamma1", "gamma2", "u0", "v0")
        ]
        coeffs = np.array([d.getNode(k).real() for k in ("k1", "k2", "p1", "p2")])
        xi = storage.getNode("mirror_parameters").getNode("xi").real()
        size = tuple(
            int(storage.getNode(k).real()) for k in ("image_width", "image_height")
        )
        return NativeCalibration(matrix, coeffs, xi, size)
    finally:
        storage.release()


def project(points, lens):
    uvs = []
    for block in np.array_split(
        np.ascontiguousarray(points, dtype="f8"), max(1, (len(points) + 9999) // 10000)
    ):
        uv, _ = cv2.omnidir.projectPoints(
            block[:, None],
            np.zeros(3),
            np.zeros(3),
            lens.intrinsic,
            lens.xi,
            lens.distortion,
        )
        uvs.append(uv[:, 0])
    return np.concatenate(uvs)


def camera_limits(path):
    lens = native_calibration(path)
    inverse = MeiCamera.read(path)
    w, h = lens.size
    # Independent forward projection of the mirror's limiting circle determines the valid native image.
    phi = np.linspace(0, 2 * np.pi, 16384, endpoint=False)
    z = -1 / lens.xi
    r = np.sqrt(1 - z * z)
    boundary = project(
        np.c_[r * np.cos(phi), r * np.sin(phi), np.full_like(phi, z)], lens
    )
    origin = lens.intrinsic[:2, 2]
    bearing = np.arctan2(*(boundary - origin).T[::-1])
    order = np.argsort(bearing)
    phase = bearing[order]
    extent = np.linalg.norm(boundary - origin, axis=-1)[order]
    totals = {
        "checked": 0,
        "visible": 0,
        "behind_optical_plane": 0,
        "domain_disagreements": 0,
        "max_pixel_error": 0.0,
    }
    for top in range(0, h, 56):
        count = min(56, h - top)
        directions, mask = inverse.unproject(0, top, w, count, 0.25, -0.25)
        rays, valid = directions[::7, ::7], mask[::7, ::7]
        y, x = np.mgrid[top : top + count : 7, :w:7]
        uv = np.stack([x + 0.25, y - 0.25], axis=-1)
        delta = uv - origin
        angle = np.arctan2(delta[..., 1], delta[..., 0])
        radius = np.interp(angle, phase, extent, period=2 * np.pi)
        distance_to_boundary = radius - np.linalg.norm(delta, axis=-1)
        reliable = np.abs(distance_to_boundary) > 0.005
        totals["domain_disagreements"] += int(
            np.count_nonzero(reliable & (valid != (distance_to_boundary >= 0)))
        )
        totals["checked"] += int(valid.size)
        totals["visible"] += int(valid.sum())
        totals["behind_optical_plane"] += int(
            np.count_nonzero(valid & (rays[..., 2] < 0))
        )
        if valid.any():
            err = np.linalg.norm(project(rays[valid], lens) - uv[valid], axis=-1).max()
            totals["max_pixel_error"] = max(totals["max_pixel_error"], float(err))
    if (
        totals["domain_disagreements"]
        or totals["max_pixel_error"] > 0.001
        or totals["behind_optical_plane"] < 100
    ):
        raise ValueError(f"Raw lens failed C++ forward projection: {totals}")
    return totals


def render(models, folder, output):
    pair = folder.parent.parent.name
    members = read(folder.parent / "appearance.json")["instances"]
    lens = native_calibration(folder / "calibration.yaml")
    with np.load(folder / "view.npz") as packet:
        pose = packet["camera_to_world"]
        left, top, w, h = map(int, packet["sensor_window_xywh"])
        offsets = packet["sample_offsets_px"]
        protected = packet["protected"]
        near = packet["host_range"]
    ctx = moderngl.create_standalone_context(backend="egl")
    ctx.enable(moderngl.DEPTH_TEST)
    shader = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    shader["image_size"] = (w, h)
    shader["min_cosine"] = -1 / lens.xi
    fbo = ctx.framebuffer(
        [ctx.texture((w, h), 2, dtype="f4")], ctx.depth_renderbuffer((w, h))
    )
    fbo.use()
    geometry = []
    for index, item in enumerate(members, 1):
        key = item["asset"]
        path = models / key / "asset.glb"
        if not path.exists():
            (path,) = models.glob(f"*/series/assets/*/{key}/asset.glb")
        anchor = item["anchor"]
        scene = trimesh.load(path, force="scene", process=False)
        for node in scene.graph.nodes_geometry:
            t, name = scene.graph[node]
            mesh = scene.geometry[name]
            xyz = trimesh.transform_points(mesh.vertices, t)[:, [0, 2, 1]] * [1, -1, 1]
            world = xyz @ np.array(anchor["basis_columns"]).T + anchor["center_world"]
            camera = (world - pose[:3, 3]) @ pose[:3, :3]
            pixel = project(camera, lens)
            data = ctx.buffer(np.c_[pixel, camera].astype("f4").tobytes())
            faces = ctx.buffer(np.uint32(mesh.faces).tobytes())
            vao = ctx.vertex_array(
                shader, [(data, "2f 3f", "raw_pixel", "camera_xyz")], faces
            )
            code = {
                "Road adjoining the fracture": 0,
                "Fracture lip and exposed bed": index,
            }[node]
            geometry.append((vao, code))
    labels, ranges = [], []
    for (dx, dy), protect in zip(offsets, protected):
        fbo.clear(0, 0, 0, 0, depth=1.0)
        shader["first_pixel"] = (left + dx, top + dy)
        for vao, code in geometry:
            shader["instance_id"] = float(code)
            vao.render(moderngl.TRIANGLES)
        result = np.frombuffer(fbo.read(components=2, dtype="f4"), dtype="f4").reshape(
            h, w, 2
        )
        code, radial = result.transpose(2, 0, 1)
        use = (code > 0) & ~protect & (near >= radial - 1e-5)
        labels.append(np.where(use, code, 0).astype("u2"))
        ranges.append(np.where(use, radial, np.inf))
    ranges = np.stack(ranges)
    nearest = ranges.argmin(0)[None]
    label = np.take_along_axis(np.stack(labels), nearest, axis=0)[0]
    distance = ranges.min(0)
    distance[label == 0] = np.nan
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, instance=label, range=distance)
    print(
        pair,
        folder.parent.name,
        "native raster pixels",
        np.bincount(label.ravel()).tolist(),
        flush=True,
    )
    ctx.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--calibrations",
        type=Path,
        required=True,
        help="Original left and right MEI YAML files",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {
        k: camera_limits(args.calibrations / (k + ".yaml"))
        for k in ("image_02", "image_03")
    }
    (args.out / "lenses.json").write_text(json.dumps(stats, indent=2) + "\n")
    for folder in sorted(args.packets.glob("*/*/image_03")):
        pair, site, _ = folder.relative_to(args.packets).parts
        render(args.assets, folder, args.out / pair / site / "raster.npz")
