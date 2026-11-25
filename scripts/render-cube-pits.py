"""Render six native OpenGL framebuffers and sample their world-position passes with a hardware samplerCube."""

import argparse
import json
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
FACE_SIZE = 384
LENS_SIZE = 416
# OpenGL +X,-X,+Y,-Y,+Z,-Z. Data exported to NumPy are top row first.
AIMS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
RIGHTS = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]])
FRAMES = np.stack([RIGHTS, np.cross(AIMS, RIGHTS), AIMS], axis=-1).astype(float)
SHUFFLE = [4, 1, 3, 5, 0, 2]

VERT = """#version 330
in vec3 position;
in vec2 uv;
uniform mat4 clip_from_road;
out vec3 road;
out vec2 st;
void main(){ road=position; st=uv; gl_Position=clip_from_road*vec4(position,1); }
"""
FRAG = """#version 330
in vec3 road;
in vec2 st;
uniform float identity;
uniform sampler2D paint;
uniform mat3 apertures[2];
layout(location=0) out vec4 color;
layout(location=1) out vec4 position_id;
void main(){
    if (identity==3) {
        for (int i=0; i<2; i++) {
            vec2 uv=(apertures[i]*vec3(road.xy,1)).xy;
            if (all(lessThanEqual(abs(uv),vec2(1)))) discard;
        }
    }
    color=texture(paint,st);
    position_id=vec4(road,identity==3 ? 1 : identity);
}
"""
FULLSCREEN = """#version 330
void main(){
    vec2 p=vec2((gl_VertexID<<1)&2, gl_VertexID&2);
    gl_Position=vec4(p*2.0-1.0,0,1);
}
"""
LOOKUP = """#version 330
uniform samplerCube cube_pass;
uniform sampler2D bearings;
uniform mat3 lens_to_cube;
out vec4 reference;
void main(){
    vec3 direction=texelFetch(bearings,ivec2(gl_FragCoord.xy),0).xyz;
    reference=texture(cube_pass,lens_to_cube*direction);
}
"""


def rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


class Rasterizer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.sampler = self.ctx.program(
            vertex_shader=FULLSCREEN, fragment_shader=LOOKUP
        )
        self.screen = self.ctx.vertex_array(self.sampler, [])
        self.sampler["cube_pass"] = 0
        self.sampler["bearings"] = 1
        self.size = (FACE_SIZE, FACE_SIZE)
        self.rgb = self.ctx.texture(self.size, 4)
        self.xyzid = self.ctx.texture(self.size, 4, dtype="f4")
        self.depth = self.ctx.depth_texture(self.size)
        self.fbo = self.ctx.framebuffer([self.rgb, self.xyzid], self.depth)
        self.pass_cube = self.ctx.texture_cube(self.size, 4, dtype="f4")
        self.pass_cube.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def mesh(self, xyz, uv, triangles, img, code):
        texture = self.ctx.texture(
            img.size,
            3,
            img.convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes(),
            alignment=1,
        )
        texture.build_mipmaps()
        data = self.ctx.buffer(np.float32(np.column_stack([xyz, uv])).tobytes())
        faces = self.ctx.buffer(np.uint32(triangles).tobytes())
        vao = self.ctx.vertex_array(
            self.program, [(data, "3f 2f", "position", "uv")], faces
        )
        return vao, texture, code

    def render(self, draws, cube, reverse):
        near, far = 0.075, (np.inf if reverse else 42.0)
        az, bz = (
            (-1.0, 2 * near)
            if reverse
            else ((far + near) / (far - near), -2 * far * near / (far - near))
        )
        projection = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, az, bz], [0, 0, 1, 0]]
        )
        colors, depths, labels = [], [], []
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = ">" if reverse else "<"
        for face, rot in enumerate(FRAMES):
            pose = cube.copy()
            pose[:3, :3] = cube[:3, :3] @ rot
            self.fbo.use()
            self.fbo.clear(0.32, 0.38, 0.43, 0.0, depth=0.0 if reverse else 1.0)
            # World-position background has ID zero (alpha from the shared clear call).
            self.program["clip_from_road"].write(
                np.float32(projection @ np.linalg.inv(pose)).T.tobytes()
            )
            for vao, texture, code in draws:
                self.program["identity"] = code
                texture.use(0)
                vao.render()
            pixel = np.frombuffer(self.rgb.read(), "u1").reshape(
                FACE_SIZE, FACE_SIZE, 4
            )
            depths.append(
                np.frombuffer(self.depth.read(), "f4").reshape(FACE_SIZE, FACE_SIZE)[
                    ::-1
                ]
            )
            raw = self.xyzid.read()
            self.pass_cube.write(face, raw)
            labels.append(
                np.frombuffer(raw, "f4")
                .reshape(FACE_SIZE, FACE_SIZE, 4)[::-1, :, 3]
                .astype("u1")
            )
            colors.append(pixel[::-1, :, :3])
        ids = np.stack(labels)[SHUFFLE]
        return dict(
            rgb=np.stack(colors)[SHUFFLE],
            device_depth=np.stack(depths)[SHUFFLE],
            depth_valid=ids > 0,
            receiver_mask=ids == 1,
            occluder_mask=ids == 2,
            near_m=near,
            far_m=far,
            reversed_z=reverse,
            face_to_cube=FRAMES[SHUFFLE],
        )

    def lookup(self, rays, lens_to_cube):
        # Standard hardware cube sampling is independent of the Python face selector.
        h, w = rays.shape[:2]
        bearings = self.ctx.texture((w, h), 3, np.float32(rays).tobytes(), dtype="f4")
        out = self.ctx.texture((w, h), 4, dtype="f4")
        fb = self.ctx.framebuffer([out])
        fb.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sampler["lens_to_cube"].write(np.float32(lens_to_cube.T).tobytes())
        self.pass_cube.use(0)
        bearings.use(1)
        self.screen.render(vertices=3)
        raw = np.frombuffer(out.read(), "f4").reshape(h, w, 4)
        fb.release()
        out.release()
        bearings.release()
        return raw

    def reference(self, rays, orient, eye):
        raw = self.lookup(rays, orient)
        ids = raw[..., 3].astype("u1")
        radial = np.where(
            ids > 0, np.linalg.norm(raw[..., :3] - eye, axis=-1), np.inf
        ).astype("f4")
        return ids, radial

    def addresses(self, rays, orient):
        y, x = np.indices((FACE_SIZE, FACE_SIZE))
        for face in range(6):
            texels = np.stack(
                [
                    np.full_like(x, SHUFFLE.index(face)),
                    x,
                    FACE_SIZE - 1 - y,
                    np.ones_like(x),
                ],
                axis=-1,
            )
            self.pass_cube.write(face, np.float32(texels).tobytes())
        return self.lookup(rays, orient)[..., :3].astype(np.int16)


def geometry(render, models, layout):
    asphalt = Image.open(
        ROOT / "gen2hole/examples/traffic-cones/asphalt_02_diff_1k.jpg"
    ).convert("RGB")

    def road(xy, faces):
        return render.mesh(np.c_[xy, np.zeros(len(xy))], xy * 0.6, faces, asphalt, 1)

    ground = road(rect(-32, 32, -32, 32), [[0, 1, 2], [0, 2, 3]])
    # Raised thin road-survey bar crosses the two holes and reveals occlusion errors.
    bar = trimesh.creation.box(extents=[3.85, 0.052, 0.074])
    bar.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(7), [0, 0, 1])
    )
    bar.apply_translation([0, 0.02, 0.11])
    marker = render.mesh(
        bar.vertices,
        np.zeros((len(bar.vertices), 2)),
        bar.faces,
        Image.new("RGB", (1, 1), (238, 154, 40)),
        2,
    )
    native = []
    apertures = []
    for i, member in enumerate(layout["instances"], 1):
        key = member["asset"]
        folder = models / key
        if not (folder / "asset.glb").is_file():
            (folder,) = models.glob(f"*/series/assets/*/{key}")
        yaw = Rotation.from_euler("z", member["heading_deg"], degrees=True).as_matrix()
        shift = np.r_[member["xy_m"], 0.0]
        scene = trimesh.load(folder / "asset.glb", force="scene", process=False)
        for node in scene.graph.nodes_geometry:
            pose, name = scene.graph[node]
            mesh = scene.geometry[name]
            xyz = trimesh.transform_points(mesh.vertices, pose)
            xyz = xyz[:, [0, 2, 1]] * [1, -1, 1] @ yaw.T + shift
            code = {
                "Fracture lip and exposed bed": i * 32 + 2,
                "Road adjoining the fracture": 1,
            }[node]
            native.append(
                render.mesh(
                    xyz,
                    mesh.visual.uv,
                    mesh.faces,
                    mesh.visual.material.baseColorTexture,
                    code,
                )
            )
        with np.load(folder / "surface.npz") as f:
            w, h = float(f["width"]) / 2, float(f["length"]) / 2
        local = np.eye(3)
        local[:2, :2] = yaw[:2, :2].T / np.array([[w], [h]])
        local[:2, 2] = -local[:2, :2] @ shift[:2]
        apertures.append(local)
    # Discard precisely the two rotated rectangular aprons before drawing their GLBs.
    # Joining their corners to an axis-aligned road tile creates overlapping triangles.
    render.program["apertures"].write(
        np.float32(apertures).transpose(0, 2, 1).tobytes()
    )
    native.append((*ground[:2], 3))
    return [ground, marker], native, marker


def make_lens():
    row, col = np.mgrid[:LENS_SIZE, :LENS_SIZE] + 0.5
    x, y = (
        (col - LENS_SIZE / 2) / (LENS_SIZE / 2),
        (row - LENS_SIZE / 2) / (LENS_SIZE / 2),
    )
    r = np.hypot(x, y)
    angle = r * np.deg2rad(111.0)
    return np.stack(
        [np.sin(angle) * x / r, np.sin(angle) * y / r, np.cos(angle)], axis=-1
    ).astype("f4"), r > 1


def capture(models, recipe, out):
    layout = json.loads(recipe.read_text())
    raster = Rasterizer()
    host, native, marker = geometry(raster, models, layout)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("zyx", [31, -4, 2], degrees=True).as_matrix()
    world[:3, 3] = [4.3, -7.1, 1.45]
    rays, masked = make_lens()
    for shot, yaw, pitch, eye, reverse in [
        ("side", 77, 27, [0, -1.8, 1.12], False),
        ("rear", 178, 31, [0, 2.05, 1.24], True),
    ]:
        yaw, pitch = np.deg2rad([yaw, pitch])
        forward = np.array(
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)]
        )
        right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
        lens = np.eye(4)
        lens[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
        lens[:3, 3] = eye
        cube = lens.copy()
        aimed = np.r_[layout["instances"][0]["xy_m"], 0.0] - eye
        aimed /= np.linalg.norm(aimed)
        cross = np.cross([0.0, 0.0, 1.0], aimed)
        cross /= np.linalg.norm(cross)
        first, third = (aimed + cross) / np.sqrt(2), (aimed - cross) / np.sqrt(2)
        cube[:3, :3] = np.stack([first, np.cross(third, first), third], axis=1)
        dest = out / layout["id"] / shot
        dest.mkdir(exist_ok=True, parents=True)
        faces = raster.render(host, cube, reverse)
        np.savez_compressed(dest / "faces.npz", **faces, camera_to_world=world @ cube)
        orient = cube[:3, :3].T @ lens[:3, :3]
        host_id, host_range = raster.reference(rays, orient, eye)
        address = raster.addresses(rays, orient)
        np.savez_compressed(
            dest / "lens.npz",
            rays_camera=rays,
            protected=masked,
            camera_to_world=world @ lens,
            cube_texel=address,
        )
        raster.render(native, cube, reverse)
        clear_id, _ = raster.reference(rays, orient, eye)
        raster.render([*native, marker], cube, reverse)
        native_id, native_range = raster.reference(rays, orient, eye)
        np.savez_compressed(
            dest / "reference.npz",
            host_id=host_id,
            host_range=host_range,
            clear_id=clear_id,
            native_id=native_id,
            native_range=native_range,
        )
    (out / "frame.json").write_text(
        json.dumps(
            dict(
                seed=world[:3, 3].tolist(),
                heading=world[:3, 0].tolist(),
                up=world[:3, 2].tolist(),
                renderer=raster.ctx.info["GL_RENDERER"],
            ),
            indent=2,
        )
        + "\n"
    )
    raster.ctx.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("layouts", type=Path, nargs="+")
    a = p.parse_args()
    for layout in a.layouts:
        capture(a.models, layout, a.out)
