"""Render a near-camera sweep of the installed asset and a canonical Gaussian preview."""
from pathlib import Path
import json
import sys

from gsplat.rendering import rasterization
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/street_crafter"), str(ROOT / "work/asset_pipeline")]

from create_scene import create_scene
from gaussian_assets import tensor
from street_gaussian.config import cfg
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer


def unit(vector):
    return vector / np.linalg.norm(vector)


def camera_pose(position, target, up):
    forward = unit(target - position)
    right = unit(np.cross(forward, up))
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    pose[:3, 3] = position
    return tensor(pose)


def to_image(rgb):
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("Non-finite render")
    return np.uint8(rgb.clamp(0, 1).cpu().numpy() * 255 + .5)


@torch.no_grad()
def render():
    out = Path(cfg.model_path)
    meta = json.loads((out / "summary.json").read_text())
    asset = meta["assets"][0]
    name, placement = asset["name"], asset["placement"]
    pothole = "pothole" in name
    source = ROOT / "outputs/road_hazards" / name
    scene = create_scene()
    renderer = StreetGaussianRenderer()
    cameras = sorted(scene.getTrainCameras() + scene.getTestCameras(), key=lambda c: c.meta["frame"])
    camera = cameras[40]
    initial_pose = camera.get_extrinsic().cpu().numpy()
    center = np.array(placement["center_world"])
    basis = np.array(placement["basis_columns"])
    up = basis[:, 2]
    # Use the same side of the road seen on the driving pass.
    offset = initial_pose[:3, 3] - center
    direction = unit(offset - (offset @ up) * up)
    across = np.cross(up, direction)

    if pothole:
        radius, elevation, target_height, focal = 1.18, 1.12, -.007, 2200
    else:
        radius, elevation, target_height, focal = 2.55, 1.78, .04, 1775
    width, height = 1920, 1080
    intrinsics = tensor([[focal, 0, width * .5], [0, focal, height * .5], [0, 0, 1]])
    camera.image_width, camera.image_height = width, height
    camera.set_intrinsic(intrinsics)
    frames = 88
    raised = "raised" in name
    if raised:
        fraction = .5 - .5 * np.cos(np.linspace(0, np.pi, frames))
        angles = np.deg2rad(-23 + 177 * fraction)
    else:
        angles = np.deg2rad(31 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, frames)))
    target = center + target_height * up
    output = out / "closeup_3dgs.mp4"
    writer = imageio.get_writer(str(output), fps=24, codec="libx264", quality=9, macro_block_size=2)
    hero_index = 16 if raised else frames // 4
    for index, angle in enumerate(angles):
        position = center + radius * (np.cos(angle) * direction + np.sin(angle) * across) + elevation * up
        camera.set_extrinsic(camera_pose(position, target, up))
        result = renderer.render(camera, scene.gaussians)
        frame = to_image(result["rgb"].permute(1, 2, 0))
        writer.append_data(frame)
        if index == hero_index:
            Image.fromarray(frame).save(out / "closeup.jpg", quality=95)
        if raised and index == 80:
            Image.fromarray(frame).save(out / "closeup_lid_face.jpg", quality=95)
    writer.close()

    state = torch.load(source / "gaussians/asset.pth", weights_only=True, map_location="cuda")
    if pothole:
        position = np.array([.059, -.235, .245])
        target = np.array([0., 0., -.005])
        focal = 1670
    else:
        position = np.array([.96, -1.36, 1.62])
        target = np.array([0., 0., -.04])
        focal = 1190
    pose = camera_pose(position, target, np.array([0., 0., 1.]))
    k = tensor([[focal, 0, 600], [0, focal, 400], [0, 0, 1]])
    coeffs = torch.cat([state["feature_dc"], state["feature_rest"]], dim=1)
    arguments = (state["xyz"], state["rotation"], state["scaling"].exp(), state["opacity"].sigmoid()[:, 0])
    opts = dict(sh_degree=1, packed=False, backgrounds=tensor([[.30, .33, .36]]),
                rasterize_mode="antialiased")
    rgb, _, _ = rasterization(*arguments, coeffs, torch.linalg.inv(pose)[None], k[None], 1200, 800, **opts)
    Image.fromarray(to_image(rgb[0])).save(source / "gaussians/preview.png")

    # Neutral presentation color makes the geometry easier to compare to the real source.
    points = state["xyz"]
    depth = (-points[:, 2] / (.05 if pothole else .98)).clamp(0, 1)
    normals = torch.stack([
        2 * (state["rotation"][:, 1] * state["rotation"][:, 3] + state["rotation"][:, 0] * state["rotation"][:, 2]),
        2 * (state["rotation"][:, 2] * state["rotation"][:, 3] - state["rotation"][:, 0] * state["rotation"][:, 1]),
        1 - 2 * (state["rotation"][:, 1] ** 2 + state["rotation"][:, 2] ** 2)
    ], dim=-1)
    lambert = (normals @ tensor(unit(np.array([-.6, -.4, .9])))).clamp(0, 1)
    tint = tensor([.77, .77, .73])[None] * (.40 + .60 * lambert[:, None])
    tint = tint * (1 - depth[:, None] * .5) + .5 * depth[:, None] * tensor([.18, .43, .59])
    rgb, _, _ = rasterization(*arguments, tint, torch.linalg.inv(pose)[None], k[None], 1200, 800,
                              packed=False, backgrounds=tensor([[.19, .22, .26]]),
                              rasterize_mode="antialiased")
    Image.fromarray(to_image(rgb[0])).save(source / "geometry.jpg", quality=95)
    (out / "closeup_camera.json").write_text(json.dumps(
        {"frame": 40, "source": "joint 3DGS; virtual orbit around the existing world position",
         "width": width, "height": height, "fps": 24, "frames": frames, "focal_pixels": float(intrinsics[0, 0]),
         "camera_height_above_road_m": elevation, "camera_radius_m": radius,
         "asset_scale_multiplier": 1}, indent=2) + "\n")


if __name__ == "__main__":
    render()
