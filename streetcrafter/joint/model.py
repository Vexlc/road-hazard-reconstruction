"""Differentiable StreetGS scene composition, SC-GS node warp and 3DGUT rasterization."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
from pytorch3d.ops import sample_farthest_points
from scipy.spatial.transform import Rotation
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.models.gaussian_model import GaussianModel
from street_gaussian.utils.graphics_utils import BasicPointCloud
from utils.time_utils import ControlNodeWarp
from threedgut_tracer import Tracer


class RenderGaussians:
    """3DGUT's tensor interface. Every tensor retains its StreetGS/SC-GS autograd history."""

    ray_feature_dim = 3

    def __init__(self, xyz, rotation, scale, density, features, degree):
        self.positions = xyz
        self.num_gaussians = len(xyz)
        self.rotation, self.scale, self.density, self.features = rotation, scale, density, features
        self.n_active_features = degree

    def get_rotation(self):
        return self.rotation

    def get_scale(self):
        return self.scale

    def get_density(self):
        return self.density

    def get_features(self):
        return self.features


class JointScene(nn.Module):
    def __init__(self, metadata, render_config):
        super().__init__()
        self.graph = StreetGaussianModel(metadata)
        self.warp = ControlNodeWarp(
            is_blender=False, node_num=128, K=4, hyper_dim=0, with_arap_loss=False, max_d_scale=2
        ).cuda()
        self.renderer = Tracer(render_config)
        self.metadata = metadata
        self.register_buffer(
            "center", torch.as_tensor(metadata["scene_center"], device="cuda", dtype=torch.float32)
        )
        self.units = max(2.0, float(metadata["scene_radius"]) / 2)
        self.degree = 0

    def initialize(self, seeds):
        support = []
        for name in self.graph.model_name_id:
            layer = getattr(self.graph, name)
            cloud = seeds[name]
            # All layers have measured colors and Fourier dimension one.
            # Use the original Gaussian initializer also for sparse tracked objects.
            pcd = BasicPointCloud(cloud[:, :3], cloud[:, 3:] / 255.0, np.zeros_like(cloud[:, :3]))
            GaussianModel.create_from_pcd(layer, pcd, 1.0)
            if name == "background":
                continue
            tracks = self.metadata["camera_tracklets"][0, :, layer.obj_meta["id"]]
            xyz = cloud[:: max(1, len(cloud) // 128), :3]
            valid = tracks[tracks[:, -1] == 1]
            for t in valid[:: max(1, len(valid) // 8)]:
                rot = Rotation.from_quat(t[[4, 5, 6, 3]]).as_matrix()
                support.append(xyz @ rot.T + t[:3])
        xyz = torch.as_tensor(np.concatenate(support), device="cuda", dtype=torch.float32)
        xyz = (xyz - self.center) / self.units
        if len(xyz) < self.warp.node_num:
            raise ValueError("Insufficient node support")
        nodes, _ = sample_farthest_points(xyz[None].contiguous(), K=self.warp.node_num)
        with torch.no_grad():
            self.warp.nodes.copy_(nodes[0])
            distances = torch.cdist(nodes[0], nodes[0])
            radius = distances.topk(4, largest=False).values[:, -1].clamp_min(0.04)
            self.warp._node_radius.copy_(radius.log())
            self.warp.inited.fill_(True)

    def exposure(self, frame, cam, t, heldout, w2c, k):
        meta = dict(frame=frame, frame_idx=frame, cam=cam, timestamp=t, is_val=heldout, is_novel_view=False)
        visible = ["background"]
        for name in self.graph.obj_list:
            obj = getattr(self.graph, name)
            track = self.metadata["camera_tracklets"][cam, frame, obj.obj_meta["id"]]
            if track[-1] != 1:
                continue
            if obj.obj_meta["class"] != "untracked residual":
                x, y, z = w2c[:3, :3] @ track[:3] + w2c[:3, 3]
                # Conservative sphere/frustum intersection; annotated scenes use pinhole cameras.
                radius = np.linalg.norm(obj.bbox) * 0.55 + 1.0
                _, w, h, fx, fy, cx, cy = k
                if (
                    z < -radius
                    or x * fx + cx * z < -radius * np.hypot(fx, cx)
                    or (w - cx) * z - x * fx < -radius * np.hypot(fx, w - cx)
                    or y * fy + cy * z < -radius * np.hypot(fy, cy)
                    or (h - cy) * z - y * fy < -radius * np.hypot(fy, h - cy)
                ):
                    continue
            visible.append(name)
        return SimpleNamespace(meta=meta), visible

    def forward(self, batch, exposure, time, disable_warp=False):
        camera, visible = exposure
        graph = self.graph
        graph.set_visibility(visible)
        graph.parse_camera(camera)
        xyz, rot, scale = graph.get_xyz, graph.get_rotation, graph.get_scaling
        n = len(graph.background.get_xyz)
        delta = None
        if len(xyz) > n and not disable_warp:
            field = self.warp(
                (xyz[n:] - self.center) / self.units,
                time,
                feature=None,
                motion_mask=1.0,
                is_training=self.training,
            )
            delta = 1.5 * torch.tanh(field["d_xyz"])
            xyz = torch.cat((xyz[:n], xyz[n:] + delta))
            rot = torch.cat((rot[:n], F.normalize(rot[n:] + field["d_rotation"], dim=-1)))
            scale = torch.cat((scale[:n], scale[n:] * field["d_scaling"].exp()))
        particles = RenderGaussians(
            xyz, rot, scale, graph.get_opacity, graph.get_features.flatten(1), self.degree
        )
        out = self.renderer.render(particles, batch, train=self.training)
        # Directions come from the calibrated 3DGUT camera (including fisheye).
        rays = batch.rays_dir @ batch.T_to_world[0, :3, :3].T
        sky = dr.texture(
            graph.sky_cubemap.sky_cube_map.sigmoid()[None],
            rays.contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )
        rgb = out["pred_features"] + (1.0 - out["pred_opacity"]) * sky
        return rgb, delta

    def parameters_for_fit(self):
        rates = {
            "_xyz": 0.003,
            "_features_dc": 0.008,
            "_features_rest": 0.001,
            "_scaling": 0.008,
            "_rotation": 0.002,
            "_opacity": 0.025,
        }
        groups = [
            dict(params=[getattr(getattr(self.graph, n), key) for n in self.graph.model_name_id], lr=lr)
            for key, lr in rates.items()
        ]
        groups.extend(
            [
                dict(params=self.warp.network.parameters(), lr=0.00018),
                dict(params=[self.warp._node_radius, self.warp._node_weight], lr=0.0007),
                dict(params=[self.graph.sky_cubemap.sky_cube_map], lr=0.015),
            ]
        )
        return groups

    def checkpoint(self):
        layers = {
            n: {k: v.detach().cpu() for k, v in getattr(self.graph, n).named_parameters()}
            for n in self.graph.model_name_id
        }
        return dict(
            metadata=self.metadata,
            layers=layers,
            sky=self.graph.sky_cubemap.sky_cube_map.detach().cpu(),
            warp={k: v.detach().cpu() for k, v in self.warp.state_dict().items()},
            degree=self.degree,
        )

    def restore(self, state):
        for n, params in state["layers"].items():
            layer = getattr(self.graph, n)
            for k, v in params.items():
                setattr(layer, k, nn.Parameter(v.cuda()))
        self.graph.sky_cubemap.sky_cube_map.data.copy_(state["sky"])
        self.warp.load_state_dict({k: v.cuda() for k, v in state["warp"].items()})
        self.degree = state["degree"]
