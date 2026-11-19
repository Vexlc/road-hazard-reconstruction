"""Verify asset geometry, Gaussian formats, rendered frames and unchanged scene actors."""
import argparse
import json
from pathlib import Path
import struct

import imageio_ffmpeg
import numpy as np
from plyfile import PlyData
import torch


CASES = ["stereo_pothole_dataset1_02", "stereo_pothole_dataset2_25",
         "photo_manhole_missing", "photo_manhole_raised65"]


def check_glb(path):
    with path.open("rb") as stream:
        magic, version, file_size, json_size, chunk_type = struct.unpack("<4sIII4s", stream.read(20))
        assert magic == b"glTF" and version == 2 and file_size == path.stat().st_size
        assert chunk_type == b"JSON"
        info = json.loads(stream.read(json_size))
    assert all("bufferView" in image for image in info["images"])
    mins, maxes, triangles = [], [], 0
    for mesh in info["meshes"]:
        for primitive in mesh["primitives"]:
            vertices = info["accessors"][primitive["attributes"]["POSITION"]]
            indices = info["accessors"][primitive["indices"]]
            mins.append(vertices["min"])
            maxes.append(vertices["max"])
            assert primitive.get("mode", 4) == 4
            triangles += indices["count"] // 3
    low, high = np.min(mins, 0), np.max(maxes, 0)
    assert -1.2 < low[1] < -.008 and high[1] < .75
    return {"triangles": triangles, "embedded_textures": len(info["images"]),
            "bounds_y_up_m": [low.tolist(), high.tolist()]}


def check_ply(path, expected):
    data = PlyData.read(path)["vertex"].data
    assert len(data) == expected
    required = ["x", "y", "z", "opacity"] + [f"scale_{i}" for i in range(3)]
    required += [f"rot_{i}" for i in range(4)] + [f"f_dc_{i}" for i in range(3)]
    required += [f"f_rest_{i}" for i in range(9)]
    assert all(np.isfinite(data[field]).all() for field in required)
    assert np.abs(np.sum(np.stack([data[f"rot_{i}"] for i in range(4)]) ** 2, axis=0) - 1).max() < 1e-4
    return {"gaussians": len(data), "finite": True, "sh_degree": 1,
            "local_z_m": [float(np.min(data["z"])), float(np.max(data["z"]))]}


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return bool(left == right)


def rendering_parameters(component):
    if not isinstance(component, dict):
        return component
    training_keys = {
        "optimizer", "xyz_gradient_accum", "denom", "max_radii2D",
        "spatial_lr_scale", "active_sh_degree"
    }
    return {key: value for key, value in component.items() if key not in training_keys}


def check_video(path, frames, fps, size):
    n, time = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert n == frames, (path, n)
    assert abs(time - frames / fps) < .035, (path, time)
    probe = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(probe)
        assert tuple(meta["size"]) == size and abs(meta["fps"] - fps) < .015
    finally:
        probe.close()
    return {"frames": frames, "fps": fps, "resolution": size}


def verify(project):
    base = torch.load(project / "outputs/scene_B_055/checkpoint_7000.pth",
                      weights_only=False, map_location="cpu")
    stable = [key for key in base if key not in {"background", "iter"}]
    report = {}
    for name in CASES:
        asset = project / "outputs/road_hazards" / name
        scene = project / "outputs/road_hazard_insertions" / ("B_055_" + name)
        summary = json.loads((scene / "summary.json").read_text())
        assert summary["validation"]["checkpoint_reload_max_rgb_error"] < 1e-6
        model = torch.load(scene / "trained_model/iteration_7000.pth",
                           weights_only=False, map_location="cpu")
        assert set(stable) == set(model) - {"background", "iter"}
        assert stable and all(
            identical(rendering_parameters(base[key]), rendering_parameters(model[key])) for key in stable
        )
        del model
        glb = asset / ("pothole.glb" if "pothole" in name else name + ".glb")
        result = {
            "mesh": check_glb(glb),
            "gaussians": check_ply(asset / "gaussians/asset.ply", summary["assets"][0]["gaussians"]),
            "actors_and_non_background_parameters_equal": True,
            "checkpoint_reload_max_rgb_error": summary["validation"]["checkpoint_reload_max_rgb_error"],
            "drive": check_video(scene / "pure_3dgs.mp4", 80, 10, (1920, 1080)),
            "same_camera_before_after": check_video(scene / "before_after.mp4", 80, 10, (3840, 1080)),
            "close": check_video(scene / "closeup_3dgs.mp4", 88, 24, (1920, 1080))
        }
        report[name] = result
        # Correct the generic insertion exporter label for capture-derived sources.
        meta_path = asset / "gaussians/metadata.json"
        meta = json.loads(meta_path.read_text())
        capture = json.loads((asset / "capture_metadata.json").read_text())
        meta["source"] = capture["source"]
        meta["conversion"] = ("Measured stereo height field, projected source RGB, tangent splats"
                              if "pothole" in name else
                              "Rectified photographic texture on modeled mesh, anisotropic splats and ray-baked shadows")
        if "image_credit" in capture:
            meta["image_credit"] = capture["image_credit"]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print("PASS", name, result["mesh"]["triangles"], result["gaussians"]["gaussians"], flush=True)
    destination = project / "outputs/reproducibility/capture_to_3d/validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    verify(parser.parse_args().project)
