"""Verify all PandaSet renders and compare exported scene actors to the source models."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image
import numpy as np
import torch

OMIT_TRAINING = {"optimizer", "xyz_gradient_accum", "denom", "max_radii2D", "spatial_lr_scale", "active_sh_degree"}


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        aa, bb = {k for k in a if k not in OMIT_TRAINING}, {k for k in b if k not in OMIT_TRAINING}
        return aa == bb and all(exact(a[k], b[k]) for k in aa)
    return a == b


def frames(path, expected, rate):
    count, elapsed = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == expected and abs(elapsed-count/rate) < .06, (path, count, elapsed, expected)


def run(project, work):
    root = project / "outputs/road_hazard_expansion"
    baseline = {}
    reports = []
    for path in sorted((root / "scene_trials").glob("*/summary.json")):
        folder = path.parent
        summary = json.loads(path.read_text())
        frames(folder / "drive.mp4", 80, 10)
        frames(folder / "closeup.mp4", 45, 20)
        before_after = np.asarray(Image.open(folder / "before_after_near.jpg").convert("RGB"))
        assert before_after.shape == (720, 2560, 3)
        changed = np.mean(abs(before_after[:, :1280].astype("f4")-
                              before_after[:, 1280:].astype("f4")), axis=-1) > 7
        assert 650 < changed.sum() and summary["removed_road_gaussians"] > 0
        site = summary["site"]
        preserved = None
        if summary["full_editable_checkpoint"]:
            if site not in baseline:
                label = "A" if site == "056" else "B"
                baseline[site] = torch.load(project / f"outputs/scene_{label}_{site}/checkpoint_7000.pth",
                                            map_location="cpu", weights_only=False)
            check = torch.load(folder / "trained_model/iteration_7000.pth", map_location="cpu",
                               weights_only=False)
            old_components = set(baseline[site]) - {"background", "iter"}
            assert set(check)-{"background", "iter"} == old_components
            assert all(exact(check[key], baseline[site][key]) for key in old_components)
            assert (folder / "scene_frame040.ply").stat().st_size > 1_000_000
            preserved = True
        reports.append({"id": summary["id"], "site": site, "drive_frames": 80,
                        "near_frames": 45, "changed_pixels_near": int(changed.sum()),
                        "editable_scene": summary["full_editable_checkpoint"],
                        "actors_bit_exact_in_checkpoint": preserved})
    counts = {site: sum(r["site"] == site for r in reports) for site in ("055", "056")}
    assert counts == {"055": 11, "056": 5}
    (root / "scene_trial_validation.json").write_text(json.dumps(reports, indent=2)+"\n")
    concat = work / "scene_concat.txt"
    names = [root / f"scene_trials_{site}.mp4" for site in ("055", "056")]
    concat.write_text("".join(f"file '{path}'\n" for path in names))
    movie = root / "PandaSet_16种插入演示.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(movie)],
                   check=True)
    frames(movie, 16*75, 20)
    print("validated", len(reports), "PandaSet scenes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.work.resolve())
