"""Check both visible instances against independent casts at the calibrated camera subpixels."""

import argparse
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from build import Surface, KEY
from corridor import drv, read, digest, write


def check(raw, models, corridor, out):
    manifest = read(corridor / "report.json")
    path = models / "raised/model/series/assets/raised" / KEY / "asset.glb"
    mesh = Surface(path)
    assert digest(path) == manifest["mesh_sha256"]
    result = {}
    for side, r in manifest["cameras"].items():
        cam = raw / "camera" / (side + "_camera")
        frame = r["representative"]["frame"]
        pose = drv.drive.pose_matrix(read(cam / "poses.json")[frame])
        model = drv.pinhole(cam, read(cam / "intrinsics.json"), frame)
        d = corridor / side
        packet = np.load(d / "both.npz")
        x, y = packet["crop_xy"]
        ids = packet["instance"]
        parts = packet["part"]
        dist = packet["range_m"]
        h, w = ids.shape
        report = []
        assert imageio_ffmpeg.count_frames_and_secs(str(d / "drive.mp4"))[0] == len(r["frames"])
        for idx, anchor in enumerate(manifest["anchors"], 1):
            yy, xx = np.where((ids == idx) & (parts > 1))
            minimum = np.full(len(yy), np.inf)
            R = np.array(anchor["basis_columns"])
            centre = np.array(anchor["center_world"])
            for dx, dy in drv.drive.renderer.SUBPIXELS:
                dirs = model[2](x, y, w, h, dx, dy)[0][yy, xx] @ (pose[:3, :3].T @ R)
                eye = (pose[:3, 3] - centre) @ R
                t, l, _ = mesh.cast(np.broadcast_to(eye, dirs.shape), dirs)
                minimum = np.minimum(
                    minimum, np.where(l == parts[yy, xx], abs(t - dist[yy, xx]), np.inf)
                )
            assert len(yy) == r["representative"]["visible_defect_px"][idx - 1]
            assert minimum.max() < 0.0003
            report.append(dict(tested=len(yy), max_error_m=float(minimum.max())))
        result[side] = dict(
            frames=len(r["frames"]), joint_frames=r["joint_frames"], first_hits=report
        )
    write(out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("raw", "models", "corridor", "out"):
        p.add_argument(k, type=Path)
    check(**vars(p.parse_args()))
