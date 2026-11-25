"""Make and verify stereoscopic, full-longitude and seam-crossing camera inputs."""

from importlib import import_module

import numpy as np
from PIL import Image

files = import_module("check-mei-pits")
cli = import_module("check-cone-cycles").cli
from drain_stereo_inputs import pack_view, pair_maps
from drain_stereo_checks import register, load


def project(xyz, meta, top):
    pose = np.array(meta["camera_to_world"])
    q = (xyz - pose[:3, 3]) @ pose[:3, :3]
    _, w, h = meta["projection"]
    return np.stack(
        [
            w * (0.5 + np.arctan2(q[..., 0], q[..., 2]) / (2 * np.pi)) - 0.5,
            h * (0.5 + np.arctan2(q[..., 1], np.hypot(q[..., 0], q[..., 2])) / np.pi)
            - 0.5
            - top,
        ],
        -1,
    )


def prepare(root, dest, pairs):
    for pair in pairs:
        for shot in pair.lower().split("-"):
            src = root / pair / shot
            d = dest / pair / shot
            d.mkdir(parents=True, exist_ok=True)
            hosts = [src / name / "host" for name in ("primary", "secondary")]
            metas = [files.read(p / "camera.json") for p in hosts]
            _, w, h = metas[0]["projection"]
            top = h // 2 - 42
            ids = [np.load(p / "id.npy")[top:] for p in hosts]
            for i, name in enumerate(("primary", "secondary")):
                pose = metas[i]["camera_to_world"]
                rgb = np.array(Image.open(hosts[i] / "rgb.png").convert("RGB"))[top:]
                pack_view(
                    d,
                    name,
                    pose,
                    rgb,
                    metas[i]["projection"],
                    [0, top],
                    np.zeros_like(ids[i], bool),
                    empty_space_mask=ids[i] == 0,
                    **(
                        {"receiver_mask": ids[i] == 1, "occluder_mask": ids[i] == 2}
                        if i == 0
                        else {}
                    ),
                )
            maps = [
                project(np.load(p / "position.npy")[top:], metas[1 - i], top)
                for i, p in enumerate(hosts)
            ]
            maps[1][ids[1] == 0] = np.nan
            pair_maps(d, *maps, ids[0] > 0)
            native = src / "primary/native"
            np.savez_compressed(
                d / "truth.npz",
                host_id=ids[0],
                host_range=np.load(hosts[0] / "range.npy")[top:],
                native_id=np.load(native / "id.npy")[top:],
                native_range=np.load(native / "range.npy")[top:],
            )
            files.save(d / "ground.json", metas[0]["reference_ground_to_world"])


def cropped_replay(model, anchor, src, host, dest):
    # Join the right and left strips in their original detector order. No raster reprojection.
    packet = dict(host)
    del packet["rays_camera"]
    w = packet["rgb"].shape[1]
    half = w // 5
    cols = np.arange(w - half, w + half) % w
    packet["rgb"] = packet["rgb"][:, cols]
    for field in ("protected", "host_range", "occluder_range", "receiver_mask"):
        packet[field] = packet[field][:, :, cols]
    original = load(src / "primary.npz")
    _, top, _, h = original["sensor_window_xywh"]
    packet["sensor_window_xywh"] = [w - half, top, 2 * half, h]
    target = dest / "wrapped.npz"
    np.savez_compressed(target, **packet)
    cli(
        "render",
        model,
        anchor,
        target,
        dest / "wrapped",
        "--calibration",
        src / "primary.json",
    )
    for name in ("after.png", "part.png", "instance.png", "range.npy", "shadow.npy"):

        def read(p):
            return np.load(p) if name.endswith(".npy") else np.array(Image.open(p))

        a = read(dest / "full" / name)[:, cols]
        b = read(dest / "wrapped" / name)
        # Ray differentials at the outside crop boundaries have different neighbours.
        if name.endswith(".npy"):
            tol = 1e-5 if name == "range.npy" else 2e-6
            np.testing.assert_allclose(
                a[:, 2:-2], b[:, 2:-2], rtol=0, atol=tol, equal_nan=True, err_msg=name
            )
        else:
            np.testing.assert_array_equal(a[:, 2:-2], b[:, 2:-2], err_msg=name)
    return int((len(cols) - 4) * h)


def inspect(src, dest, model, anchor, focus):
    truth = load(src / "truth.npz")
    host = load(dest / "host.npz")
    valid = ~host["protected"][0]
    native = truth["native_id"]
    path = dest / "full"
    cli("render", model, anchor, dest / "host.npz", path)
    after = np.array(Image.open(path / "after.png"))
    detected = np.array(Image.open(path / "instance.png"))
    radial = np.load(path / "range.npy")
    frozen = ~valid | ((truth["host_id"] == 2) & (native == 2))
    if detected[frozen].any() or not np.array_equal(after[frozen], host["rgb"][frozen]):
        raise ValueError("An unmatched sample or visible foreground marker changed")
    host_mask = valid & (truth["host_id"] > 0) & (truth["host_range"] < 8)
    error = abs(host["host_range"][0][host_mask] - truth["host_range"][host_mask])
    stats = dict(
        host_errors_p95_m=float(np.percentile(error, 95)),
        host_pixels=int(host_mask.sum()),
        registered_marker_pixels=int(((truth["host_id"] == 2) & valid).sum()),
        assets={},
    )
    xy = load(src / "matches.npz")["right_xy"][..., 0]
    seam = host_mask & ((xy < 0) | (xy > host_mask.shape[1] - 1))
    if seam.sum() < 10:
        raise ValueError(
            "Insufficient secondary matches between the last and first columns"
        )
    stats["interpolated_secondary_seam_pixels"] = int(seam.sum())
    stats["secondary_seam_p95_m"] = float(
        np.percentile(abs(host["host_range"][0][seam] - truth["host_range"][seam]), 95)
    )
    if stats["secondary_seam_p95_m"] > 0.012:
        raise ValueError("Periodic stereo range drifted near the secondary seam")
    if stats["host_errors_p95_m"] > 0.012 or stats["registered_marker_pixels"] < 20:
        raise ValueError("Range or foreground marker failed")
    for i, key in enumerate(model.name.split("-"), 1):
        expected = native == 2 + i
        observed = detected == i
        shared = expected & observed & valid
        if not np.any(shared):
            raise ValueError("No matched asset pixels: " + key)
        values = dict(
            native_pixels=int(expected.sum()),
            coverage=float(valid[expected].mean()),
            matched_iou=float(shared.sum() / ((expected | observed) & valid).sum()),
            range_p95_m=float(
                np.percentile(abs(truth["native_range"][shared] - radial[shared]), 95)
            ),
        )
        values["visible_over_confirmed_background"] = int(
            (shared & (truth["host_id"] == 0)).sum()
        )
        values["finite_stereo_coverage"] = float(
            (valid & (truth["host_id"] > 0))[expected].mean()
        )
        if i == focus:
            values["native_left_seam_pixels"] = int(expected[:, :12].sum())
            values["native_right_seam_pixels"] = int(expected[:, -12:].sum())
            if (
                min(
                    values["native_right_seam_pixels"],
                    values["native_left_seam_pixels"],
                )
                < 12
            ):
                raise ValueError(f"An asset must span the seam: {key} {values}")
        if (
            values["coverage"] < 0.93
            or values["matched_iou"] < 0.90
            or values["range_p95_m"] > 0.018
        ):
            raise ValueError(f"Native geometry disagrees: {key} {values}")
        stats["assets"][key] = values
    stats["compared_wrapped_pixels_per_channel"] = cropped_replay(
        model, anchor, src, host, dest
    )
    return stats


def check(source, dest, models, pairs):
    from check_spherical_samples import verify

    verify()
    report = {}
    for pair in pairs:
        model = models / pair
        base = dest / pair
        sources = [source / pair / s for s in pair.lower().split("-")]
        paths = [register(s, base / s.name) for s in sources]
        frame = np.array(files.read(sources[0] / "ground.json"))
        cli("road-points", base / "road.npy", *paths, "--stride", 2)
        data = load(paths[0])
        rgb = data["rgb"][data["receiver_mask"][0]]
        from scipy.spatial.transform import Rotation

        files.save(
            base / "light.json",
            dict(
                road_srgb=np.median(rgb, axis=0).tolist(),
                direct_fraction=0.48,
                sun_world=Rotation.from_euler("xyz", [0.44, -0.63, -0.3])
                .apply([0, 0, 1])
                .tolist(),
            ),
        )
        anchor = base / "anchor.json"
        cli(
            "fit",
            base / "road.npy",
            anchor,
            "--seed",
            *frame[:3, 3],
            "--heading",
            *frame[:3, 0],
            "--up",
            *frame[:3, 2],
            "--radius",
            1.5,
            "--asset",
            model,
            "--lighting",
            base / "light.json",
        )
        for i, s in enumerate(sources, 1):
            report[pair + "/" + s.name] = inspect(s, base / s.name, model, anchor, i)
    files.save(dest / "check.json", report)
