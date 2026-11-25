"""Pack raw sensor calibrations and reproduce two measured pit pairs in three driving sequences."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen2hole/downstream"))
from mei_camera import calibrated_samples

EXAMPLES = ROOT / "gen2hole/examples/mei-pits"
PAIRS = ("P272-P273", "P274-P275")
SITES = {
    "kitti": "kitti360-0000",
    "avenue": "kitti360-0003-right",
    "audi": "a2d2-frontleft",
}
PRODUCTS = ("after.png", "part.png", "instance.png", "range.npy")


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(out, views):
    from importlib import import_module

    subpixels = import_module("render-mixed-road").SUBPIXELS

    for pair in PAIRS:
        for site, scene in SITES.items():
            src = views / pair / site
            dest = out / pair / site
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / "anchor.json", dest / "appearance.json")
            rig = read(ROOT / "work/streetcrafter-rig" / scene / "all/split.json")[
                "source_calibration"
            ]
            for sensor, details in read(src / "index.json")[pair]["views"].items():
                raw = src / pair / sensor
                folder = dest / sensor
                folder.mkdir(exist_ok=True)
                spec = rig[sensor]
                if spec["model"] == "MEI":
                    road = "kitti360-0000-0200" if site == "kitti" else scene
                    calibration = folder / "calibration.yaml"
                    shutil.copyfile(
                        ROOT / "work/data" / road / "calibration" / (sensor + ".yaml"),
                        calibration,
                    )
                else:
                    k = np.array(spec["K"])
                    fisheye = spec["model"] == "Fisheye"
                    calibration = folder / "calibration.json"
                    params = [
                        "OPENCV_FISHEYE" if fisheye else "PINHOLE",
                        *spec["size"],
                        k[0, 0],
                        k[1, 1],
                        k[0, 2] + 0.5,
                        k[1, 2] + 0.5,
                    ]
                    if fisheye:
                        params.extend(np.ravel(spec["D"]).tolist())
                    save(calibration, params)
                with np.load(raw / "view.npz", allow_pickle=False) as data:
                    packet = dict(data)
                old = packet.pop("rays_camera")
                h, w = packet["rgb"].shape[:2]
                packet.update(
                    sensor_window_xywh=np.array([*details["crop_xy"], w, h]),
                    sample_offsets_px=np.array(subpixels),
                )
                rays, valid = calibrated_samples(calibration, packet)
                np.testing.assert_array_equal(rays, old)
                if np.any(~valid & ~packet["protected"]):
                    raise ValueError(
                        "A contributing sample lies outside the calibrated lens"
                    )
                np.savez_compressed(folder / "view.npz", **packet)
                assets = details["best_frame_instance_pixels"]
                if len(assets) != 2 or min(assets.values()) < 12:
                    raise ValueError(
                        "Expected two visible assets in the registered exposure"
                    )
                save(
                    folder / "record.json",
                    {
                        "frame": details["frame"],
                        "calibration": calibration.name,
                        "frame_count_both": len(
                            details["frames_showing_all_instances"]
                        ),
                        "unclipped": details["unclipped_preview"],
                        "pixels": assets,
                        "original_packet_bytes": (raw / "view.npz").stat().st_size,
                        "compact_packet_bytes": (folder / "view.npz").stat().st_size,
                        "sha256": {key: digest(raw / key) for key in PRODUCTS},
                    },
                )


def replay(inputs, models, out):
    report = {}
    for record in sorted(inputs.glob("*/*/*/record.json")):
        src = record.parent
        pair, site, sensor = src.relative_to(inputs).parts
        target = out / pair / site / sensor
        info = read(record)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/transfer-mesh-view.py"),
                "render",
                str(EXAMPLES / (pair + ".json")),
                str(src.parent / "appearance.json"),
                str(src / "view.npz"),
                str(target),
                "--calibration",
                str(src / info["calibration"]),
                "--asset-root",
                str(models),
            ],
            check=True,
        )
        actual = {key: digest(target / key) for key in PRODUCTS}
        if actual != info["sha256"]:
            raise ValueError(
                "Compact-calibration replay differs from exported dense rays: "
                + str(src)
            )
        report[str(src.relative_to(inputs))] = actual
    if len(report) != 12:
        raise ValueError("Expected twelve physical sensor exposures")
    save(out / "check.json", report)


def check_camera_inputs(path):
    from host_depth import geometry
    from host_position import register_positions

    spec = {
        "sensor_window_xywh": np.array([680, 680, 8, 4]),
        "sample_offsets_px": np.array([[-0.25, 0.25], [0.25, -0.25]]),
        "rgb": np.zeros((4, 8, 3), "u1"),
        "protected": np.zeros((4, 8), bool),
        "camera_to_world": np.eye(4),
    }
    rays, valid = calibrated_samples(path, spec)
    if not valid.all():
        raise ValueError("Expected a visible near-axis test window")
    # Use off-centre ray origins and a row-dependent translation in a large world frame.
    origins = np.zeros_like(rays)
    origins[..., 0] = 0.028
    rows = np.repeat(np.eye(4)[None], 4, axis=0)
    rows[:, 0, 3] = np.arange(4) * 0.012
    delta = np.array([2413100.0, 5361200.0, 81.0])
    local = rows[None, :, None, :3, 3] + origins + rays.astype("f8") * 3.2
    rows[:, :3, 3] += delta
    transform = np.eye(4)
    transform[:3, 3] = delta
    spec.update(
        world_from_position=transform,
        surface_position=local,
        ray_origins_camera=origins,
        camera_to_world_rows=rows,
        position_valid=valid,
    )
    spec["camera_to_world"][:3, 3] = delta
    packet = register_positions(spec, calibration=path)
    _, radial, protected, _, _ = geometry(packet, with_origins=True)
    np.testing.assert_allclose(radial, 3.2, atol=5e-7, rtol=0)
    if protected.any():
        raise ValueError("A known test hit was discarded")
    for bad in (
        {"sensor_window_xywh": [-1, 680, 8, 4]},
        {"sensor_window_xywh": [680, 680, 9, 4]},
        {"sample_offsets_px": [[float("nan"), 0]]},
        {"sample_offsets_px": np.array([[-128, 0]], "i1")},
    ):
        try:
            calibrated_samples(path, dict(spec, **bad))
        except ValueError:
            continue
        raise ValueError("Accepted an invalid raw sensor window or sampling offset")


def optics(inputs, controls, models, out):
    check_camera_inputs(inputs / PAIRS[0] / "kitti/image_03/calibration.yaml")
    report = {}
    import tempfile

    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        for ref in sorted(controls.glob("*/*/raster.npz")):
            pair, site = ref.parent.relative_to(controls).parts
            src = inputs / pair / site / "image_03"
            target = Path(tmp) / pair / site
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/transfer-mesh-view.py"),
                    "render",
                    str(EXAMPLES / (pair + ".json")),
                    str(src.parent / "appearance.json"),
                    str(src / "view.npz"),
                    str(target),
                    "--calibration",
                    str(src / "calibration.yaml"),
                    "--asset-root",
                    str(models),
                ],
                check=True,
            )
            actual = np.asarray(Image.open(target / "instance.png"))
            distance = np.load(target / "range.npy")
            with np.load(ref) as reference:
                expected, radial = reference["instance"], reference["range"]
            instances = []
            for i in (1, 2):
                a, b = actual == i, expected == i
                common = a & b
                delta = abs(distance[common] - radial[common])
                iou = np.count_nonzero(common) / np.count_nonzero(a | b)
                p95 = float(np.percentile(delta, 95))
                if common.sum() < 20 or iou < 0.96 or p95 > 0.008:
                    raise ValueError(
                        f"{pair} {site} instance {i}: {iou=}, {p95=}, {common.sum()=}"
                    )
                instances.append(
                    {
                        "common_pixels": int(common.sum()),
                        "instance_iou": iou,
                        "depth_p95_m": p95,
                    }
                )
            report[pair + "/" + site] = instances
    if len(report) != 4:
        raise ValueError(
            "Expected native forward projections of four real fisheye exposures"
        )
    save(out / "check.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pack", "replay", "optics"])
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "runs/mei-pits")
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/mei-pits")
    parser.add_argument("--controls", type=Path, default=ROOT / "work/mei-controls")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.stage == "pack":
        pack(args.packets, args.views)
    else:
        if args.out is None:
            parser.error("replay and optics require --out")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.stage == "optics":
            optics(args.packets, args.controls, args.assets, args.out)
        else:
            replay(args.packets, args.assets, args.out)
