"""Compare both trained models across the full rig using grouped holdout timestamps."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare import metrics, read

WIDTH, HEIGHT = 1920, 962
NAMES = {
    "image_00": "前左 · 针孔",
    "image_01": "前右 · 针孔",
    "image_02": "左侧 · 鱼眼",
    "image_03": "右侧 · 鱼眼",
    "front_left": "左前 · 鱼眼",
    "front_right": "右前 · 鱼眼",
    "front_center": "正前 · 长焦",
    "side_left": "左侧 · 鱼眼",
    "side_right": "右侧 · 鱼眼",
    "rear_center": "正后 · 鱼眼",
    "front_camera": "正前",
    "front_left_camera": "左前",
    "front_right_camera": "右前",
    "left_camera": "左侧",
    "right_camera": "右侧",
    "back_camera": "正后",
}
ORDER = ["front_center", "front_left", "front_right", "side_left", "side_right", "rear_center"]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def average(rows):
    return {key: float(np.mean([m[key] for m in rows])) for key in ("psnr_db", "ssim")}


def compare(root, scene, steps):
    bundle = root / "work/streetcrafter-rig" / scene
    meta = json.loads((bundle / "all/split.json").read_text())
    fish = (bundle / "sensor_eval").is_dir()
    source = bundle / ("sensor_eval" if fish else "eval") / "images"
    run = root / "runs/streetcrafter-rig" / scene
    methods = ["3dgut", "sc-gs"]
    inputs = [run / f"{m}-{steps}" / ("sensor_frames" if fish else "frames") for m in methods]
    output = root / "outputs/streetcrafter-rig" / scene / str(steps)
    output.mkdir(parents=True, exist_ok=True)
    sensors = ORDER if scene.startswith("a2d2") else meta["sensors"]
    label = {
        "055": "PandaSet 055",
        "056": "PandaSet 056",
        "kitti360-0003-right": "KITTI-360 0003",
        "kitti360-0000": "KITTI-360 0000",
        "a2d2-frontleft": "Audi A2D2 20180810",
    }[scene]
    ground = {v["group"]: {} for v in meta["views"].values()}
    for name, v in meta["views"].items():
        if v["tile"] == 0:
            ground[v["group"]][v["sensor"]] = name
    report = {
        "scene": scene,
        "steps": int(steps),
        "sensors": sensors,
        "seed_points": meta["seed_points"],
        "physical_images_train": sum(n in meta["train"] for group in ground.values() for n in group.values()),
        "physical_images_test": len(meta["test"]),
        "scgs_training_views": len(meta["train"]),
        "projection": {
            s: (
                "calibrated_fisheye"
                if meta["projections"]["gut_train"][str(meta["sensors"].index(s) * 10 + 1)][0]
                == "OPENCV_FISHEYE"
                else "pinhole"
            )
            for s in sensors
        },
        "held_out": [],
        "per_camera": {},
        "max_group_skew_ms": 0.0,
    }
    big = ImageFont.truetype(FONT, 34, index=2)
    small = ImageFont.truetype(FONT, 24, index=2)
    t0 = np.mean([meta["times"][n] for n in ground[0].values()])
    column = (WIDTH - 130) // len(sensors)
    with TemporaryDirectory(dir=output) as tmp:
        video = Path(tmp) / "all-cameras.mp4"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "3",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            for i, frame in enumerate(meta["frames"]):
                names = [ground[i][s] for s in sensors]
                times = [meta["times"][n] for n in names]
                skew = (max(times) - min(times)) * 1000
                report["max_group_skew_ms"] = max(report["max_group_skew_ms"], skew)
                test = names[0] in meta["test"]
                if not all((n in meta["test"]) == test for n in names):
                    raise ValueError("Split crossed camera boundaries")
                canvas = Image.new("RGB", (WIDTH, HEIGHT), "#12191c")
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 12), label, fill="white", font=big)
                info = (
                    f"{len(sensors)} 路联合重建    |    参考帧 {frame}    |    {np.mean(times) - t0:.2f} 秒    |    "
                    + ("留出时刻" if test else "训练时刻")
                )
                draw.text(
                    (WIDTH - 24, 40), info, anchor="rm", fill="#8cdec5" if test else "#c5cdd1", font=small
                )
                for row, title in enumerate(["实拍", "3DGUT", "SC-GS"]):
                    draw.text((20, 246 + 280 * row), title, fill="white", font=small, anchor="lm")
                for j, (sensor, name) in enumerate(zip(sensors, names)):
                    x = 128 + column * j
                    draw.text((x + column // 2, 89), NAMES[sensor], anchor="mm", font=small, fill="#dce6e8")
                    ref = cv2.imread(str(source / name))
                    if ref is None:
                        raise FileNotFoundError(source / name)
                    imgs = [ref, *[read(folder / name, ref.shape[1::-1]) for folder in inputs]]
                    if test:
                        report["held_out"].append(
                            dict(
                                group=i,
                                source_frame=meta["views"][name]["source_frame"],
                                sensor=sensor,
                                time=times[j],
                                **{method: metrics(ref, img) for method, img in zip(methods, imgs[1:])},
                            )
                        )
                    for row, img in enumerate(imgs):
                        tile = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        tile.thumbnail((column - 8, 272), Image.Resampling.LANCZOS)
                        canvas.paste(
                            tile, (x + (column - tile.width) // 2, 112 + row * 280 + (272 - tile.height) // 2)
                        )
                proc.stdin.write(canvas.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Video encoding failed")
        video.replace(output / video.name)
    report["per_camera"] = {
        s: {m: average([r[m] for r in report["held_out"] if r["sensor"] == s]) for m in methods}
        for s in sensors
    }
    report["mean"] = {m: average([r[m] for r in report["held_out"]]) for m in methods}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(scene, report["mean"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("scene")
    p.add_argument("steps", type=int)
    compare(**vars(p.parse_args()))
