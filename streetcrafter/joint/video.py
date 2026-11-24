"""Encode synchronized rig video and evaluate the joint checkpoint on held-out observations."""

import argparse
import html
import os
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends"))
from compare import metrics, read
from rig_video import FONT, NAMES, ORDER, average

SCENES = {
    "a2d2-frontleft": "Audi A2D2 · 20180810_150607",
    "kitti360-0003-right": "KITTI-360 · 0003",
    "kitti360-0000": "KITTI-360 · 0000",
    "055": "PandaSet · 055",
    "056": "PandaSet · 056",
}


def render_video(root, scene, steps):
    bundle = root / "work/streetcrafter-rig" / scene
    split = json.loads((bundle / "all/split.json").read_text())
    source = bundle / ("sensor_eval" if (bundle / "sensor_eval").exists() else "eval") / "images"
    run = root / "runs/streetgs-joint" / scene / str(steps)
    out = root / "outputs/streetgs-joint" / scene
    out = out / str(steps)
    out.mkdir(exist_ok=True, parents=True)
    sensors = ORDER if scene == "a2d2-frontleft" else split["sensors"]
    cols = len(sensors) // 2
    width, height = 1536, 1024
    group_height = 460
    test = set(split["test"])
    mapping = {(v["group"], v["sensor"]): n for n, v in split["views"].items() if not v["tile"]}
    font = ImageFont.truetype(FONT, 21, index=2)
    title = ImageFont.truetype(FONT, 31, index=2)
    records = []
    with TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / "joint.mp4"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
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
            str(path),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            for i, frame in enumerate(split["frames"]):
                canvas = Image.new("RGB", (width, height), "#151a1e")
                draw = ImageDraw.Draw(canvas)
                draw.text((22, 14), SCENES[scene], font=title, fill="white")
                heldout = mapping[i, sensors[0]] in test
                draw.text(
                    (width - 22, 24),
                    f"{len(sensors)} 路相机  ·  参考帧 {frame}  ·  " + ("留出组" if heldout else "训练组"),
                    anchor="rt",
                    font=font,
                    fill="#93e2bf" if heldout else "#c7d1d5",
                )
                draw.text(
                    (22, 57),
                    "每列上图为实拍，下图为 StreetGS + 3DGUT + SC-GS 联合重建",
                    font=font,
                    fill="#ced6d6",
                )
                for j, sensor in enumerate(sensors):
                    name = mapping[i, sensor]
                    gt = cv2.imread(str(source / name))
                    if gt is None:
                        raise FileNotFoundError(source / name)
                    pred = read(run / "frames" / name, gt.shape[1::-1])
                    if heldout:
                        records.append(dict(name=name, sensor=sensor, group=i, **metrics(gt, pred)))
                    x = (j % cols) * width // cols
                    y = 94 + (j // cols) * group_height
                    draw.text(
                        (x + width // cols // 2, y), NAMES[sensor], font=font, fill="white", anchor="mt"
                    )
                    for row, img in enumerate([gt, pred]):
                        tile = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        factor = min((width // cols - 8) / tile.width, 210 / tile.height)
                        tile = tile.resize(
                            (round(tile.width * factor), round(tile.height * factor)),
                            Image.Resampling.LANCZOS,
                        )
                        canvas.paste(
                            tile,
                            (
                                x + (width // cols - tile.width) // 2,
                                y + 28 + row * 213 + (210 - tile.height) // 2,
                            ),
                        )
                proc.stdin.write(canvas.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("FFmpeg encoding failed")
        path.replace(out / "joint.mp4")
    evidence = json.loads((run / "training.json").read_text())
    effect = json.loads((run / "warp_effect.json").read_text())
    if len(records) != len(test) or any(n not in evidence["samples"] for n in set(mapping.values()) - test):
        raise ValueError("Incomplete camera coverage")
    report = dict(
        scene=scene,
        steps=steps,
        sensors=sensors,
        groups=len(split["frames"]),
        train_images=len(evidence["samples"]),
        test_images=len(records),
        nodes=evidence["nodes"],
        tracked_objects=len(evidence["layers"]) - 1 if scene in ["055", "056"] else 0,
        layering="annotated object trajectories"
        if scene in ["055", "056"]
        else "untracked multi-view residual layer",
        mean=average(records),
        per_camera={s: average([r for r in records if r["sensor"] == s]) for s in sensors},
        held_out=records,
        photo_gradient_norms=evidence["photo_gradient_norms"],
        warp_effect_mean_rgb=float(np.mean([r["mean_rgb_difference"] for r in effect])),
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(scene, report["mean"])


def combine(root, steps):
    dest = root / "outputs/streetgs-joint"
    with TemporaryDirectory(dir=dest) as tmp:
        temp = Path(tmp)
        playlist = temp / "list.txt"
        playlist.write_text("".join(f"file '../{s}/{steps}/joint.mp4'\n" for s in SCENES))
        path = temp / "joint-all-scenes.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
        )
        path.replace(dest / f"joint-all-scenes-{steps}.mp4")
    reports = [json.loads((dest / s / str(steps) / "metrics.json").read_text()) for s in SCENES]
    (dest / f"results-{steps}.json").write_text(json.dumps(reports, indent=2))
    sections = []

    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(path, dest), quote=True)}">{label}</a>'

    for report in reports:
        scene = report["scene"]
        run = root / "runs/streetgs-joint" / scene / str(steps)
        bundle = root / "work/streetcrafter-rig" / scene
        output = dest / scene / str(steps)
        split = json.loads((bundle / "all/split.json").read_text())
        rows = []
        for sensor in report["sensors"]:
            cid = str(split["sensors"].index(sensor) * 10 + 1)
            model, w, h, fx, fy, cx, cy, *dist = split["projections"]["gut_train"][cid]
            metrics = report["per_camera"][sensor]
            rows.append(
                f"<tr><td>{NAMES[sensor]}<small>{sensor}</small></td><td>{model}</td><td>{w} × {h}</td>"
                f"<td>{fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}</td>"
                f"<td>{metrics['psnr_db']:.2f}</td><td>{metrics['ssim']:.3f}</td></tr>"
            )
        proof = report["photo_gradient_norms"]
        effects = json.loads((run / "warp_effect.json").read_text())
        change = np.mean([r["joint_psnr"] - r["warp_off_psnr"] for r in effects])
        layering = (
            f"{report['tracked_objects']} 个带轨迹的物体层"
            if report["tracked_objects"]
            else "一个无实例标注的残余形变层"
        )
        sections.append(
            f"<section><h2>{SCENES[scene]}</h2><p>{report['groups']} 个采集组，"
            f"{report['train_images']} 张训练图、{report['test_images']} 张留出图，{layering}。</p>"
            f'<video controls preload="metadata" src="{scene}/{steps}/joint.mp4"></video>'
            "<div><table><tr><th>物理相机</th><th>训练投影</th><th>训练尺寸</th><th>训练内参 fx, fy, cx, cy（像素）</th><th>PSNR（dB）</th><th>SSIM</th></tr>"
            + "".join(rows)
            + "</table></div>"
            + f"<p>记录到同一张训练图的光度损失梯度范数：背景 {proof['streetgs_background']:.3g}，"
            f"物体层 {proof['streetgs_object']:.3g}，形变网络 {proof['scgs_network']:.3g}。"
            f"在有可见形变层的 {len(effects)} 张留出图上，同一检查点启用形变相对关闭形变的平均 PSNR 差为 {change:+.2f} dB。</p>"
            + " · ".join(
                [
                    link(run / "joint.pt", "联合检查点"),
                    link(output / "metrics.json", "逐图指标"),
                    link(bundle / "all/split.json", "原始标定与时间戳"),
                    link(bundle / "eval/sparse/0/images.txt", "逐曝光外参"),
                    link(run / "training.json", "训练采样与梯度"),
                    link(run / "warp_effect.json", "形变开关检查"),
                ]
            )
            + "</section>"
        )
    header = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>StreetGS 联合重建</title>
<style>body{background:#151a1e;color:#e9edef;margin:auto;max-width:1260px;padding:3vw;font:16px/1.6 Inter,system-ui,sans-serif}video{width:100%;max-height:70vh;background:#111619}a{color:#99dec5}section{margin:3rem 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #364045}small{display:block;color:#bac4cb}div{overflow-x:auto}</style>
<main><h1>StreetGS + 3DGUT + SC-GS 联合重建</h1><p>所有相机共享一个场景图。StreetGS 组成背景、天空和物体层，SC-GS 学习物体层的残余运动，3DGUT 将整组高斯渲染成单张图像。视频里每台相机的上图为实拍，下图为该联合检查点的输出。</p>"""
    (dest / "index.html").write_text(
        header
        + link(dest / f"joint-all-scenes-{steps}.mp4", "五场景合辑")
        + "".join(sections)
        + "</main></html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("scene", choices=[*SCENES, "all"])
    parser.add_argument("steps", type=int)
    a = parser.parse_args()
    if a.scene == "all":
        combine(a.root, a.steps)
    else:
        render_video(a.root, a.scene, a.steps)
