"""Assemble the five completed rig replays and their per-camera calibration reports."""

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np
from scipy.spatial.transform import Rotation

from rig_video import NAMES

SCENES = {
    "a2d2-frontleft": "Audi A2D2 · 20180810_150607",
    "kitti360-0003-right": "KITTI-360 · 0003",
    "kitti360-0000": "KITTI-360 · 0000",
    "055": "PandaSet · 055",
    "056": "PandaSet · 056",
}


def gallery(root, steps):
    dest = root / "outputs/streetcrafter-rig"
    sections, inputs = [], []

    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(path, dest), quote=True)}">{html.escape(label)}</a>'

    for scene, title in SCENES.items():
        folder = dest / scene / str(steps)
        bundle = root / "work/streetcrafter-rig" / scene
        run = root / "runs/streetcrafter-rig" / scene
        sc_model = run / f"sc-gs-{steps}" / "fit_node"
        m = json.loads((bundle / "all/split.json").read_text())
        r = json.loads((folder / "metrics.json").read_text())
        poses = {}
        for line in (bundle / "eval/sparse/0/images.txt").read_text().splitlines():
            if line:
                _, qw, qx, qy, qz, tx, ty, tz, cid, name = line.split()
                rotation = Rotation.from_quat(np.array([qx, qy, qz, qw], float)).as_matrix()
                poses.setdefault(cid, -rotation.T @ np.array([tx, ty, tz], float))
        rows = []
        for sensor in r["sensors"]:
            i = m["sensors"].index(sensor)
            cid = str(10 * i + 1)
            c = m["source_calibration"][sensor]
            if "K" in c:
                k = np.array(c["K"])
                intr = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
                dist = ", ".join(f"{v:.6g}" for v in c.get("D", [])) or "—"
            else:
                spec = c["parameters"]
                intr = [spec["projection_parameters"][k] for k in ("gamma1", "gamma2", "u0", "v0")]
                dist = (
                    "ξ = "
                    + str(round(spec["mirror_parameters"]["xi"], 7))
                    + "<br>"
                    + ", ".join(f"{k}={v:.5g}" for k, v in spec["distortion_parameters"].items())
                )
            ext = ", ".join(f"{v:.3f}" for v in poses[cid])
            gut, sc = [r["per_camera"][sensor][n] for n in ("3dgut", "sc-gs")]
            source_ids = [
                v["source_frame"] for v in m["views"].values() if v["sensor"] == sensor and v["tile"] == 0
            ]
            rows.append(
                f"<tr><th>{NAMES[sensor]}<small>{sensor} · {min(source_ids)}–{max(source_ids)}</small></th>"
                f"<td>{c['model']}<small>{int(c['size'][0])} × {int(c['size'][1])}</small></td>"
                f"<td>{', '.join(f'{x:.2f}' for x in intr)}</td><td>{dist}</td><td>{ext}</td>"
                f"<td>{gut['psnr_db']:.2f} / {gut['ssim']:.3f}</td><td>{sc['psnr_db']:.2f} / {sc['ssim']:.3f}</td></tr>"
            )
        video = folder / "all-cameras.mp4"
        inputs.append(video)
        count = len(r["sensors"])
        sections.append(
            f"<section><h2>{title}</h2><p>{count} 台相机 · {len(m['frames'])} 个采集组 · "
            f"{r['physical_images_train']} 张训练原图 · {r['physical_images_test']} 张留出原图 · "
            f"{m['seed_points']:,} 个初始化点 · 组内最大曝光时差 {r['max_group_skew_ms']:.2f} ms</p>"
            f'<video controls preload="metadata" src="{video.relative_to(dest)}"></video>'
            '<div class="scroll"><table><thead><tr><th>相机与源帧号</th><th>相机模型／源尺寸</th>'
            "<th>源图内参 fx, fy, cx, cy（像素）</th><th>畸变参数</th><th>首帧光心 X, Y, Z（米）</th>"
            "<th>3DGUT<br>PSNR / SSIM</th><th>SC-GS<br>PSNR / SSIM</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div><p>"
            + " · ".join(
                [
                    link(bundle / "all/split.json", "完整标定、曝光时间与划分"),
                    link(bundle / "eval/sparse/0/images.txt", "逐图外参"),
                    link(bundle / "gut_train/sparse/0/cameras.txt", "3DGUT 训练内参"),
                    link(bundle / "all/sparse/0/cameras.txt", "SC-GS 切平面内参"),
                    link(folder / "metrics.json", "逐相机与逐留出图指标"),
                    link(run / f"3dgut-{steps}/scene.ply", "3DGUT 高斯点云"),
                    link(sc_model / f"point_cloud/iteration_{steps}/point_cloud.ply", "SC-GS 高斯点云"),
                    link(sc_model / f"deform/iteration_{steps}/deform.pth", "SC-GS 动态形变权重"),
                ]
            )
            + "</p></section>"
        )
    with TemporaryDirectory(dir=dest) as tmp:
        stage = Path(tmp)
        sources = stage / "concat.txt"
        sources.write_text("".join(f"file '../{p.relative_to(dest)}'\n" for p in inputs))
        video = stage / "all-scenes.mp4"
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
                str(sources),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
        video.replace(dest / video.name)
    head = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>StreetCrafter 全相机重建</title>
<style>body{font:16px/1.55 Inter,system-ui,sans-serif;background:#101719;color:#e6ebee;max-width:1480px;margin:auto;padding:3vw}h1{font-size:clamp(28px,4vw,52px)}h2{margin-top:48px}p,small{color:#b7c2c9}small{display:block}video{width:100%;background:#12191c}a{color:#86dccc}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #394144}thead{background:#203034}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto;margin-top:16px}</style><main><h1>StreetCrafter 全相机重建</h1>"""
    intro = f"<p>每个场景的全部物理相机共同拟合一个 3DGUT 静态模型及一个 SC-GS 动态模型，各 {steps:,} 步。视频每列为一台相机，三行依次为实拍、3DGUT、SC-GS。PSNR 单位为 dB，SSIM 无量纲，均在留出组上计算。</p>"
    detail = "<p>A2D2 五路鱼眼保留完整画幅，正前长焦按标定校正。KITTI-360 两路侧摄按 MEI 重采样为水平 125° 鱼眼。前方双目的内参对应数据集发布的校正图像。PandaSet 保留各路曝光时间和相机位姿。表中光心采用各场景自己的米制坐标系。视频每秒播放 10 个采集组。</p>"
    (dest / "index.html").write_text(
        head
        + intro
        + detail
        + link(dest / "all-scenes.mp4", "五场景视频合辑")
        + "".join(sections)
        + "</main></html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("steps", type=int, default=3000, nargs="?")
    gallery(**vars(parser.parse_args()))
