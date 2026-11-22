"""Show a photographed angular lid, its true openings and four matched driving-camera states."""

import argparse
from importlib import import_module
import json
from pathlib import Path

from PIL import Image, ImageOps
from road_camera_frames import full_driving_frames

Plate = import_module("render-pit-survey").Plate
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def render(run, audit, traffic, output):
    bundle = run / "series"
    survey = read(bundle / "calibration/survey.json")
    states = survey["states"]
    sites = {}
    for scene in sorted((bundle / "scenes").glob("*/scene.json")):
        info = read(scene)
        (key,) = info["assets"]
        sites.setdefault(info["site"], {})[key] = info["id"]
    raw = Path(read(ROOT / "gen2hole/config/local.json")["pandaset_raw"])
    frames = full_driving_frames(run, (item for cases in sites.values() for item in cases.items()), raw)
    plate = Plate(1640, 1818)
    rear = [
        f
        for views in frames.values()
        for f in views
        if f["site"] == "034" and f["camera"] == "back_camera"
    ]
    box = (
        min(f["bbox"][0] - f["roi"][0] for f in rear) - 4,
        min(f["bbox"][1] - f["roi"][1] for f in rear) - 4,
        max(f["bbox"][2] - f["roi"][0] for f in rear) + 4,
        max(f["bbox"][3] - f["roi"][1] for f in rear) + 4,
    )
    plate.text(30, 24, "六边形井盖的落座、错转、掀起与缺失", 48, True)
    plate.text(32, 100, "实拍校正的同一块盖板与固定井体　｜　两段 PandaSet 道路、相同机位与原始尺寸", 24)
    evaluation = run / "evaluation"
    plate.picture(evaluation / "source-landmarks.png", 30, 155, 454, 321)
    plate.text(31, 492, "来源照片　青色拟合点 · 黄色留出点 · 红色边缘凹口", 17)
    plate.picture(evaluation / "orthophoto.png", 504, 155, 321, 321)
    plate.text(505, 492, "去除平面透视后的盖面纹理", 17)
    sample = next(
        f for f in frames[audit["missing"]] if f["camera"] == "front_camera" and f["site"] == "109"
    )
    plate.picture(run / sample["full_after"], 847, 155, 762, 321)
    plate.text(847, 492, f"缺盖迁入后　PandaSet 109 · 原车前视第 {sample['frame']} 帧", 17)
    width, pitch = 382, 401
    for i, state in enumerate(states):
        x, key = 30 + i * pitch, state["id"]
        plate.text(x, 550, key + "   " + state["label"], 23, True)
        plate.picture(bundle / "views" / (key + "_reverse.jpg"), x, 596, width, 210)
        # Restrict the physical plan to the same central 0.80 m square for all four states.
        with Image.open(evaluation / key / "plan.png") as plan:
            view = ImageOps.contain(plan.crop((56, 56, 457, 457)), (width, 174))
            plate.draw.rectangle((x, 818, x + width - 1, 991), fill="#e3e2d7")
            plate.image.paste(view, (x + (width - view.width) // 2, 818))
        percent = 100 * audit["states"][key]["visible_floor_fraction"]
        number = f"{percent:.3f}" if 0 < percent < 0.01 else f"{percent:.2f}".rstrip("0").rstrip(".")
        plate.text(x, 1009, "从井口正上方可见井底　" + number + "%", 18)
        for top, (site, camera, label) in zip(
            (1063, 1282), (("109", "front_camera", "前视"), ("034", "back_camera", "后视"))
        ):
            shot = next(f for f in frames[key] if f["site"] == site and f["camera"] == camera)
            plate.picture(run / shot["after"], x, top, width, 169)
            plate.text(x, top + 180, f"PandaSet {site} · {label}第 {shot['frame']} 帧", 17)
        part = traffic["visible_parts"][key]["034-back_camera"]
        with Image.open(run / part["image"]) as cropped:
            detail = ImageOps.contain(cropped.crop(box), (width, 147), Image.Resampling.NEAREST)
            plate.image.paste(
                detail, (x + (width - detail.width) // 2, 1504 + (147 - detail.height) // 2)
            )
        counts = part["pixels"]
        plate.text(x, 1664, f"盖背 {counts['10']:,}　井壁 {counts['6']:,}　井底 {counts['7']:,} 像素", 17)
    plate.text(
        30,
        1714,
        "末行为同一后视画面的等比例局部。黄为盖面，粉为盖背，青为井壁。缺盖后此机位可见井框和井壁。",
        18,
    )
    plate.text(
        30,
        1750,
        "俯视图取自同尺度 GLB 射线。金色为盖面，深蓝为井底。参考跨度 0.68 米、盖厚 38 毫米、井深 0.72 米及异常姿态采用设计值。",
        18,
    )
    plate.text(
        30,
        1787,
        "原照 Stephen Craven / Geograph 3739078，CC BY-SA 2.0。盖面校正、轮廓与模型改编沿用原许可。道路底图来自 PandaSet。",
        17,
    )
    plate.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render(args.run, read(args.audit), read(args.transfer), args.out)
