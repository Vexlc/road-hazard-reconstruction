"""Publish a photographed perforated cover, its four exported states and a matched road frame."""

import argparse
import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from road_camera_frames import camera_frames

module = importlib.util.spec_from_file_location(
    "family", Path(__file__).with_name("render-chamber-family.py")
)
family = importlib.util.module_from_spec(module)
module.loader.exec_module(family)


def render(run, report, site, out):
    bundle = run / "series"
    survey = family.read(bundle / "calibration/survey.json")
    states = survey["states"]
    if len(states) != 4:
        raise ValueError("This plate compares four states")
    audit = family.read(report)
    stock = states[0]["id"]
    is_rect = survey["perforations"]["shape"] == "rect"
    evidence = run / "evaluation"
    shared, match = family.shared_frame(bundle, site, [s["id"] for s in states])
    processed = Path(family.read(family.ROOT / 'gen2hole/config/local.json')['pandaset_processed'])
    streets = camera_frames(run, processed)
    pairs = {}
    for case, record in streets.items():
        (key,) = family.read(bundle / 'scenes' / case / 'scene.json')['assets']
        pairs.setdefault(key, []).append((case, record))
    road_key, road_pair = max(pairs.items(), key=lambda item: len({r['site'] for _, r in item[1]}))
    page = Image.new("RGB", (1520, 1568), "#f3f2ed")
    draw = ImageDraw.Draw(page)
    ink = "#172d2a"
    muted = "#48605a"

    def words(x, y, value, size, bold=False, color=ink):
        font = ImageFont.truetype(family.BOLD if bold else family.FONT, size, index=2)
        draw.text((x, y), value, font=font, fill=color, anchor="lt")

    def panel(path, box, crop=None, bg="#dfdfd6"):
        x, y, w, h = box
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=bg)
        with Image.open(path) as src:
            im = src.convert("RGB")
            if crop:
                im = im.crop(crop)
            thumb = ImageOps.contain(im, (w, h), Image.Resampling.LANCZOS)
            page.paste(thumb, (x + (w - thumb.width) // 2, y + (h - thumb.height) // 2))

    words(32, 22, "有断条的雨水篦" if is_rect else "嵌在铺石中的穿孔圆盖", 46, True)
    side = audit.get("same_photo_side_check")
    sub = (
        f"原照中取出的完整格栅有 60 个贯穿孔。按 0.45 米参考宽度换算，盖长 {side['length_m']:.3f} 米，厚 {side['thickness_m'] * 1000:.1f} 毫米。"
        if is_rect
        else "同一份摄影测量彩色图与位移图提供盖面、铺石和 16 个孔缘。按来源图幅换算，盖板直径 0.655 米。"
    )
    words(33, 85, sub, 23, color=muted)
    source = bundle / "references/frame.png"
    columns = [32, 401, 770, 1139]
    if is_rect:
        inputs = [
            ("取出的格栅原照", source, (0, 57, 644, 437)),
            ("原机位投影 · 蓝绿为侧壁", evidence / stock / "source-pose.jpg", None),
            ("下缘复核 · 黄点为留出标记", evidence / stock / "side-fit.jpg", None),
            ("断口放大 · 左完整，右断条", evidence / states[1]["id"] / "broken-web.png", None),
        ]
    else:
        inputs = [
            ("落座原貌", source, (160, 130, 908, 862)),
            ("孔缘回投 · 青线为网格", evidence / stock / "openings.jpg", None),
            ("垂直向下看井口", evidence / stock / "rays-0.png", None),
            ("偏离垂直方向 55°", evidence / stock / "rays-55.png", None),
        ]
    for x, (title, path, crop) in zip(columns, inputs):
        words(x, 142, title, 20, True)
        panel(path, (x, 177, 348, 188), crop)
    description = (
        f"同图留出的 6 个边缘点距模型下缘中位数 {side['median_px']:.2f} 像素。断条使两个孔连通，其他格条保留。"
        if is_rect
        else "射线颜色显示首先碰到的表面。金色为盖面，棕色为孔壁，蓝绿色为井壁，深蓝为井底。"
    )
    words(32, 387, description, 22, color=muted)
    draw.line((32, 431, 1487, 431), fill="#bec7bb", width=2)
    for x, state in zip(columns, states):
        key = state["id"]
        words(x, 456, key + "   " + state["label"], 25, True)
        panel(bundle / "views" / (key + "_inspection.jpg"), (x, 501, 348, 233), bg="#e7e7df")
    words(32, 766, f"PandaSet {site}    同一世界坐标、前视第 {match['frame']} 帧、同一裁切", 26, True)
    for x, state in zip(columns, states):
        panel(shared / (state["id"] + ".png"), (x, 816, 348, 146))
    footer = (
        "原照记录完整格栅。断条、安装姿态、座圈与井体按设计补全。"
        if is_rect
        else "圆盖起伏幅度、盖厚、井深和三种异常状态由设计给定。一处孔缘跨过遮挡的树枝。"
    )
    for x, (case, record) in zip([32, 770], sorted(road_pair)):
        words(x, 991, f"{road_key}   路段 {record['site']} · 前视第 {record['frame']} 帧", 23, True)
        panel(run / 'evaluation/streets' / (case + '-after.jpg'), (x, 1033, 717, 403))
    words(32, 1471, footer, 21, color=muted)
    credit = (
        "U.S. Air Force / A1C Julian Batista · DVIDS 9735953 · Public domain"
        if is_rect
        else "ambientCG · Manhole Cover 004 · CC0"
    )
    words(
        32,
        1520,
        credit + "     |     PandaSet · Scale AI and Hesai · CC BY 4.0 and PandaSet Dataset Terms",
        15,
        color=muted,
    )
    page.save(out, quality=94, subsampling=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--site", default="109")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.report, args.site, args.out)
