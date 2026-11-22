"""Show the native RGB-D paver, its measured incline and its visibility in two driving scenes."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

parts = import_module("render-observed-pavement")
Plate = parts.Plate


def panel(plate, image, box):
    x, y, w, h = box
    plate.draw.rectangle((x, y, x + w - 1, y + h - 1), fill="#dfe3da")
    im = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
    plate.image.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def make(run, key, report, out):
    folder = run / "evaluation" / key
    survey = parts.read_json(run / "series/calibration" / (key + "_capture.json"))["spec"]
    measurements = parts.read_json(run / "evaluation/measurements.json")[key]
    proof = parts.read_json(report)
    observations = parts.observations(run, key)
    p = Plate(1440, 1100)
    p.text(28, 24, "泥水坑旁，一块砖的两端跨过路面", 43, True)
    p.text(30, 90, f"{key}　·　PothRGBD {survey['anchor']}　·　青线为实拍砖轮廓，黄线为原始坑洞标注", 22)
    spec = np.asarray(survey["surface_regions"]["paver"]["outline_px"])
    with Image.open(folder / "source-outlines.png") as image:
        panel(p, image, (28, 175, 450, 294))
        low, high = spec.min(0) - 17, spec.max(0) + 18
        panel(p, image.crop((*low, *high)), (496, 175, 452, 294))
    with Image.open(folder / "isolated-render.jpg") as image:
        pix = np.asarray(image.convert("RGB"), dtype=int)
        y, x = np.where(abs(pix - pix[0, 0]).max(2) > 22)
        crop = image.crop(
            (
                max(0, x.min() - 9),
                max(0, y.min() - 9),
                min(image.width, x.max() + 10),
                min(image.height, y.max() + 10),
            )
        )
        panel(p, crop, (966, 175, 446, 294))
    for x, title in [(28, "实拍整坑"), (496, "实拍砖缘放大"), (966, "单独导出的砖顶与侧面")]:
        p.text(x, 137, title, 22, True)
    p.text(28, 505, "沿砖面倾斜方向的高程", 27, True)
    p.text(28, 551, "相对邻近路面，毫米", 17)
    sample = proof["slope_section"]
    pos = np.array([s["position_m"] for s in sample])
    xs = (108 + (pos - pos.min()) / (pos.max() - pos.min()) * 570).round().astype(int)

    def yy(mm):
        return int(696 - 2.6 * mm)

    for mm in [-30, 0, 30]:
        p.draw.line((95, yy(mm), 697, yy(mm)), width=2, fill="#75887e" if mm == 0 else "#c6cec1")
        p.text(38, yy(mm) - 10, f"{mm:+d}" if mm else "0", 16)
    for field, color, rad in [("mesh_m", "#b46839", 5), ("measured_m", "#235c57", 2)]:
        points = [(int(x), yy(s[field] * 1000)) for x, s in zip(xs, sample)]
        p.draw.line(points, fill=color, width=2)
        for x, y in points:
            p.draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=color)
    p.text(96, 799, f"{pos[0] * 100:.1f} cm", 16)
    p.text(550, 799, f"+{pos[-1] * 100:.1f} cm", 16)
    p.text(28, 846, "橙色为导出网格，深绿为留出量程。每一点取该段中位数。", 19)
    p.text(
        28,
        893,
        f"砖面倾角　实测 {proof['heldout_top_tilt_deg']:.2f}°　网格 {proof['mesh_top_tilt_deg']:.2f}°",
        24,
        True,
    )
    errs = measurements["withheld_mesh"]
    p.text(
        28,
        945,
        f"整片资产留出 {errs['samples']:,} 个量程，光轴误差中位数 {errs['median_mm']:.2f} 毫米。",
        19,
    )
    p.text(28, 1000, "同次曝光的分块验证。湿区记录可见表面，水深与砖底面未知。", 18)
    p.text(736, 505, "同一资产迁入两段道路", 27, True)
    p.text(736, 551, "左图为车载前视局部，右图标出砖顶（青）和侧面（橙）。", 19)
    for i, site in enumerate(["034", "109"]):
        view = next(q for q in observations["camera"] if q["site"] == site and q["direction"] == "前视")
        y = 607 + i * 232
        p.text(
            736,
            y,
            f"PandaSet {site}　·　帧 {view['frame']}　·　砖面共 {view['part_pixels']} 像素",
            20,
            True,
        )
        for j, name in enumerate(["texture", "overlay"]):
            p.picture(folder / view[name], 736 + j * 344, y + 44, 333, 145)
    p.save(out)
    return observations


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True, type=Path)
    cli.add_argument("--id", required=True)
    cli.add_argument("--measurements", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--parts", required=True, type=Path)
    a = cli.parse_args()
    parts.save_json(a.parts, make(a.run, a.id, a.measurements, a.out))
