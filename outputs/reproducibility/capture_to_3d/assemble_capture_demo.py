"""Create the source, geometry and in-scene comparison for the four finished assets."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

CASES = [
    ("stereo_pothole_dataset1_02", "实拍深坑", "双目样本 01 / 02 · 坑口约 8 × 8 cm · P95 深约 4.4 cm"),
    ("stereo_pothole_dataset2_25", "实拍浅坑", "双目样本 02 / 25 · 坑口约 12 × 11 cm · P95 深约 1.4 cm"),
    ("photo_manhole_missing", "窨井盖缺失", "实拍井沿 · 井筒和井底建模"),
    ("photo_manhole_raised65", "窨井盖掀开", "实拍盖面纹理 · 建模开盖 65°")
]
FONT_FAMILY = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG = "#101721"
TEXT = "#f4f5f7"
MUTED = "#b4c0cd"
ACCENT = "#8bd9ec"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_FAMILY, size)


def asset_paths(project, name):
    return (project / "outputs/road_hazards" / name,
            project / "outputs/road_hazard_insertions" / ("B_055_" + name))


def fit(source, size, center=(.5, .5)):
    resampling = Image.Resampling.LANCZOS
    return ImageOps.fit(source.convert("RGB"), size, method=resampling, centering=center)


def load(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def close_crop(image, is_pit):
    if is_pit:
        return image.crop((540, 285, 1380, 795))
    return image.crop((130, 136, 1790, 746))


def source_crop(asset, name):
    image = load(asset / "source_crop.png")
    if "pothole" not in name:
        return image
    mask = load(asset / "source_mask.png").getbbox()
    x0, y0, x1, y1 = mask
    cx, cy = (x0 + x1) * .5, (y0 + y1) * .5
    half = max((x1 - x0) * .78, (y1 - y0) * 1.09)
    return image.crop((max(0, round(cx - half)), max(0, round(cy - .70 * half)),
                       min(image.width, round(cx + half)), min(image.height, round(cy + .70 * half))))


def storyboard(project, destination):
    canvas = Image.new("RGB", (1600, 1384), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 22), "道路病害 · 实拍到 3D 场景", font=font(44, True), fill=TEXT)
    for label, x in [("原始照片裁图", 36), ("三维几何", 475), ("PandaSet 055 插入近景（裁切放大）", 914)]:
        draw.text((x, 96), label, font=font(24, True), fill=ACCENT)
    for index, (name, title, context) in enumerate(CASES):
        asset, insert = asset_paths(project, name)
        y = 142 + 303 * index
        draw.text((36, y), title, font=font(29, True), fill=TEXT)
        draw.text((230, y + 7), context, font=font(20), fill=MUTED)
        closeup_file = "closeup_lid_face.jpg" if "raised" in name else "closeup.jpg"
        images = [source_crop(asset, name), load(asset / "geometry.jpg"),
                  close_crop(load(insert / closeup_file), "pothole" in name)]
        for image, x, width in zip(images, [36, 475, 914], [412, 412, 650]):
            canvas.paste(fit(image, (width, 239)), (x, y + 44))
    canvas.save(destination, quality=96, subsampling=0)


def demo(project, destination):
    writer = imageio.get_writer(str(destination), fps=24, codec="libx264",
                                quality=8, macro_block_size=2, ffmpeg_log_level="error",
                                output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
    for name, title, context in CASES:
        asset, insert = asset_paths(project, name)
        layout = Image.new("RGB", (1920, 1080), BG)
        draw = ImageDraw.Draw(layout)
        draw.text((36, 25), title, font=font(45, True), fill=TEXT)
        draw.text((36, 91), context, font=font(25), fill=MUTED)
        draw.text((36, 152), "原始照片裁图", font=font(24, True), fill=ACCENT)
        layout.paste(fit(source_crop(asset, name), (430, 275)), (36, 193))
        draw.text((36, 500), "独立 3D 资产", font=font(24, True), fill=ACCENT)
        layout.paste(fit(load(asset / "gaussians/preview.png"), (430, 285)), (36, 541))
        draw.text((505, 1003), "PandaSet 055 · 3DGS 场景联合渲染", font=font(24), fill=MUTED)

        street = imageio.get_reader(insert / "pure_3dgs.mp4", "ffmpeg")
        # Present the original perspective around the moment of closest useful approach.
        for index in range(35):
            frame_id = 28 + int(index * 10 / 24)
            frame = Image.fromarray(street.get_data(frame_id))
            panel = layout.copy()
            d = ImageDraw.Draw(panel)
            d.text((505, 152), "行车视角", font=font(24, True), fill=ACCENT)
            panel.paste(fit(frame, (1379, 776)), (505, 193))
            writer.append_data(np.asarray(panel))
        street.close()

        with imageio.get_reader(insert / "closeup_3dgs.mp4", "ffmpeg") as near:
            for frame in near:
                panel = layout.copy()
                d = ImageDraw.Draw(panel)
                label = "同一位置 · 新视角近景"
                if "pothole" in name:
                    label += "（裁切放大）"
                d.text((505, 152), label, font=font(24, True), fill=ACCENT)
                image = Image.fromarray(frame)
                if "pothole" in name:
                    image = close_crop(image, True)
                panel.paste(fit(image, (1379, 776)), (505, 193))
                writer.append_data(np.asarray(panel))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    out = args.project / "outputs"
    storyboard(args.project, out / "实拍到3D_道路病害对比.jpg")
    demo(args.project, out / "实拍道路病害_场景演示.mp4")
