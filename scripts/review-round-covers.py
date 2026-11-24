"""Join both registered drives and show the front, actual rear and independent overhead views."""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("M240-A241", "A240-M241")


def video(views, site, dest):
    clips = [views / pair / site / "driving.mp4" for pair in PAIRS]
    with TemporaryDirectory() as temp:
        work = Path(temp)
        for i, path in enumerate(clips):
            (work / f"{i}.mp4").symlink_to(path.resolve(strict=True))
        manifest = work / "clips.txt"
        manifest.write_text("file '0.mp4'\nfile '1.mp4'\n")
        subprocess.run(
            [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
        )
    expected = sum(count_frames_and_secs(str(p))[0] for p in clips)
    if count_frames_and_secs(str(dest))[0] != expected:
        raise ValueError("The combined video lost a frame")


def plate(views):
    page = Image.new("RGB", (1380, 616), "#202529")
    draw = ImageDraw.Draw(page)
    family = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(family, 20, index=2)
    for j, title in enumerate(("Audi 左前鱼眼", "Audi 驶过后的车尾鱼眼", "Cycles 俯视 · 白块为遮挡测标")):
        draw.text((j * 460 + 15, 8), title, font=font, fill="white")
    for i, pair in enumerate(PAIRS):
        top = 47 + i * 282
        draw.text(
            (15, top),
            pair + ("   通气井缺盖 / 填芯盖开裂" if i == 0 else "   通气盖错位 / 填芯井缺盖"),
            font=font,
            fill="#e6e1d8",
        )
        for j, name in enumerate(("front_left", "rear_center", "ortho")):
            if name == "ortho":
                folder = views / "cycles" / pair / name / "native"
                image = folder / "rgb.png"
                mask = np.load(folder / "id.npy") >= 32
            else:
                folder = views / pair / "audi" / pair / name
                image = folder / "after.png"
                mask = np.asarray(Image.open(folder / "instance.png")) > 0
            yy, xx = np.nonzero(mask)
            with Image.open(image) as source:
                box = (
                    max(0, xx.min() - 18),
                    max(0, yy.min() - 16),
                    min(source.width, xx.max() + 19),
                    min(source.height, yy.max() + 17),
                )
                crop = ImageOps.contain(source.convert("RGB").crop(box), (436, 236), Image.Resampling.LANCZOS)
            page.paste(crop, (j * 460 + (460 - crop.width) // 2, top + 32 + (236 - crop.height) // 2))
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/scanned-round-covers")
    parser.add_argument(
        "--plate", type=Path, help="Save a plate using the native Cycles and two real camera views"
    )
    args = parser.parse_args()
    video(args.views, "audi", args.views / "audi-front-rear.mp4")
    video(args.views, "kitti", args.views / "kitti-front-side.mp4")
    if args.plate:
        args.plate.parent.mkdir(parents=True, exist_ok=True)
        plate(args.views).save(args.plate, quality=92)


if __name__ == "__main__":
    main()
