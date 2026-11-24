"""Append the jointly visible rear exposure to the actual drive for a longer inspection of both openings."""

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import numpy as np
from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def append_rear_hold(site, pair, labels, dest):
    folder = site / pair / "rear_center"
    frame = json.loads((site / "index.json").read_text())[pair]["views"]["rear_center"][
        "frame"
    ]
    ids = np.array(Image.open(folder / "instance.png"))
    card = Image.new("RGB", (960, 480), "#20272a")
    draw = ImageDraw.Draw(card)
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font, 26, index=2)
    small = ImageFont.truetype(font, 19, index=2)
    draw.text(
        (16, 9), f"Audi 车尾鱼眼 · 驶离后同帧停留   {frame}", font=title, fill="white"
    )
    for i, label in enumerate(labels, 1):
        x0 = (i - 1) * 480 + 12
        draw.text((x0, 54), label, font=small, fill="#d4dedb")
        yy, xx = np.nonzero(ids == i)
        y, x = ids.shape
        margin = 16
        box = (
            max(0, xx.min() - margin),
            max(0, yy.min() - margin),
            min(x, xx.max() + margin + 1),
            min(y, yy.max() + margin + 1),
        )
        for j, (name, text) in enumerate([("before", "原曝光"), ("after", "同帧迁入")]):
            crop = ImageOps.contain(
                Image.open(folder / (name + ".png")).crop(box),
                (450, 158),
                Image.Resampling.LANCZOS,
            )
            draw.text((x0, 82 + j * 194), text, font=small, fill="white")
            card.paste(
                crop,
                (
                    x0 + (450 - crop.width) // 2,
                    112 + j * 194 + (158 - crop.height) // 2,
                ),
            )
    with TemporaryDirectory(dir=dest.parent) as temp:
        work = Path(temp).resolve()
        (work / "drive.mp4").symlink_to((site / "driving.mp4").resolve())
        with imageio.get_writer(
            str(work / "rear.mp4"),
            fps=10,
            codec="libx264",
            quality=7,
            macro_block_size=2,
            output_params=["-threads", "3"],
        ) as writer:
            for _ in range(25):
                writer.append_data(np.array(card))
        (work / "list.txt").write_text("file 'drive.mp4'\nfile 'rear.mp4'\n")
        result = work / "joined.mp4"
        subprocess.run(
            [
                get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(work / "list.txt"),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(result),
            ],
            check=True,
        )
        if (
            count_frames_and_secs(str(result))[0]
            != count_frames_and_secs(str(site / "driving.mp4"))[0] + 25
        ):
            raise ValueError("Missing drive frames or rear inspection")
        result.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, default=ROOT / "outputs/asphalt-breakouts")
    args = parser.parse_args()
    append_rear_hold(args.views / "audi", "P246-P247", ["P246   碎集料龟裂坑", "P247   弧形剥落坑"], args.views / "audi-front-rear.mp4")


if __name__ == "__main__":
    main()
