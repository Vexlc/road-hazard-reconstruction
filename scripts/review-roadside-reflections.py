"""Replay registered road cameras and isolate the image change due to finite-distance reflections."""

import argparse
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

rig = import_module("check-cone-cycles")
from metric_reflection import MetricReflectors

ROOT = rig.ROOT
PAIR = "P256-P258"
LAYOUT = ROOT / "gen2hole/examples/roadside-reflections/pair.json"


def main(views, models):
    append = import_module("review-asphalt-breakouts").append_rear_hold
    append(views / "audi", PAIR, ["P256   碎边宽坑", "P258   椭圆碎石积水坑"], views / "audi-front-rear.mp4")
    stats = {}
    pictures = []
    with TemporaryDirectory(dir=views) as named:
        tmp = Path(named)
        for site in ("urban", "avenue", "audi"):
            folder = views / site
            target = models / site
            meta = json.loads((folder / "index.json").read_text())[PAIR]["views"]
            for sensor, row in meta.items():
                camera = folder / PAIR / sensor

                def render(packet, dest, *opts):
                    rig.cli(
                        "render", LAYOUT, folder / "anchor.json", packet, dest, "--asset-root", target, *opts
                    )

                with np.load(camera / "view.npz", allow_pickle=False) as raw:
                    near = MetricReflectors.from_packet(raw)
                    if near is None:
                        raise ValueError(f"Missing observed reflecting surfaces in {site}/{sensor}")
                    np.savez_compressed(
                        tmp / "directional.npz",
                        **{k: raw[k] for k in raw.files if k not in MetricReflectors.fields},
                    )
                    np.savez_compressed(tmp / "surface.npz", **near.packet())
                render(tmp / "directional.npz", tmp / "flat")
                render(tmp / "directional.npz", tmp / "metric", "--reflection-mesh", tmp / "surface.npz")
                for name in ("after.png", "part.png", "instance.png", "range.npy"):
                    if (tmp / "metric" / name).read_bytes() != (camera / name).read_bytes():
                        raise ValueError(
                            f"External reflection mesh replay differs at {site}/{sensor}/{name}"
                        )
                    if (
                        name != "after.png"
                        and (tmp / "flat" / name).read_bytes() != (camera / name).read_bytes()
                    ):
                        raise ValueError("Changing reflection moved the actual visible geometry")
                base = np.asarray(Image.open(tmp / "flat/after.png"))
                after = np.asarray(Image.open(camera / "after.png"))
                parts = np.asarray(Image.open(camera / "part.png"))
                wet = parts == 4
                delta = np.abs(after.astype(int) - base).max(2)
                stat = dict(
                    water_pixels=int(wet.sum()),
                    changed_water_pixels=int((delta[wet] > 0).sum()),
                    max_channel_delta=int(delta.max()),
                    triangles=len(near.triangles),
                    frame=row["frame"],
                    protected_edits=row["protected_edits"],
                    both_visible_frames=len(row["frames_showing_all_instances"]),
                )
                if row["protected_edits"] or not stat["both_visible_frames"]:
                    raise ValueError(f"Missing safe paired exposure {site}/{sensor}")
                stats[site + "/" + sensor] = stat
                print(site, sensor, stat, flush=True)
                if (site, sensor) in (
                    ("avenue", "image_03"),
                    ("audi", "rear_center"),
                    ("audi", "front_left"),
                    ("urban", "image_00"),
                ):
                    ids = np.asarray(Image.open(camera / "instance.png"))
                    yy, xx = np.nonzero(ids == 2)
                    if not len(xx):
                        raise ValueError("Wet asset missing")
                    box = (
                        max(0, xx.min() - 15),
                        max(0, yy.min() - 13),
                        min(ids.shape[1], xx.max() + 16),
                        min(ids.shape[0], yy.max() + 14),
                    )
                    pictures.append(
                        (
                            site,
                            sensor,
                            stat,
                            [
                                Image.open(p).convert("RGB").crop(box)
                                for p in (
                                    camera / "before.png",
                                    tmp / "flat/after.png",
                                    camera / "after.png",
                                )
                            ],
                        )
                    )
    if sum(r["changed_water_pixels"] for r in stats.values()) < 30:
        raise ValueError("No measurable reflections in real camera data")
    page = Image.new("RGB", (990, 916), "#20262b")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 19, index=2)
    for j, title in enumerate(("同一曝光 · 原路面", "天空与坑壁反光", "加入实测近物反光")):
        draw.text((j * 330 + 14, 8), title, font=font, fill="white")
    for i, (site, sensor, stat, imgs) in enumerate(pictures):
        y = 45 + i * 216
        name = {
            "urban": "KITTI 0000 前视",
            "avenue": "KITTI 0003 右侧",
            "audi": "Audi 驶过后车尾" if sensor == "rear_center" else "Audi 左前",
        }[site]
        draw.text(
            (14, y),
            f"{name} · 第 {stat['frame']} 帧 · {stat['changed_water_pixels']} 个水面像素出现近物贡献",
            font=font,
            fill="white",
        )
        for j, im in enumerate(imgs):
            crop = ImageOps.contain(im, (314, 175), Image.Resampling.LANCZOS)
            page.paste(crop, (j * 330 + (330 - crop.width) // 2, y + 30 + (175 - crop.height) // 2))
    page.save(views / "real-reflections.jpg", quality=94)
    rig.save_json(views / "real-check.json", stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("views", type=Path)
    p.add_argument("models", type=Path)
    a = p.parse_args()
    main(a.views, a.models)
