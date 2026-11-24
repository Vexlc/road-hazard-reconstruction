"""Compare the two water states on exactly the same real rear exposures."""

import argparse
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

rig = import_module("check-cone-cycles")
ROOT = rig.ROOT
SRC = ROOT / "gen2hole/examples/wet-pavement"
PAIRS = ("P252-P255", "P254-P253")
LABELS = ("浅层剥落坑", "粗石碗坑")


def read_product(folder, name):
    path = folder / name
    return np.load(path) if path.suffix == ".npy" else np.array(Image.open(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path)
    parser.add_argument("models", type=Path)
    args = parser.parse_args()
    out = args.views
    join = import_module("review-round-covers").video
    append = import_module("review-asphalt-breakouts").append_rear_hold
    page = Image.new("RGB", (1080, 514), "#202529")
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 21, index=2)
    draw = ImageDraw.Draw(page)
    for col, label in enumerate(("Audi 车尾 · 原路面", "同一位置 · 干坑", "同一位置 · 积水")):
        draw.text((col * 360 + 16, 8), label, font=font, fill="white")
    stats = {}
    with TemporaryDirectory(dir=out) as temp:
        tmp = Path(temp)
        clips = []
        rows = []
        for pair in PAIRS:
            layout = json.loads((SRC / (pair + ".json")).read_text())
            labels = [
                row["asset"] + "  " + row.get("water", {}).get("label", LABELS[i])
                for i, row in enumerate(layout["instances"])
            ]
            site = out / pair / "audi"
            clip = tmp / (pair + ".mp4")
            append(site, pair, labels, clip)
            clips.append(clip)
            meta = json.loads((site / "index.json").read_text())[pair]["views"]["rear_center"]
            rows.append((meta["frame"], meta["crop_xy"], meta["source_rgb_sha256"]))
            cam = site / pair / "rear_center"
            replay = tmp / pair
            model = args.models / "audi" / pair

            def render(packet, folder):
                rig.cli(
                    "render",
                    SRC / (pair + ".json"),
                    site / "anchor.json",
                    packet,
                    folder,
                    "--asset-root",
                    model,
                )

            render(cam / "view.npz", replay)
            for name in ("after.png", "part.png", "instance.png", "range.npy"):
                if (cam / name).read_bytes() != (replay / name).read_bytes():
                    raise ValueError(f"Real rear camera replay differs: {pair}/{name}")
            with np.load(cam / "view.npz", allow_pickle=False) as raw:
                np.savez_compressed(
                    tmp / "baked.npz", **{k: raw[k] for k in raw.files if not k.startswith("reflection_")}
                )
            render(tmp / "baked.npz", tmp / "control")
            before = np.asarray(Image.open(tmp / "control/after.png"), dtype=int)
            after = np.asarray(Image.open(cam / "after.png"), dtype=int)
            wet = np.asarray(Image.open(cam / "part.png")) == 4
            # Rotate a crop of the same four-subray exposure. Keep the world illumination fixed.
            yy, xx = np.nonzero(wet)
            r, c = np.mgrid[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
            source_xy = np.rot90(np.stack((c, r), axis=-1))
            np.save(tmp / "map.npy", source_xy)
            rig.cli("remap-camera", cam / "view.npz", tmp / "map.npy", tmp / "rotated.npz")
            render(tmp / "rotated.npz", tmp / "rotated")
            x, y = np.moveaxis(source_xy, -1, 0)
            for name in ("after.png", "part.png", "instance.png"):
                if not np.array_equal(read_product(cam, name)[y, x], read_product(tmp / "rotated", name)):
                    raise ValueError(
                        f"World probe or ray registration changed in the rotated crop: {pair}/{name}"
                    )
            expected_range = np.load(cam / "range.npy")[y, x]
            observed_range = np.load(tmp / "rotated/range.npy")
            valid = np.isfinite(expected_range)
            if not np.array_equal(valid, np.isfinite(observed_range)):
                raise ValueError("Rotated crop changes the depth support")
            range_error = float(np.max(abs(expected_range[valid] - observed_range[valid])))
            if range_error > 5e-6:
                raise ValueError(f"Rotated range drifted {range_error} m on {pair}")
            diff = np.max(np.abs(before - after), axis=2)
            if (diff[wet] > 0).sum() < 50 or diff[wet].max() < 5:
                raise ValueError("Insufficient observed water appearance change")
            stats[pair] = dict(
                rotated_replay_pixels=int(x.size),
                rotated_range_max_error_m=range_error,
                wet_pixels=int(wet.sum()),
                changed_wet_pixels=int((diff[wet] > 0).sum()),
                median_max_channel_change=float(np.median(diff[wet])),
                max_channel_change=int(diff.max()),
            )
        if rows[0] != rows[1]:
            raise ValueError("Rear exposure, crop or image hash differs between the layouts")
        join(out, "audi", out / "audi-front-rear.mp4", PAIRS, clips)
    join(out, "kitti", out / "kitti-front-side.mp4", PAIRS)
    for i, label in enumerate(LABELS):
        dry, wet = (out / p / "audi" / p / "rear_center" for p in (PAIRS if i == 0 else PAIRS[::-1]))
        ids = np.array(Image.open(dry / "instance.png"))
        yy, xx = np.nonzero(ids == i + 1)
        h, w = ids.shape
        box = (max(0, xx.min() - 22), max(0, yy.min() - 24), min(w, xx.max() + 23), min(h, yy.max() + 25))
        y = 48 + i * 228
        draw.text(
            (16, y),
            label + ("   P252 / P254 · 5 L" if i == 0 else "   P253 / P255 · 8 L"),
            font=font,
            fill="white",
        )
        for j, file in enumerate((dry / "before.png", dry / "after.png", wet / "after.png")):
            crop = ImageOps.contain(
                Image.open(file).convert("RGB").crop(box), (344, 185), Image.Resampling.LANCZOS
            )
            page.paste(crop, (j * 360 + (360 - crop.width) // 2, y + 35 + (185 - crop.height) // 2))
    page.save(out / "rear-dry-wet.jpg", quality=94)
    rig.save_json(out / "rear-check.json", stats)


if __name__ == "__main__":
    main()
