"""Create the library overview and a numbered contact sheet for every asset."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

CATEGORIES = {
    "stereo": "双目实拍坑洞", "rgbd": "RGB-D 实拍坑洞", "pothole": "建模坑洞",
    "missing": "缺盖窨井", "raised": "掀盖与移位"
}
FEATURED = {
    "stereo": [
        ("stereo_pothole_fan_1_01", "深陷"),
        ("stereo_pothole_fan_2_28", "骨料剥落"),
        ("stereo_pothole_fan_2_40", "狭长"),
        ("stereo_pothole_fan_3_05", "潮湿")
    ],
    "rgbd": [
        ("rgbd_pothole_20250305_135046", "斜向破口"),
        ("rgbd_pothole_20250305_123822", "宽坑"),
        ("rgbd_pothole_20250227_135438", "浅层碎裂"),
        ("rgbd_pothole_20250305_142220", "裂损")
    ],
    "pothole": [
        ("model_pothole_deep_break_fine_asphalt_02", "深坑"),
        ("model_pothole_waterlogged_aged_asphalt_02", "积水"),
        ("model_pothole_wheel_rut_chip_seal_02", "车辙长坑"),
        ("model_pothole_joined_concrete_02", "连通坑")
    ],
    "missing": [
        ("model_missing_round_masonry_aged_asphalt", "圆井砖壁"),
        ("model_missing_square_broken_flange_fine_asphalt", "方井破损井沿"),
        ("model_missing_oval_flush_concrete", "椭圆开口"),
        ("model_missing_rectangular_settled_concrete", "矩形沉降井框")
    ],
    "raised": [
        ("model_raised_round_raised62_aged_asphalt", "圆盖掀起 62°"),
        ("model_raised_square_propped43_concrete", "方盖掀起 43°"),
        ("model_raised_oval_upright82_concrete", "椭圆盖掀起 82°"),
        ("model_raised_rectangular_removed_side_chip_seal", "盖板移到井旁")
    ]
}

REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size=18, heavy=False):
    return ImageFont.truetype(BOLD if heavy else REGULAR, size)


def preview(library, m, dimensions):
    with Image.open(library / m["preview"]) as im:
        return ImageOps.fit(im.convert("RGB"), dimensions, method=Image.Resampling.LANCZOS)

def featured_preview(library, metadata, dimensions):
    with Image.open(library / metadata["preview"]) as im:
        rgb = im.convert("RGB")
        pixels = np.asarray(rgb, dtype="i2")
        backdrop = np.median(pixels[[0, -1]][:, [0, -1]].reshape(-1, 3), axis=0)
        visible = np.max(np.abs(pixels-backdrop), axis=2) > 12
        rows, columns = np.where(visible)
        if len(rows):
            padding = 21
            rgb = rgb.crop((max(0, columns.min()-padding), max(0, rows.min()-padding),
                            min(im.width, columns.max()+padding), min(im.height, rows.max()+padding)))
        return ImageOps.pad(rgb, dimensions, method=Image.Resampling.LANCZOS,
                            color=tuple(int(channel) for channel in backdrop))


def hero(library, rows):
    counts = Counter(m["category"] for m in rows)
    lookup = {m["id"]: m for m in rows}
    sheet = Image.new("RGB", (1700, 1080), "#131c26")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), f'{len(rows):,} 个新增道路病害 3D 资产', font=font(42, True), fill="#f4f7fa")
    draw.text((32, 81), "每个资产提供带纹理 GLB、3DGS PLY 和放置用路面高程", font=font(22), fill="#a8bfd0")
    for col, category in enumerate(CATEGORIES):
        x = 32 + 328*col
        draw.text((x, 126), f'{CATEGORIES[category]} · {counts[category]}', font=font(23, True), fill="#a9dce3")
        entries = [(name, title) for name, title in FEATURED[category] if name in lookup]
        missing = 4-len(entries)
        entries.extend((row["id"], "样例") for row in rows if row["category"] == category and row["id"] not in dict(entries) and missing)
        for i, (name, title) in enumerate(entries[:4]):
            y = 168 + 220*i
            m = lookup[name]
            sheet.paste(featured_preview(library, m, (311, 176)), (x, y))
            draw.text((x, y+180), title, font=font(20), fill="#eef2f6")
            dim = f'{m["width_m"]*100:.0f} × {m["length_m"]*100:.0f} cm'
            draw.text((x+196, y+185), dim, font=font(14), fill="#b7c6d3")
    sheet.save(library / "扩充资产预览.jpg", quality=95, subsampling=0)


def contacts(library, rows, storage):
    by_cat = [[m for m in rows if m["category"] == category] for category in CATEGORIES]
    number = 0
    references = []
    storage.mkdir(parents=True, exist_ok=True)
    for items in by_cat:
        category = items[0]["category"]
        n = 45
        for page_num, start in enumerate(range(0, len(items), n), 1):
            page = items[start:start+n]
            overview = Image.new("RGB", (1800, 112 + 163*((len(page)+8)//9)), "#131c26")
            d = ImageDraw.Draw(overview)
            d.text((18, 14), f'{CATEGORIES[category]} · 第 {page_num} 页', font=font(30, True), fill="#f4f7fa")
            d.text((20, 62), f'{start+1}–{start+len(page)} / {len(items)}', font=font(19), fill="#a0cbd2")
            for j, m in enumerate(page):
                x, y = 15 + (j%9)*198, 104 + (j//9)*163
                number += 1
                overview.paste(preview(library, m, (187, 125)), (x, y))
                d.text((x+1, y+126), f'{number:04d}   {m["width_m"]*100:.0f} × {m["length_m"]*100:.0f} cm',
                       font=font(16), fill="#f4f7fa")
                references.append({"number": number, "id": m["id"], "category": category, "mesh": m["glb"]})
            overview.save(storage / f'{category}_{page_num:02d}.jpg', quality=90)
    assert number == len(rows)
    (storage / "图集编号与资产ID.json").write_text(json.dumps(references, ensure_ascii=False, indent=2)+"\n")
    with ZipFile(library / "全部资产预览图集.zip", "w", compression=ZIP_DEFLATED, compresslevel=2) as archive:
        for file in sorted(storage.iterdir()):
            archive.write(file, file.name)
        assert sum(it.endswith(".jpg") for it in archive.namelist()) >= len(rows)/45
    with ZipFile(library / "全部资产预览图集.zip") as archive:
        assert archive.testzip() is None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.library / "asset_manifest.json").read_text())
    hero(args.library, records)
    contacts(args.library, records, args.work)
