"""Write verified, independently extractable volumes of combined GLB and Gaussian assets."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from verify_library import ORDER

CATEGORY = {
    "stereo": "双目实拍坑洞", "rgbd": "RGB-D实拍坑洞", "pothole": "建模坑洞",
    "missing": "缺盖窨井", "raised": "掀盖与移位窨井"
}

NOTICE = """# 道路病害 3D 资产

这个分卷的每个资产都有 GLB、3DGS PLY、预览和 road cutout 高程及透明度字段。
GLB 单位为米，Y-up。3DGS PLY 单位为米，Z-up，包含一阶球谐系数、对数尺度、
WXYZ 四元数和透明度 logit。请用支持 3DGS 的渲染器查看 PLY。

凹陷资产原点为本地路平面的病害中心。插入时应替换同位置的原路面。
height 字段记录替换表面的标高，alpha 给出局部作用范围，width 和 length 为米。
GLB 文件打包了材质图。按 metadata.json 和 capture_metadata.json 查询尺寸、来源和参数。

双目帧来自 Rui (Ranger) Fan 等人的 Stereo Pothole Datasets。
https://github.com/ruirangerfan/stereo_pothole_datasets

RGB-D 帧来自 Mustafa Yurdakul 和 Şakir Tasdemir 的 PothRGBD 数据。
https://arxiv.org/abs/2505.04207
https://github.com/ymyurdakul/PData
相机内参使用标称值，重建尺寸为估计值。作者的公开下载仓库未附许可证文件，
对外发布请按来源数据的许可条款办理。

模型变体使用 ambientCG 的 CC0 材质和参数化几何。
https://ambientcg.com/view?id=Asphalt031
https://ambientcg.com/view?id=Concrete034
https://ambientcg.com/view?id=Concrete049
https://ambientcg.com/view?id=ManholeCover009
"""


def volumes(records, raw_limit):
    for category in ORDER:
        todo = sorted((r for r in records if r["category"] == category), key=lambda r: r["id"])
        current, size = [], 0
        index = 1
        for row in todo:
            next_size = row["glb_bytes"] + row["ply_bytes"]
            if current and next_size + size > raw_limit:
                yield category, index, current
                index += 1
                current, size = [], 0
            current.append(row)
            size += next_size
        if current:
            yield category, index, current


def pack(job):
    library, output, license_path, category, index, records = job
    filename = f"{CATEGORY[category]}_{index:02d}.zip"
    target = output / filename
    temporary = output / (filename + ".partial")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        archive.writestr("README.md", NOTICE)
        archive.writestr("asset_manifest.json", json.dumps(records, ensure_ascii=False, indent=2))
        if category == "stereo":
            archive.write(license_path, "credits/Fan_dataset_repository_LICENSE")
        for record in records:
            name = record["id"]
            for key in ("glb", "ply", "preview", "surface", "metadata"):
                path = library / record[key]
                archive.write(path, f"{category}/{name}/{path.name}")
            if category in {"stereo", "rgbd"}:
                folder = (library / record["metadata"]).parent
                archive.write(folder / "capture_metadata.json", f"{category}/{name}/capture_metadata.json")
                archive.write(folder / "source_crop.png", f"{category}/{name}/source_crop.png")
                archive.write(folder / "source_mask.png", f"{category}/{name}/source_mask.png")
    with ZipFile(temporary, "r") as archive:
        members = archive.namelist()
        assert sum(p.endswith(".glb") for p in members) == len(records)
        assert sum(p.endswith(".ply") for p in members) == len(records)
        assert archive.testzip() is None
    temporary.replace(target)
    digest = sha256()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(2**22), b""):
            digest.update(block)
    print("PACK", filename, len(records), target.stat().st_size, flush=True)
    return {"file": filename, "category": category, "assets": len(records),
            "bytes": target.stat().st_size, "sha256": digest.hexdigest(),
            "first_id": records[0]["id"], "last_id": records[-1]["id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fan-license", type=Path, required=True)
    parser.add_argument("--max-raw-mib", type=int, default=930)
    args = parser.parse_args()
    manifest = json.loads((args.library / "asset_manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.library, args.out, args.fan_license, k, i, r)
            for k, i, r in volumes(manifest, args.max_raw_mib*1024**2)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        reports = list(pool.map(pack, jobs))
    assert sum(r["assets"] for r in reports) == len(manifest)
    count = Counter(row["category"] for row in manifest)
    (args.out / "volumes.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print("PACKAGED", len(manifest), len(jobs), dict(count), flush=True)


if __name__ == "__main__":
    main()
