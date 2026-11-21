"""Index local mesh, Gaussian and camera evidence by asset ID for the offline atlas."""
import json
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "tools/asset_atlas"
GEN = ROOT / "gen2hole"
RELEASE = (GEN / "published").resolve()
SENSORS = RELEASE / "道路病害_传感器与几何增强"
EXPANSION = ROOT / "outputs/road_hazard_expansion"
FACTORY = ROOT / "outputs/道路病害_稳定生成与迁移_v3"
JOINT = ROOT / "outputs/road_hazard_joint_scenes"
CATEGORIES = {"stereo": "坑洞", "rgbd": "坑洞", "pothole": "坑洞", "missing": "缺盖井", "raised": "异常井盖"}
METHODS = {"stereo": "实拍双目", "rgbd": "实拍 RGB-D"}
TERMS = dict(round="圆形", oval="椭圆", square="方形", rectangular="长方形", bowled="碗状坑",
             cluster="群坑", crescent="月牙坑", deep_break="深坑", joined="连通坑", patch_break="补丁破损",
             slab_corner="板角破损", spall="剥落", waterlogged="积水坑", wheel_rut="车辙坑",
             ajar14="微开", broken_flange="破损井沿", flush="齐平井沿", masonry="砖砌井壁",
             propped43="支起", raised62="掀开", removed_side="移至井旁", settled="沉降井框",
             skewed="斜移", slid="平移", tilted29="倾斜", upright82="近直立", aged_asphalt="老化沥青",
             chip_seal="碎石封层", concrete="水泥路面", fine_asphalt="细粒沥青")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return Path(os.path.relpath(path, ATLAS)).as_posix()


def entry(row, base, method, label, preview):
    shape = "P95 凹深" if row["category"] in METHODS else "设计深度"
    depth = row["depth_p95_m"] if row["category"] in METHODS else row.get("depth_measured_m", row.get("depth_design_m"))
    if "depth_measured_m" in row:
        shape = "可见井底深度"
    return dict(id=row["id"], label=label, category=CATEGORIES[row["category"]], method=method,
                preview=link(preview), width=row["width_m"], length=row["length_m"], depth=depth,
                depthLabel=shape, gaussians=row["gaussians"],
                glb=link(base / row["glb"]), ply=link(base / row["ply"]),
                metadata=link(base / row["metadata"]), surface=link(base / row["surface"]), scenes=[])


def camera_views(root):
    shots = {}
    for path in sorted((root / "camera").glob("*/views.json")):
        for shot in read(path):
            if shot["available"]:
                shots.setdefault(shot["asset_id"], []).append(dict(
                    label=f"{shot['site']} · {'前视' if shot['camera']=='front_camera' else '后视'} · 帧 {shot['frame']}",
                    after=link(path.parent / shot["after_stem"]), before=link(path.parent / shot["before_stem"]),
                    site=shot["site"], camera=shot["camera"].split("_")[0]))
    return shots


def generated():
    shots = camera_views(SENSORS)
    rows = {}
    for spec in read(GEN / "catalog/assets.json").values():
        folder = RELEASE / spec["published_dir"]
        bundle = folder.parents[2]
        meta = read(folder / "metadata.json")
        name = meta["id"]
        assert name == spec["id"] and meta["source_sha256"] == spec["source_sha256"]
        paths = dict(glb="asset.glb", ply="asset.ply", metadata="metadata.json", surface="surface.npz")
        row = entry(meta | paths, folder, "生成 RGB", spec["label"], bundle / "views" / (name+"_inspection.jpg"))
        row.update(reference=link(bundle / meta["source_image"]), reverse=link(bundle / "views" / (name+"_reverse.jpg")),
                   triangles=spec["triangles"], scenes=shots[name], sensor=link(SENSORS / "传感器对照.html"))
        rows[name] = row
    return rows


def captured_and_designed():
    rows = {}
    for folder in (EXPANSION, FACTORY / "资产库"):
        for r in read(folder / "asset_manifest.json"):
            cat = r["category"]
            if cat in METHODS:
                label = ("双目坑洞 · " if cat=="stereo" else "深度相机坑洞 · ") + str(r["family"])
            else:
                label = " · ".join(TERMS[r[k]] for k in ("family","state","material") if r.get(k))
            row = entry(r, folder, METHODS.get(cat,"程序化"), label, folder / r["preview"])
            row["triangles"] = r.get("glb_triangles") or sum(r["geometry_parts"].values())
            if cat in METHODS:
                row["reference"] = link((folder / r["metadata"]).parent / "source_crop.png")
            if r["id"] in rows:
                raise ValueError(f"Duplicate ID: {r['id']}")
            rows[r["id"]] = row
    for folder in (JOINT, FACTORY / "迁入场景"):
        for path in sorted(folder.glob("*/scene.json")):
            scene = read(path)
            for p in scene["placements"]:
                row = rows[p["asset_id"]]
                view = dict(label=f"{scene['site']} · {path.parent.name} · 迁入近景")
                if folder == JOINT:
                    view["after"] = link(path.parent / p["preview_file"])
                else:
                    category = p["category"]
                    view.update(after=link(path.parent / (category+"_after.png")),
                                before=link(path.parent / (category+"_before.png")))
                row["scenes"].append(view)
    return rows


def real_rgb():
    bundle = ROOT / "runs/rgb-box-P90/series"
    if not bundle.exists():
        return {}
    r, = read(bundle / "assets/asset_manifest.json")
    name = r["id"]
    row = entry(r, bundle / "assets", "实拍 RGB", r["label"], bundle / "views" / (name+"_inspection.jpg"))
    row.update(triangles=sum(r["geometry_parts"].values()), reference=link(bundle / r["source_image"]),
               reverse=link(bundle / "views" / (name+"_reverse.jpg")), audit=link(ROOT / "validation/rgb-box-P90.json"))
    case = bundle / "scenes/034_90"
    if (case / "scene.json").is_file():
        assert read(case/"scene.json")["assets"] == [name]
        row["scenes"].append(dict(label="034 · 034_90 · 迁入近景", after=link(case/"pothole_after.png"),
                                  before=link(case/"pothole_before.png")))
        row["video"] = link(bundle / "sensor_views/034_90/front_after.mp4")
    sensor_root = bundle.parent / "sensors"
    if sensor_root.is_dir():
        row["scenes"] = camera_views(sensor_root)[name] + row["scenes"]
        row["scan"] = link(sensor_root / "checks/scan_audit.json")
    row["related"] = read(ROOT / "validation/rgb-box-P90.json")["source_asset"]
    return {name: row}


def photo_families():
    rows = {}
    for info in read(ROOT / 'catalog/chamber-families.json'):
        run = ROOT / info['run']
        if not run.exists():
            continue
        bundle = run / 'series'
        spec = read(bundle / 'calibration/survey.json')
        state_ids = [s['id'] for s in spec['states']]
        scenes = {}
        for path in sorted((bundle / 'scenes').glob('*/scene.json')):
            for placed in read(path)['placements']:
                cat = placed['category']
                scenes.setdefault(placed['asset_id'],[]).append(dict(label=path.parent.name+" · 迁入近景",
                    after=link(path.parent/(cat+'_after.png')),before=link(path.parent/(cat+'_before.png'))))
        sensor_root = run/'sensors'
        if (run/'sensors-multisite/checks/camera_audit.json').is_file():
            sensor_root = run/'sensors-multisite'
        shots = camera_views(sensor_root)
        paired = {}
        for path in sorted((bundle/'comparison').glob('*/comparison.json')):
            match = read(path)
            for key in match['states']:
                paired.setdefault(key,[]).append(dict(label=f"{match['site']} · 前视 · 同帧 {match['frame']}",
                    site=match['site'],camera='front',before=link(path.parent/'before.png'),after=link(path.parent/(key+'.png'))))
        for meta in read(bundle / 'assets/asset_manifest.json'):
            key = meta['id']
            if key in rows:
                raise ValueError(f"Asset ID collision: {key}")
            row = entry(meta, bundle/'assets', info['method'], meta['label'], bundle/'views'/(key+'_inspection.jpg'))
            row.update(family=spec['family'], states=state_ids, label='圆井 · '+meta['label'], state=meta['label'],
                       reference=link(bundle/'references/frame.png'), lidReference=link(bundle/'references/lid.png'),
                       audit=link(ROOT / info['audit']), reverse=link(bundle/'views'/(key+'_reverse.jpg')),
                       credit=spec.get('source_url'), note=info['note'], triangles=sum(meta['geometry_parts'].values()),
                       scenes=paired.get(key,[])+shots.get(key,[])+scenes.get(key,[]))
            if meta['lid_angle_degrees'] == 0 and meta['lid_offset_m'] == [0,0]:
                row['category']='落座参照'
            if (sensor_root/'checks/scan_audit.json').is_file():
                row['scan']=link(sensor_root/'checks/scan_audit.json')
            rows[key]=row
    return rows


def crack_examples():
    root = ROOT/'runs/photo-cracks'
    if not root.exists():
        return {}
    bundle = root/'series'
    provenance = read(GEN/'examples/photo-cracks/sources.json')
    shots = camera_views(root/'sensors')
    rows = {}
    for meta in read(bundle/'assets/asset_manifest.json'):
        key=meta['id']
        row=entry(meta,bundle/'assets',"实拍 RGB",meta['label'],bundle/'views'/(key+'_inspection.jpg'))
        row.update(category="裂缝",depthLabel="设计深度上限",triangles=sum(meta['geometry_parts'].values()),
                   note="实拍 RGB 和人工像素掩膜给出缝迹与分叉。裁切图按设计跨度映射到局部平面，沟槽剖面与深度由设计给定。",
                   reference=link(bundle/meta['source_image']),mask=link(bundle/'calibration'/(key+'_outline.png')),
                   reverse=link(bundle/'views'/(key+'_reverse.jpg')),credit=provenance['source'],scenes=shots.get(key,[]))
        for path in sorted((bundle/'scenes').glob('*/scene.json')):
            if key in read(path)['assets']:
                row['scenes'].append(dict(label=path.parent.name+' · 迁入近景',after=link(path.parent/'pothole_after.png'),
                    before=link(path.parent/'pothole_before.png')))
        rows[key]=row
    return rows


def main():
    rows = {}
    for group in (generated(), captured_and_designed(), real_rgb(), photo_families(), crack_examples()):
        if group.keys() & rows.keys():
            raise ValueError("Asset ID collision")
        rows.update(group)
    for row in rows.values():
        orbit = ATLAS / "orbits" / (row["id"] + ".webp")
        if orbit.is_file():
            row["orbit"] = link(orbit)
        if row.get("related"):
            rows[row["related"]]["related"] = row["id"]
    featured = ["A93","P90","P102","A13","M11","P02_wet_wheel_track","rgbd_pothole_20250227_135438","A20",
                "model_raised_rectangular_removed_side_chip_seal","stereo_pothole_fan_1_03","P20",
                "M03_brick_lined","A11","model_pothole_cluster_chip_seal_01","M13","P13"]
    ordered = [rows[k] for k in dict.fromkeys(featured + sorted(rows)) if k in rows]
    # A local script works when opened directly with file://, including in offline browsers.
    data = json.dumps(ordered, ensure_ascii=False, separators=(",",":"), allow_nan=False)
    target = ATLAS / "catalog.js"
    with NamedTemporaryFile(mode="w", dir=ATLAS, encoding="utf-8", delete=False) as f:
        f.write("window.roadAssets="+data+";\n")
    os.replace(f.name,target)
    print(f"{len(rows)} assets, {sum(len(r['scenes']) for r in ordered)} views. {ATLAS / 'index.html'}")


if __name__ == "__main__":
    main()
