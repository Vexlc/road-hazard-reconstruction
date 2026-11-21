"""Place one rebuilt group in a calibrated PandaSet road and optionally render front RGB and masks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
CATEGORIES = ("pothole", "missing", "raised")
SLOTS = {"034": (0, 2, 4), "109": (0, 2, 4), "110": (0, 1, 3)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def published_group(number):
    if number == 36:
        return ["P36", "M36"]
    if number == 37:
        return ["P37", "A36"]
    if 1 <= number <= 35:
        return [f"{prefix}{number:02}" for prefix in "PMA"]
    raise ValueError("新组合请用 --assets 指定此次构建中的资产编号")


def place(args, bundle, work):
    cfg = read(args.config)
    project = Path(cfg["road_project"])
    road = read(Path(cfg["road_config"]))
    original = read(Path(road["plan"]))
    manifest = read(bundle / "assets/asset_manifest.json")
    aliases = {alias: m for m in manifest for alias in (m["id"], m["id"].split("_")[0])}
    codes = args.assets or published_group(args.group)
    if any(k not in aliases for k in codes):
        raise ValueError(f"构建目录缺少组合成员 {sorted(set(codes) - aliases.keys())}")
    rows = sorted((aliases[k] for k in codes), key=lambda m: CATEGORIES.index(m["category"]))
    categories = [r["category"] for r in rows]
    if len(categories) != len(set(categories)):
        raise ValueError("每个道路锚点对应一个类别，同组中每类最多选一件")
    if any("ply" not in r for r in rows):
        raise ValueError("迁入需要 3DGS，请使用完整 build 输出")
    groups = [{"number": args.group, "assets": [r["id"] for r in rows]}]
    allocation = work / "case_plan.json"
    if allocation.exists() and read(allocation)["groups"] != groups:
        raise ValueError("此路段的组号已有其他资产，请选择新组号")
    save(allocation, dict(schema=1, groups=groups))
    plan = dict(category_order=categories, drive_start_by_site=original["drive_start_by_site"],
                slots_by_site={s: [original["slots_by_site"][s][indices[CATEGORIES.index(c)]]
                                   for c in categories] for s, indices in SLOTS.items()})
    save(work / "plan.json", plan)
    shutil.copytree(Path(road["work"]) / "road_masks", work / "road_masks", dirs_exist_ok=True)
    adapter = HERE / "native_entry.py"
    material_hashes = {r["id"]: hashlib.sha256((bundle / "assets" / r["ply"]).with_name("splat_material.npz").read_bytes()).hexdigest()
                       for r in rows}
    job = dict(output=str(bundle / "scenes"), work=str(work), libraries=[str(bundle / "assets")],
               plan=str(work / "plan.json"), configs=road["configs"], annotated_road_cache=road["annotated_road_cache"],
               road_occlusion_guard=any(c != "pothole" for c in categories), road_guard_spacing_m=.018,
               material_contact=False, mesh_cache=str(bundle / "mesh_cache"), case_groups=[args.group],
               image_fit_code=hashlib.sha256(adapter.read_bytes()).hexdigest(), image_labels=material_hashes)
    job_file = work / "job.json"
    save(job_file, job)
    modules = ["work/EasyVolcap", "work/asset_pipeline", "work/road_hazard_pipeline",
               "outputs/road_hazard_expansion/reproduction", "work/street_crafter"]
    env = dict(os.environ, ROAD_PROJECT=str(project), HAZARD_FACTORY_JOB=str(job_file), ROAD_BATCH_WORK=str(work),
               GEN2HOLE_BUNDLE=str(bundle), PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH=os.pathsep.join(str(project / d) for d in modules))
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    for script in ([adapter, HERE / "sensor_views.py"] if args.front else [adapter]):
        log = work / (script.stem + ".log")
        print(f"→ {script.name}  {args.site}_{args.group:02}  {log}", flush=True)
        with log.open("w") as file:
            result = subprocess.run([cfg["compute_python"], "-u", script, "--config", job["configs"][args.site],
                                     "mode", "trajectory"], env=env, cwd=project / "work/street_crafter",
                                    stdout=file, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(log.read_text(errors="replace")[-2500:])
    print(bundle / "scenes" / f"{args.site}_{args.group:02}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="例如 runs/demo/series")
    p.add_argument("--site", choices=SLOTS, required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--front", action="store_true", help="另输出 40 帧前视视频、掩膜和包围框")
    p.add_argument("--assets", nargs="+", help="覆盖默认发布组合，按本次构建中的 ID 选择")
    p.add_argument("--config", type=Path, default=HERE.parent / "config/local.json")
    args = p.parse_args()
    bundle = args.bundle.resolve()
    try:
        recipe = read(bundle / "run.json")
        if "ids" not in recipe or args.group <= 0:
            raise ValueError("--bundle 需要指向构建中的 pilot 或 series，--group 为正整数")
        if recipe["mesh_only"]:
            raise ValueError("迁入需要完整构建，请去掉 --mesh-only")
        work = bundle / "migration" / f"{args.site}_{args.group:02}"
        work.mkdir(parents=True, exist_ok=True)
        with (work / "place.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            place(args, bundle, work)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        p.exit(1, f"place: {error}\n")


if __name__ == "__main__":
    main()
