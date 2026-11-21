"""Rebuild metric road hazards from generated survey images."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
CATEGORIES = {"P": "pothole", "M": "missing", "A": "raised"}


def read(path):
    return path.read_text(encoding="utf-8")


def load(path):
    return json.loads(read(path))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(custom):
    if custom is None:
        return load(ROOT / "catalog/assets.json")
    rows = {}
    for design in load(custom / "calibration/designs.json"):
        key = design["id"]
        if not re.fullmatch(r"[PMA][0-9]{2,}", key) or key in rows:
            raise ValueError(f"设计编号无效或重复：{key}")
        if design["category"] != CATEGORIES[key[0]]:
            raise ValueError(f"编号和类别不匹配：{key}")
        rows[key] = dict(edition="series", key=key, id=key, category=design["category"], label=design["label"])
    return rows


def select(rows, ids):
    if not ids:
        raise ValueError("请选择资产编号或使用 --all")
    names = {row["id"]: code for code, row in rows.items()}
    selected = []
    for arg in ids:
        code = names.get(arg, arg.upper())
        if code not in rows:
            raise ValueError(f"找不到资产 {arg}，可用 list 查看编号")
        if code not in selected:
            selected.append(code)
    return selected


def designs(source):
    return {row["id"]: row for row in load(source / "calibration/designs.json")}


def input_root(custom, edition):
    return custom if custom is not None else ROOT / "inputs" / edition


def show(row, custom):
    source = input_root(custom, row["edition"])
    key = row["key"]
    if row["edition"] == "pilot":
        prompt = load(source / "references/prompts.json")["prompts"][key]
    else:
        prompt = designs(source)[key]["prompt"]
    print(f"{row['id']}  {row['label']}\n{prompt}\n\nRGB  {source / 'references' / (key + '.png')}")
    if "published_dir" in row:
        folder = ROOT / "published" / row["published_dir"]
        for filename in ("asset.glb", "asset.ply", "surface.npz", "metadata.json"):
            print(f"     {folder / filename}")


def image_jobs(rows, selected, custom):
    jobs = []
    editions = dict.fromkeys(rows[k]["edition"] for k in selected)
    for edition in editions:
        source = input_root(custom, edition)
        requested = [rows[k]["key"] for k in selected if rows[k]["edition"] == edition]
        if edition == "pilot":
            prompts = load(source / "references/prompts.json")["prompts"]
            spec = load(source / "calibration/layout.json")
            edits = {}
        else:
            spec = designs(source)
            prompts = {k: v["prompt"] for k, v in spec.items()}
            edits_file = source / "calibration/overrides.json"
            edits = load(edits_file) if edits_file.exists() else {}
        for k in required_keys(requested, spec, edits):
            jobs.append(dict(id=spec[k]["id"], prompt=prompts[k],
                             image=str((source / "references" / (k + ".png")).absolute())))
    return jobs


def audit(rows, published):
    files = 0
    fingerprints = {}
    for line in read(ROOT / "catalog/inputs.sha256").splitlines():
        digest, name = line.split("  ", 1)
        if sha(ROOT / name) != digest:
            raise ValueError(f"归档输入不匹配：{name}")
        fingerprints[name] = digest
    jobs = image_jobs(rows, list(rows), None)
    if {j["id"] for j in jobs} != {r["id"] for r in rows.values()}:
        raise ValueError("生图任务与资产清单不匹配")
    for row in rows.values():
        rgb = Path("inputs") / row["edition"] / "references" / (row["key"] + ".png")
        if fingerprints[str(rgb)] != row["source_sha256"]:
            raise ValueError(f"原图不匹配：{rgb}")
        if published:
            base = ROOT / "published" / row["published_dir"]
            for name, digest in (("asset.glb", "glb_sha256"), ("asset.ply", "ply_sha256")):
                if sha(base / name) != row[digest]:
                    raise ValueError(f"发布文件不匹配：{base / name}")
                files += 1
    print(f"{len(fingerprints)} 个归档输入的 SHA-256 通过核验，覆盖 {len(rows)} 张资产主图")
    if published:
        print(f"{files} 个发布 GLB/PLY 的 SHA-256 通过核验")


def required_keys(requested, specs, overrides):
    # A partly open lid can use the fitted inner wall and photograph of another open shaft.
    keys = list(requested)
    for key in keys:
        if key not in specs:
            raise ValueError(f"缺少参考图的设计或拟合记录：{key}")
        cfg = specs[key]
        wall = cfg.get("wall_source")
        if cfg.get("category") == "raised" and cfg.get("model") != "offset":
            wall = cfg.get("donor", wall)
        wall = overrides.get(key, {}).get("wall_source", wall)
        if wall and wall not in keys:
            keys.append(wall)
    return keys


def run_step(python, script, args, env=None):
    print(f"→ {script.name}  " + " ".join(map(str, args)), flush=True)
    if not Path(python).is_file():
        raise ValueError(f"解释器不存在：{python}，请在 --config 指定的配置里更新路径")
    subprocess.run([python, "-u", script, *map(str, args)], env=env, check=True)


def prepare_inputs(bundle, source, edition, rows, raw):
    cal = bundle / "calibration"
    refs = bundle / "references"
    cal.mkdir(parents=True)
    refs.mkdir()
    original = source / "calibration"
    keys = [r["key"] for r in rows]
    if edition == "series":
        plan = designs(source)
        edits = load(original / "overrides.json") if (original / "overrides.json").is_file() else {}
    else:
        plan, edits = {}, {}
    if raw and edition == "series":
        needed = required_keys(keys, plan, edits)
    else:
        layout = load(original / "layout.json")
        needed = required_keys(keys, layout, {})
        # The pilot SH sampling seed includes the ordinal in the complete, original layout.
        save(cal / "layout.json", layout if edition == "pilot" else {k: layout[k] for k in needed})
    if edition == "series":
        save(cal / "designs.json", [plan[k] for k in needed])
        save(cal / "overrides.json", {k: edits[k] for k in needed if k in edits})
    else:
        shutil.copy2(source / "references/prompts.json", refs)
    for name in ("fracture", "back", "squareback"):
        path = source / "references" / (name + ".png")
        if not path.is_file():
            path = ROOT / "inputs/shared" / (name + ".png")
        shutil.copy2(path, refs)
    pits = []
    for key in needed:
        shutil.copy2(source / "references" / (key + ".png"), refs)
        spec = plan[key] if edition == "series" else layout[key]
        if spec["category"] != "pothole":
            continue
        if spec.get("depth_profile") == "masked_groove":
            shutil.copy2(original / (key + "_outline.png"), cal)
            continue
        if raw and spec.get("depth_profile") == "designed_bowl":
            continue
        pits.append(key)
        if not raw:
            for suffix in ("_depth.npz", "_outline.png"):
                shutil.copy2(original / (key + suffix), cal)
            if "water" in layout[key]:
                shutil.copy2(original / (key + "_water.png"), cal)
    return needed, pits


def build_edition(bundle, edition, rows, args, cfg):
    code = bundle / "rebuild"
    for directory in ("common", edition):
        shutil.copytree(ROOT / "recipes" / directory, code, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    raw = args.from_images or args.inputs is not None
    source = input_root(args.inputs, edition)
    needed, pits = prepare_inputs(bundle, source, edition, rows, raw)
    keys = [r["key"] for r in rows]
    ids = [r["id"] for r in rows]
    record = dict(edition=edition, ids=ids, mesh_only=args.mesh_only, preview=args.preview,
                  calibration="from_images" if raw else "published",
                  published={r["id"]: {k: r[k] for k in ("triangles", "gaussians", "glb_sha256", "ply_sha256")}
                             for r in rows if "glb_sha256" in r},
                  reference_sha256={p.name: sha(p) for p in sorted((bundle / "references").glob("*.png"))})
    for row in rows:
        if "source_sha256" in row and record["reference_sha256"][row["key"] + ".png"] != row["source_sha256"]:
            raise ValueError(f"发布原图的 SHA-256 已变化：{row['id']}。新图请使用 --inputs")
    save(bundle / "run.json", record)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0")
    mesh, compute = cfg["mesh_python"], cfg["compute_python"]
    if raw and pits:
        depth_env = env.copy()
        extra = cfg.get("depth_pythonpath")
        if extra:
            depth_env["PYTHONPATH"] = os.pathsep.join(filter(None, [extra, env.get("PYTHONPATH")]))
        model = args.model or ROOT / "models/depth-anything-v2-small"
        run_step(compute, code / "estimate_depth.py", ["--model", model, *pits], depth_env)
    if raw:
        if edition == "series":
            run_step(compute, code / "fit_images.py", ["--redo", *needed], env)
        elif pits:
            run_step(compute, code / "trace_masks.py", pits, env)
    cache = bundle / "mesh_cache"
    run_step(mesh, code / "build.py", ["--edition", edition, "--cache", cache, *keys], env)
    if not args.mesh_only:
        run_step(compute, code / "export_splats.py", ["--cache", cache, *ids], env)
    if args.preview:
        run_step(mesh, code / "render_mesh.py", ["--cache", cache, "--out", bundle / "views", *ids], env)
    run_step(mesh, code / "verify.py", [], env)


def build(args, rows, cfg):
    if args.all and args.ids:
        raise ValueError("--all 与指定编号只能选择一种")
    selected = select(rows, list(rows) if args.all else args.ids)
    groups = defaultdict(list)
    for key in selected:
        groups[rows[key]["edition"]].append(rows[key])
    dest = args.out.absolute()
    if dest.exists() or dest.is_symlink():
        raise ValueError(f"输出目录已存在，请选一个新的 --out：{dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gen2hole-", dir=dest.parent) as temp:
        stage = Path(temp) / "run"
        for edition, chosen in groups.items():
            build_edition(stage / edition, edition, chosen, args, cfg)
        save(stage / "run.json", dict(editions=list(groups), assets=selected))
        stage.rename(dest)
    print(f"完成  {dest}", flush=True)


def verify(folder, cfg):
    for edition in load(folder / "run.json")["editions"]:
        run_step(cfg["mesh_python"], folder / edition / "rebuild/verify.py", [],
                 dict(os.environ, PYTHONOPTIMIZE="0", PYTHONDONTWRITEBYTECODE="1"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/local.json", help="解释器与道路工程路径")
    commands = p.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("list", help="查看可用编号与类别")
    prompt = commands.add_parser("show", help="查看单件资产的原始生图提示词、原图与成品路径")
    prompt.add_argument("id")
    prompts = commands.add_parser("prompts", help="列出生图任务，每张图附原始提示词与落盘路径")
    prompts.add_argument("ids", nargs="*", help="缺省列出整个批次")
    audit_cmd = commands.add_parser("audit", help="核对项目收录的参考图、提示词和模型")
    audit_cmd.add_argument("--published", action="store_true", help="同时核对 109 件发布模型的 GLB、PLY 字节")
    create = commands.add_parser("build", help="从生成图和配方重建资产")
    for cmd in (catalog, prompt, prompts, create):
        cmd.add_argument("--inputs", type=Path, help="自定义批次，含 calibration/designs.json 和 references/")
    create.add_argument("ids", nargs="*", help="例如 P02 P13 M13 A13")
    create.add_argument("--all", action="store_true", help="构建所选输入库中的全部资产")
    create.add_argument("--out", type=Path, required=True, help="新建的运行目录")
    create.add_argument("--from-images", action="store_true", help="从 RGB 重算相对深度及拟合，使用已记录的人工标记")
    create.add_argument("--mesh-only", action="store_true", help="仅生成带贴图 GLB 和路面切口")
    create.add_argument("--preview", action="store_true", help="从实际 GLB 渲染两个视角")
    create.add_argument("--model", help="Depth Anything V2 Small 本地权重目录或模型 ID")
    check = commands.add_parser("verify", help="重新读取并检查整次构建的产物")
    check.add_argument("--run", type=Path, required=True)
    scan = commands.add_parser("scan", help="导出一个源帧的实测或迁入后 LiDAR 扫描")
    scan.add_argument("--root", type=Path, required=True, help="simulate 输出目录或历史增强包")
    scan.add_argument("--case", required=True)
    scan.add_argument("--frame", type=int, required=True)
    scan.add_argument("--output", required=True, type=Path)
    scan.add_argument("--asset", help="以某个实例为中心裁切")
    scan.add_argument("--margin-m", type=float, default=2)
    scan.add_argument("--mode", choices=["before", "after"], default="after")
    args = p.parse_args()
    try:
        if args.command == "verify":
            verify(args.run, load(args.config))
            return
        if args.command == "audit":
            audit(inventory(None), args.published)
            return
        if args.command == "scan":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            options = ["--root", args.root, "--case", args.case, "--frame", args.frame, "--mode", args.mode,
                       "--margin-m", args.margin_m, "--output", args.output]
            if args.asset:
                options.extend(["--asset", args.asset])
            run_step(load(args.config)["mesh_python"], ROOT / "downstream/sensors/replay.py", options)
            return
        rows = inventory(args.inputs)
        if args.command == "list":
            for key, row in rows.items():
                print(f"{key:5} {row['edition']:7} {row['category']:8} {row['label']}")
        elif args.command == "show":
            show(rows[select(rows, [args.id])[0]], args.inputs)
        elif args.command == "prompts":
            print(json.dumps(image_jobs(rows, select(rows, args.ids or list(rows)), args.inputs),
                             ensure_ascii=False, indent=2))
        else:
            build(args, rows, load(args.config))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"gen2hole: {exc}\n")


if __name__ == "__main__":
    main()
