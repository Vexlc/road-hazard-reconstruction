"""Produce and cross-check camera crops, two-LiDAR returns and surface grids for placed assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path, help="含 pilot/series 和已迁入场景的运行目录")
    p.add_argument("--out", required=True, type=Path, help="新建传感器输出目录")
    p.add_argument("--config", type=Path, default=HERE.parent / "config/local.json")
    args = p.parse_args()
    root, dest = args.run.resolve(), args.out.absolute()
    if dest.exists() or dest.is_symlink():
        p.error(f"输出目录已存在，请使用新的 --out：{dest}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    editions = json.loads((root / "run.json").read_text(encoding="utf-8"))["editions"]
    cases = [s.parent.name.split("_") for e in editions for s in (root / e / "scenes").glob("*/scene.json")]
    if not cases:
        p.error("此运行目录还没有道路放置，请先运行 downstream/place.py")
    groups = [int(num) for _, num in cases]
    raw, processed = cfg["pandaset_raw"], cfg["pandaset_processed"]
    compute = cfg["compute_python"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0")
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    def run(python, name, *options):
        print(f"→ {name}  " + " ".join(map(str, options)), flush=True)
        subprocess.run([python, "-u", HERE / "sensors" / name, *map(str, options)], env=env, check=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sensors-", dir=dest.parent) as tmp:
        stage = Path(tmp) / "result"
        cache = Path(tmp) / "rays"
        common = ["--parent", root, "--out", stage]
        span = ["--first", min(groups), "--last", max(groups)]
        run(compute, "build_geometry.py", *common)
        for site in sorted({s for s, _ in cases}):
            run(compute, "prepare_scans.py", *common, "--raw", raw, "--cache", cache, "--site", site)
        run(compute, "build_returns.py", *common, "--cache", cache, *span)
        run(compute, "render_pairs.py", *common, "--raw", raw, "--proc", processed, *span)
        surveys = [root / e / 'calibration/survey.json' for e in editions]
        for survey in surveys:
            if survey.is_file():
                spec = json.loads(survey.read_text())
                if isinstance(spec, dict) and any('fragment' in s for s in spec.get('states', [])):
                    run(compute, "render_pairs.py", *common, "--raw", raw, "--proc", processed, *span, "--matched-family")
                    break
        run(compute, "verify_scans.py", *common, "--raw", raw)
        run(cfg["mesh_python"], "verify_mesh.py", *common)
        run(compute, "verify_views.py", *common, "--raw", raw)
        run(compute, "verify_origins.py", "--out", stage, "--raw", raw)
        stage.rename(dest)
    print(f"完成  {dest}", flush=True)


if __name__ == "__main__":
    main()
