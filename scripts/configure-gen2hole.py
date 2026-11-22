"""Write checkout-local interpreter, road and sensor paths for the Gen2Hole entry points."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--mesh-python", type=Path, default=ROOT / "work/envs/assets/bin/python")
    args.add_argument("--compute-python", type=Path, default=ROOT / "work/envs/streetcrafter/bin/python")
    args.add_argument("--output", type=Path, default=ROOT / "gen2hole/config/local.json")
    opts = args.parse_args()
    paths = {
        "mesh_python": opts.mesh_python.expanduser().absolute(),
        "compute_python": opts.compute_python.expanduser().absolute(),
        "depth_pythonpath": ROOT / "work/factory-extra-deps",
        "road_project": ROOT,
        "road_config": ROOT / "outputs/道路病害_稳定生成与迁移_v3/本机运行配置.json",
        "pandaset_raw": ROOT / "tools/multiscene/data/raw",
        "pandaset_processed": ROOT / "tools/multiscene/data/processed",
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf8", dir=opts.output.parent, delete=False) as f:
        json.dump({key: str(path) for key, path in paths.items()}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    Path(f.name).replace(opts.output)
    print(opts.output)


if __name__ == "__main__":
    main()
