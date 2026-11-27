"""Verify a rebuild and package the model and synthetic sensor replay separately."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from camera import HERE

sys.path.insert(0, str(HERE.parent / "lane-damage"))
from pack import archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(build, out):
    fingerprints = json.loads((HERE / "validation/rebuild.json").read_text())
    for name, expected in fingerprints.items():
        if digest(build / name) != expected:
            raise ValueError(f"Unexpected rebuilt output {name}")
    for name, expected in json.loads((HERE / "validation/inputs.json").read_text()).items():
        if digest(HERE / "inputs" / name) != expected:
            raise ValueError(f"Unexpected input {name}")
    model_files = {
        "P592/" + Path(n).name: build / n for n in fingerprints if n.startswith("model/")
    }
    replay_files = {n: build / n for n in fingerprints if n.startswith("replay/")}
    for files in [model_files, replay_files]:
        for n in (
            "README.md",
            "survey.json",
            "figures/overview.png",
            "figures/approach.png",
            "validation/rebuild.json",
            "validation/quality.json",
        ):
            files[n] = HERE / n
    out.mkdir(parents=True, exist_ok=True)
    archive(out / "P592.zip", model_files)
    archive(out / "approach-rays.zip", replay_files)
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest(out/n)}  {n}\n" for n in ("P592.zip", "approach-rays.zip"))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("out", type=Path)
    pack(**vars(p.parse_args()))
