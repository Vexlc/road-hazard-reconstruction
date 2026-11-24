"""Rebuild selected metric pits from the private Release input archive and verify the five native outputs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog/common-pit-kit.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(key, row, output):
    folder = ROOT / Path(row["model"]).parent
    expected = row["native_hashes"]
    for name, digest in expected.items():
        if sha(folder / name) != digest:
            raise ValueError(f"{key}: native output differs: {name}")
    checks = json.loads((output / "series/checks.json").read_text())[key]
    if checks["geometry_sha256"] != row["geometry_sha256"] or checks["sha256"] != expected:
        raise ValueError(f"{key}: geometry or build record differs")


def main():
    spec = json.loads(CATALOG.read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path, help=spec["archive"])
    p.add_argument("assets", nargs="*", help="IDs to build, defaults to all 20")
    p.add_argument("--config", type=Path, default=ROOT / "gen2hole/config/local.json")
    args = p.parse_args()
    keys = list(dict.fromkeys(k.upper() for k in args.assets)) or list(spec["assets"])
    unknown = set(keys) - set(spec["assets"])
    if unknown:
        p.error("Unknown IDs: " + ", ".join(sorted(unknown)))
    if sha(args.archive) != spec["archive_sha256"]:
        p.error("Archive SHA-256 does not match this source release")
    scratch = ROOT / "work"
    scratch.mkdir(exist_ok=True)
    with ZipFile(args.archive) as z:
        for key in keys:
            row = spec["assets"][key]
            output = (ROOT / row["model"]).parents[4]
            if not output.exists():
                with TemporaryDirectory(dir=scratch) as stage:
                    inputs = Path(stage)
                    manifest = z.read(key + "/checksums.json")
                    for name, digest in json.loads(manifest).items():
                        path = PurePosixPath(name)
                        if path.is_absolute() or ".." in path.parts:
                            raise ValueError(f"{key}: invalid input path {name}")
                        target = inputs / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(z.read("objects/" + digest))
                        if sha(target) != digest:
                            raise ValueError(f"{key}: corrupt input {name}")
                    (inputs / "checksums.json").write_bytes(manifest)
                    subprocess.run([sys.executable, "-m", "gen2hole.large_pits", "build", "--inputs", str(inputs),
                                    "--config", str(args.config.resolve()), "--out", str(output)], cwd=ROOT, check=True)
            verify(key, row, output)
            print(f"{key}: geometry and five native files verified in {output}", flush=True)


if __name__ == "__main__":
    main()
