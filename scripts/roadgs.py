"""Inspect a RoadGS release, verify its files, or seed an isolated development run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT / "catalog/roadgs-releases.json"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(release, root):
    checksums = PROJECT / release["checksums"]
    count = 0
    size = 0
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        count += 1
        size += path.stat().st_size
    if (count, size) != (release["files"], release["bytes"]):
        raise ValueError(f"Unexpected totals for {checksums.name}: {count} files, {size} bytes")
    print(f"SHA-256 OK: {count} files, {size} bytes")


def init_run(name, components, version, release, root):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Use a single directory name inside runs/")
    runs = PROJECT / "runs"
    runs.mkdir(exist_ok=True)
    destination = runs / name
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Run already exists: {destination}")

    # Build off to the side so a missing input or a concurrent run cannot publish
    # a partly initialized workspace. The snapshots include uncommitted source edits.
    with tempfile.TemporaryDirectory(dir=runs) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        hashes = {}
        for component in components:
            package = release["packages"][component]["directory"]
            source = PROJECT / "pipelines" / component
            bundle = stage / package
            shutil.copytree(source, bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    hashes[str(path.relative_to(stage))] = digest(path)
            for relative in release["packages"][component]["inputs"]:
                original = root / package / relative
                target = bundle / relative
                if original.is_dir():
                    shutil.copytree(original, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
        ).strip()
        record = {"release": version, "git_commit": commit, "components": components,
                  "source_sha256": hashes}
        (stage / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        stage.rename(destination)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", help="Release ID from catalog/roadgs-releases.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show the release directory and published counts")
    commands.add_parser("verify", help="Read and check every release file against Git's SHA-256 list")
    init = commands.add_parser("init", help="Copy the editable pipeline and its inputs into runs/NAME")
    init.add_argument("name")
    init.add_argument("--component", choices=["image9", "image100", "sensors"], action="append")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    version = args.release or registry["current"]
    if version not in registry["releases"]:
        parser.error(f"Unknown release: {version}")
    release = registry["releases"][version]
    root = PROJECT / release["directory"]
    try:
        if args.command == "verify":
            verify(release, root)
        elif args.command == "init":
            components = list(dict.fromkeys(args.component or release["packages"]))
            init_run(args.name, components, version, release, root)
        else:
            print(f"{version}  {root}")
            print(json.dumps(release["counts"], ensure_ascii=False, indent=2))
            for name, package in release["packages"].items():
                print(f"{name}: {package['directory']}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
