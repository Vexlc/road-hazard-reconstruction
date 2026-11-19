#!/usr/bin/env python3
"""Restore the 15 published asset volumes to the expected project layout."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume_dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / "outputs/road_hazard_expansion"
    volumes = json.loads((args.volume_dir / "volumes.json").read_text())
    for volume in volumes:
        archive_path = args.volume_dir / volume["file"]
        if sha256(archive_path) != volume["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {archive_path}")
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if member.is_dir() or relative.is_absolute() or ".." in relative.parts:
                    continue
                if len(relative.parts) == 1:
                    target = dest / "volume-notices" / archive_path.stem / relative.name
                else:
                    target = dest.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
        print(f"Restored {archive_path.name}", flush=True)


if __name__ == "__main__":
    main()
