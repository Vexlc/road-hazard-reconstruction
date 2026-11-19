"""Fetch the exact ambientCG material archives used by the road asset builders."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/road_hazard_sources"


def verify(path, material):
    if not path.is_file() or path.stat().st_size != material["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == material["sha256"]


def download(material):
    name = material["id"]
    archive = SOURCE / material["file"]
    partial = archive.with_suffix(".part")
    for attempt in range(4):
        try:
            if not verify(archive, material):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= material["bytes"]:
                    offset = 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(material["url"], headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with partial.open(mode) as file:
                        for chunk in response.iter_content(1024 * 1024):
                            file.write(chunk)
                if not verify(partial, material):
                    if partial.stat().st_size >= material["bytes"]:
                        partial.unlink()
                    raise ValueError("Downloaded archive size or SHA-256 does not match")
                partial.replace(archive)
            with ZipFile(archive) as zip_file:
                zip_file.extractall(SOURCE / name)
            print(f"Prepared {name}", flush=True)
            return
        except (OSError, ValueError, requests.RequestException) as error:
            if attempt == 3:
                raise RuntimeError(f"Could not prepare {name}") from error
            time.sleep(2 ** attempt)


def main():
    SOURCE.mkdir(parents=True, exist_ok=True)
    manifest = Path(__file__).with_name("material_sources.json")
    materials = json.loads(manifest.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, materials))


if __name__ == "__main__":
    main()
