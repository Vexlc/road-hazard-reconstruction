"""Restore the two registered utility-cover materials from the pinned upstream archives."""
import argparse
import hashlib
from pathlib import Path
import shutil
from zipfile import ZipFile
from PIL import Image

EXAMPLES = Path(__file__).resolve().parent
ARCHIVES = {
    "001": "f53a9c4e26ac39cb5e5d915c0d02e832d602431cc23e8f3029a98c77e6ef9fd8",
    "002": "018a2dc7cff0bb83127d91dab7979a77d126a11c615de3b4967b21191f205c41",
}


def prepare(archives, target):
    materials = target / "materials"
    materials.mkdir(exist_ok=True, parents=True)
    source = EXAMPLES.parent / "perforated-covers"
    for path in [source / "lining.jpg", source / "split-iron/pavement.jpg"]:
        shutil.copyfile(path, materials / path.name)
    for key, digest in ARCHIVES.items():
        stem = f"ManholeCover{key}_1K-PNG"
        path = archives / (stem + ".zip")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wrong material archive: {path}")
        folder = target / key
        folder.mkdir(exist_ok=True)
        with ZipFile(path) as archive:
            for channel in ("Color", "Roughness", "Metalness", "NormalGL", "Opacity", "Displacement"):
                (folder / (channel + ".png")).write_bytes(archive.read(f"{stem}_{channel}.png"))
        with Image.open(folder / "Color.png") as photo:
            Image.new("L", photo.size).save(folder / "solid.png")
        shutil.copyfile(EXAMPLES / key / "survey.json", folder / "survey.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    prepare(args.archives, args.target)
