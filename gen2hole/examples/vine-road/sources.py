"""Extract the three reference JPEGs by content hash, or rebuild the dated ground-control grids."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
from scipy.spatial import cKDTree
from camera import S


def verify(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest}")


def photos(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(S["early_sources"]):
        pdf = cache / f"source-{i+1}.pdf"
        if not pdf.is_file():
            with urlopen(s["url"], timeout=90) as response, pdf.with_suffix(".part").open(
                "wb"
            ) as f:
                shutil.copyfileobj(response, f)
            verify(pdf.with_suffix(".part"), s["pdf_sha256"])
            pdf.with_suffix(".part").replace(pdf)
        verify(pdf, s["pdf_sha256"])
        with TemporaryDirectory() as folder:
            page = str(s["page"])
            subprocess.run(
                ["pdfimages", "-f", page, "-l", page, "-j", str(pdf), folder + "/image"], check=True
            )
            digests = {
                hashlib.sha256(p.read_bytes()).hexdigest(): p for p in Path(folder).glob("*.jpg")
            }
            for name, digest in s["photos"].items():
                shutil.copyfile(digests[digest], cache / (name + ".jpg"))


def ground(before, after, out):
    import laspy
    from pyproj import Transformer

    for f, name in [(before, "CA_020_NW_PtCl.laz"), (after, "6VUP610280.laz")]:
        verify(f, S["lidar_file_sha256"][name])
    origin = np.array(S["local_origin_utm6n_z_m"])
    usft = 1200 / 3937
    anchor = (
        np.array(Transformer.from_crs(26906, 26934, always_xy=True).transform(*origin[:2])) / usft
    )
    inverse = Transformer.from_crs(26934, 26906, always_xy=True)
    rows = []
    with laspy.open(before) as reader:
        for p in reader.chunk_iterator(500000):
            keep = (
                (np.asarray(p.classification) == 2)
                & (abs(p.x - anchor[0]) < 290 / usft)
                & (abs(p.y - anchor[1]) < 345 / usft)
            )
            if keep.any():
                x, y = inverse.transform(p.x[keep] * usft, p.y[keep] * usft)
                rows.append(np.c_[x, y, p.z[keep] * usft] - origin)
    early = np.vstack(rows).astype("f4")
    data = laspy.read(after)
    late = np.c_[data.x, data.y, data.z][np.asarray(data.classification) == 2] - origin
    xx, yy = np.meshgrid(np.arange(-95.0, 95.01, 0.4), np.arange(-120.0, 120.01, 0.4))
    q = np.c_[xx.ravel(), yy.ravel()]
    result = dict(xx=xx[0], yy=yy[:, 0])
    for name, pts, k in [("before", early, 6), ("after", late, 7)]:
        dist, ids = cKDTree(pts[:, :2]).query(q, k=k)
        result[name] = np.median(pts[ids, 2], axis=1).reshape(xx.shape)
        result[name + "_dist"] = dist[:, -1].reshape(xx.shape)
    np.savez_compressed(out, **result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    sub.add_parser("photos").add_argument("cache", type=Path)
    g = sub.add_parser("ground")
    for key in ("before", "after", "out"):
        g.add_argument(key, type=Path)
    args = p.parse_args()
    if args.operation == "photos":
        photos(args.cache)
    else:
        ground(args.before, args.after, args.out)
