"""Build road assets from aligned PothRGBD RGB, uint16 depth and polygon masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from reconstruct_pothole import reconstruct


SOURCE = "https://github.com/ymyurdakul/PData"
PAPER = "https://arxiv.org/abs/2505.04207"
DATA_CARD = "https://www.kaggle.com/datasets/mahyeks/pothrgbd-rgb-and-depth-images-of-potholes"
Z_SCALE = .001
ARCHIVES = None


def init_worker(source):
    global ARCHIVES
    ARCHIVES = tuple(ZipFile(source / n) for n in ("images.zip", "-depths.zip", "labels.zip"))


def measurements(source_bytes, depth_bytes, label_bytes, focal):
    color = np.array(Image.open(BytesIO(source_bytes)).convert("RGB"))
    raw = np.load(BytesIO(depth_bytes), allow_pickle=False)
    if raw.dtype != np.uint16 or raw.shape != color.shape[:2]:
        raise ValueError("Depth representation or resolution mismatch")
    rows, cols = raw.shape
    mask_image = Image.new("L", (cols, rows))
    paint = ImageDraw.Draw(mask_image)
    for line in label_bytes.decode().splitlines():
        polygon = np.array([float(v) for v in line.split()[1:]]).reshape(-1, 2)
        if len(polygon) >= 3:
            paint.polygon([tuple(x) for x in polygon * [cols-1, rows-1]], fill=255)
    mask = np.asarray(mask_image) > 0
    components, count = ndimage.label(mask)
    if not count:
        raise ValueError("Empty polygon mask")
    major = np.bincount(components.ravel())[1:].argmax() + 1
    main = components == major
    r, c = np.where(main)
    if min(r.max()-r.min(), c.max()-c.min()) < 29:
        raise ValueError("Pothole is too small in the source image")
    if main[[0,-1], :].sum() + main[:, [0,-1]].sum() > 12:
        raise ValueError("The source frame cuts off the pothole")
    z = raw.astype("f4") * Z_SCALE
    valid = (z > .24) & (z < 3.8)
    edge_area = ndimage.binary_dilation(main, iterations=18) & ~mask
    if not np.any(edge_area) or np.mean(valid[main]) < .82 or np.mean(valid[edge_area]) < .7:
        raise ValueError("Insufficient usable depth over the pothole or surrounding road")
    # Preserve invalid sensor pixels and smooth the quantization on valid neighbors.
    filled = z.copy()
    if not valid.all():
        missing = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled[~valid] = z[tuple(missing[:, ~valid])]
    filtered = ndimage.median_filter(filled, size=3)
    z[valid] = filtered[valid]
    z[~valid] = 0
    v, u = np.indices((rows, cols), dtype="f4")
    xyz = np.stack([(u-(cols-1)*.5)*z/focal, (v-(rows-1)*.5)*z/focal, z], axis=-1)
    return color, mask, xyz


def build(job):
    image_path, out, scratch, samples, focal = job
    stem = Path(image_path).stem
    stamp = "_".join(stem.split("_")[:2])
    identifier = f"rgbd_pothole_{stamp}"
    final = out / identifier
    meta_file = final / "metadata.json"
    if meta_file.is_file() and (samples / (identifier+".npz")).is_file():
        return json.loads(meta_file.read_text())
    input_names = {"rgb": image_path, "depth": f"depths/{stamp}_depth.npy", "mask": f"labels/{stem}.txt"}
    images, depths, labels = ARCHIVES
    data = {"rgb": images.read(image_path), "depth": depths.read(input_names["depth"]),
            "mask": labels.read(input_names["mask"])}
    try:
        frames = measurements(data["rgb"], data["depth"], data["mask"], focal)
    except ValueError as error:
        return {"id": identifier, "rejection": str(error)}

    temp = scratch / identifier
    temp.mkdir(exist_ok=True, parents=True)
    tmp_samples = scratch / (identifier+".npz")
    metadata = {k: {"filename": input_names[k], "sha256": sha256(data[k]).hexdigest()} for k in data}
    reasons = []
    for border in (.085, .050):
        configuration = SimpleNamespace(output=temp, samples=tmp_samples, metres_per_unit=Z_SCALE,
                                        border=border, spacing_m=.0023, maximum_grid_size=340, tree_workers=1,
                                        input_metadata=metadata, source=f"{SOURCE} / {stamp}")
        try:
            with redirect_stdout(StringIO()):
                detail = reconstruct(configuration, frames)
        except (ValueError, AssertionError) as error:
            reasons.append(str(error))
            continue
        d50, d95, d99 = detail["depth_mm_p50_p95_p99_in_mask"]
        width, length = detail["pothole_extent_xy_m_p0_4_to_p99_6"]
        quality = {
            "depth_measures_a_depression": (d50 > 3 and 8 < d95 < 240 and d99 < 290
                                             and d95 > 2*detail["road_fit_rms_mm"]),
            "pothole_size_range": .042 < min(width, length) and max(width, length) < 2.35,
            "local_road_plane_rms": detail["road_fit_rms_mm"] < 11,
            "depth_support_in_target": detail["stereo_near_sample_coverage_in_mask"] > .89,
            "depth_support_around_target": detail["stereo_near_sample_coverage_patch"] > .72,
        }
        if not all(quality.values()):
            reasons.append(", ".join(k for k, b in quality.items() if not b))
            continue
        result = {
            "id": identifier, "category": "rgbd", "family": stamp, "material": "captured_rgb",
            "source": SOURCE, "source_frame": stamp, "source_data_card": DATA_CARD, "source_paper": PAPER,
            "width_m": width, "length_m": length, "depth_p95_m": d95/1000, "units": "metres",
            "geometry_source": "real_realsense_depth_with_nominal_intrinsics", "mesh_name": "pothole.glb",
            "measured_support_in_mask": detail["stereo_near_sample_coverage_in_mask"],
            "pinhole_nominal_fx_fy_px": focal, "depth_input_metres_per_unit": Z_SCALE,
            "quality_checks": quality,
            "limitations": ["Nominal camera intrinsics. Width, length and depth estimates are approximate.",
                            "The source provides one visible surface and may include multiple frames of a defect."]
        }
        (temp / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
        tmp_samples.replace(samples / f"{identifier}.npz")
        temp.replace(final)
        return result
    shutil.rmtree(temp)
    tmp_samples.unlink(missing_ok=True)
    return {"id": identifier, "rejection": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--focal", type=float, default=462.,
                        help="Nominal focal length in pixels at the original capture resolution.")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.dataset / "images.zip") as images:
        entries = sorted(n for n in images.namelist() if n.lower().endswith(".jpg"))
    # A repeated timestamp refers to the same depth-array member.
    by_timestamp = {}
    for entry in entries:
        stamp = "_".join(Path(entry).stem.split("_")[:2])
        by_timestamp.setdefault(stamp, entry)
    names = list(by_timestamp.values())
    print("INPUT", len(entries), "RGB records,", len(names), "unique matched depth timestamps", flush=True)
    if args.limit:
        ids = np.linspace(0, len(names)-1, args.limit).astype(int)
        names = [names[i] for i in ids]
    jobs = [(name, args.out, args.scratch, args.samples, args.focal) for name in names]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.dataset,)) as pool:
        results = []
        for result in pool.map(build, jobs):
            results.append(result)
            print(result["id"], "REJECT" if "rejection" in result else "OK", flush=True)
    (args.out / "rgbd_screening.json").write_text(json.dumps(results, indent=2) + "\n")
    print("RGBD", sum("rejection" not in r for r in results), "accepted of", len(results), flush=True)


if __name__ == "__main__":
    main()
