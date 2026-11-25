"""Compare stereo-only road fits with a held-out survey and independent mesh renders."""

from importlib import import_module

import numpy as np
from PIL import Image

rig = import_module("check-cone-cycles")
files = import_module("check-mei-pits")
read, save, cli = files.read, files.save, rig.cli


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def register(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    target = out / "host.npz"
    cli(
        "from-stereo",
        folder / "primary.npz",
        folder / "secondary.npz",
        folder / "matches.npz",
        target,
        "--primary-calibration",
        folder / "primary.json",
        "--secondary-calibration",
        folder / "secondary.json",
    )
    return target


def fit_render(layout, models, hosts, transform, dest, radius):
    dest.mkdir(exist_ok=True, parents=True)
    points, anchor, colored = (
        dest / k for k in ("road.npy", "anchor.json", "appearance.json")
    )
    cli("road-points", points, *hosts.values(), "--stride", 2)
    cli(
        "fit",
        points,
        anchor,
        "--seed",
        *transform[:3, 3],
        "--heading",
        *transform[:3, 0],
        "--up",
        *transform[:3, 2],
        "--radius",
        radius,
        "--asset",
        layout,
        "--asset-root",
        models,
    )
    cli(
        "calibrate-pavement",
        layout,
        anchor,
        colored,
        *hosts.values(),
        "--asset-root",
        models,
    )
    for shot, host in hosts.items():
        cli("render", layout, colored, host, dest / shot, "--asset-root", models)
    return read(anchor)


def check_road(source, dest, models, examples, pairs):
    report = {}
    for site in ("kitti", "avenue"):
        folder = source / site
        path = register(folder, dest / site)
        host = load(path)
        reference = load(folder / "truth.npz")["road_lidar_range"]
        use = np.isfinite(reference) & ~host["protected"][0]
        error = np.abs(reference[use] - host["host_range"][0][use])
        stats = dict(
            lidar_compared=int(use.sum()),
            lidar_available=int(np.isfinite(reference).sum()),
            abs_error_median_m=float(np.median(error)),
            abs_error_p95_m=float(np.percentile(error, 95)),
            pairs={},
        )
        stats["relative_error_p95"] = float(np.percentile(error / reference[use], 95))
        if len(error) < 100 or stats["relative_error_p95"] > 0.075:
            raise ValueError(
                f"Road depth disagrees with independently projected lidar: {site} {stats}"
            )
        for pair, ref in read(folder / "survey.json").items():
            if pair not in pairs:
                raise ValueError("Unexpected road layout")
            pose = np.eye(4)
            pose[:3, 3] = ref["center_world"]
            pose[:3, :3] = ref["basis_columns"]
            out = dest / site / pair
            anchor = fit_render(
                examples / (pair + ".json"), models, {"front": path}, pose, out, 1.35
            )
            inst = np.array(Image.open(out / "front/instance.png"))
            rgb = np.array(Image.open(out / "front/after.png"))
            keep = host["protected"][0]
            if not np.array_equal(rgb[keep], host["rgb"][keep]) or np.any(inst[keep]):
                raise ValueError("Unmatched road pixels changed")
            results = {}
            for index, (row, measured) in enumerate(
                zip(anchor["instances"], ref["instances"]), 1
            ):
                fitted = row["anchor"]
                original = measured["anchor"]
                pos = np.linalg.norm(
                    np.array(fitted["center_world"]) - original["center_world"]
                )
                slope = np.rad2deg(
                    np.arccos(
                        np.clip(
                            np.array(fitted["basis_columns"])[:, 2]
                            @ np.array(original["basis_columns"])[:, 2],
                            -1,
                            1,
                        )
                    )
                )
                count = int((inst == index).sum())
                results[row["asset"]] = dict(
                    anchor_error_m=float(pos),
                    normal_error_deg=float(slope),
                    pixels=count,
                    supported_cells=int(fitted["footprint_support"]["cells"]),
                )
                if count < 10 or pos > 0.085 or slope > 5:
                    raise ValueError(
                        f"Stereo pavement fit outside surveyed support: {site} {results}"
                    )
            stats["pairs"][pair] = results
        report[site] = stats
    return report


def check_controls(source, dest, models, examples, pairs):
    report = {}
    for pair in pairs:
        hosts = {
            shot: register(source / pair / shot, dest / pair / shot)
            for shot in ("rear", "side")
        }
        ground = np.array(read(source / pair / "rear/ground.json"))
        fit_render(
            examples / (pair + ".json"),
            models,
            hosts,
            ground,
            dest / pair / "fit",
            1.35,
        )
        for shot, path in hosts.items():
            data = load(path)
            truth = load(source / pair / shot / "truth.npz")
            valid = ~data["protected"][0]
            native = truth["native_id"]
            output = dest / pair / "fit" / shot
            inst = np.array(Image.open(output / "instance.png"))
            part = np.array(Image.open(output / "part.png"))
            rgb = np.array(Image.open(output / "after.png"))
            radial = np.load(output / "range.npy")
            frozen = ~valid | (truth["host_id"] == 2)
            if np.any(inst[frozen]) or not np.array_equal(
                rgb[frozen], data["rgb"][frozen]
            ):
                raise ValueError("Foreground or unmatched fisheye samples changed")
            near = valid & (truth["host_range"] < 8)
            err = np.abs(data["host_range"][0][near] - truth["host_range"][near])
            stats = dict(
                host_pixels=int(near.sum()),
                host_p95_m=float(np.percentile(err, 95)),
                accepted_behind_optical_plane=int(
                    (near & (data["rays_camera"][0, ..., 2] < 0)).sum()
                ),
                occluder_pixels=int(((truth["host_id"] == 2) & valid).sum()),
                assets={},
            )
            if (
                stats["host_p95_m"] > 0.012
                or min(stats["occluder_pixels"], stats["accepted_behind_optical_plane"])
                < 10
            ):
                raise ValueError(f"Fisheye depth control failed: {stats}")
            for index, key in enumerate(pair.split("-"), 1):
                expected = native // 32 == index
                detected = inst == index
                overlap = expected & detected & valid
                agree = overlap & (native % 32 == part)
                result = dict(
                    coverage=float(valid[expected].mean()),
                    native_pixels=int(expected.sum()),
                    matched_iou=float(
                        overlap.sum() / ((expected | detected) & valid).sum()
                    ),
                    part_agreement=float(agree.sum() / max(1, overlap.sum())),
                    depth_p95_m=float(
                        np.percentile(
                            abs(truth["native_range"][agree] - radial[agree]), 95
                        )
                    ),
                )
                if (
                    min(
                        result["coverage"],
                        result["matched_iou"],
                        result["part_agreement"],
                    )
                    < 0.88
                    or result["depth_p95_m"] > 0.035
                ):
                    raise ValueError(f"Native insert disagrees: {pair} {shot} {result}")
                stats["assets"][key] = result
            report[f"{pair}/{shot}"] = stats
    return report


def check(source, dest, models, examples, pairs):
    from check_stereo_rays import check_ray_contract

    check_ray_contract()
    road = check_road(source, dest, models, examples, pairs)
    save(dest / "road.json", road)
    controls = check_controls(source, dest, models, examples, pairs)
    save(dest / "check.json", dict(road=road, cycles=controls))
