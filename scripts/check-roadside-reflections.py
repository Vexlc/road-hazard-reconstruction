"""Compare finite mirror rays against independent Cycles glass in held-out fisheye exposures."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

rig = import_module("check-cone-cycles")
from metric_reflection import MetricReflectors
from pavement_water import DirectionalProbe, interface
from prop_lighting import linear, srgb


SURVEYS = (
    "survey-left",
    "survey-right",
    "survey-rear",
    "survey-between",
    "survey-far",
    "survey-side",
    "survey-low",
)


def make_packets(rendered, packets):
    packets.mkdir(parents=True, exist_ok=True)
    for name in (*SURVEYS, "front", "edge", "rear"):
        folder = rendered / name / "host"
        raw, _, _ = rig.packet(folder, axial=name in ("survey-right", "edge"), flat_reference=False)
        ids = np.load(folder / "id.npy")
        raw["reflector_labels"] = np.where(ids >= 2, ids, 0).astype("u1")
        if name.startswith("survey-"):
            yy, xx = np.nonzero(raw["reflector_labels"])
            window = np.s_[max(0, yy.min() - 2) : yy.max() + 3, max(0, xx.min() - 2) : xx.max() + 3]
            raw = {k: (v[window] if v.shape[:2] == ids.shape else v) for k, v in raw.items()}
        np.savez_compressed(packets / (name + ".npz"), **raw)
        if name in ("front", "edge", "rear"):
            image = Image.open(rendered / name / "native/rgb.png").convert("RGB")
            image.save(packets / (name + "-glass.png"))


def check(packets, out):
    out.mkdir(parents=True, exist_ok=True)
    mesh_path = out / "reflectors.npz"
    extra = [arg for name in SURVEYS[1:] for arg in ("--additional-view", packets / (name + ".npz"))]
    rig.cli(
        "reflection-mesh",
        packets / (SURVEYS[0] + ".npz"),
        mesh_path,
        *extra,
        "--stride",
        1,
        "--max-edge-m",
        0.16,
    )
    with np.load(mesh_path) as data:
        mesh = MetricReflectors.from_packet(data)
    stats = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles), "views": {}}
    for name in ("front", "edge", "rear"):
        with np.load(packets / (name + ".npz")) as raw:
            rgb = raw["rgb"]
            pose = raw["camera_to_world"]
            rays = raw["rays_camera"] @ pose[:3, :3].T
            selected = raw["receiver_mask"] & ~raw["protected"] & (rays[..., 2] < -0.06)
            sky = (rays[..., 2] > 0) & ~raw["protected"]
            probe = DirectionalProbe(rays[sky], rgb[sky], 2.0)
        pos = pose[:3, 3]
        incident = rays[selected]
        t = -pos[2] / incident[:, 2]
        xy = pos + t[:, None] * incident
        reflected, _, fraction = interface(incident, np.array([0.0, 0.0, 1.0]))
        distance, color = mesh.trace(xy, reflected)
        native = linear(np.asarray(Image.open(packets / (name + "-glass.png"))))
        expected = native[selected]
        angular, support = probe.sample(reflected)
        angular *= support[:, None]

        def classify(values):
            # Nearest radiance class assigns the antialiased boundary at half coverage.
            refs = np.array([[0.0, 0.0, 0.0], [0.82, 0.035, 0.01], [0.013, 0.72, 0.04]])
            return np.linalg.norm(values[:, None] - refs, axis=2).argmin(1)

        truth = classify(expected / np.maximum(fraction[:, None], 1e-8))
        pred = classify(color)
        other = classify(angular)
        report = {}
        for i, key in [(1, "near_red"), (2, "far_green")]:
            gt, hit, baseline = truth == i, pred == i, other == i
            if not gt.any():
                if hit.sum() > 3:
                    raise ValueError("Reflection appeared without native support")
                continue
            intersection = gt & hit
            iou = float(intersection.sum() / (gt | hit).sum())
            directional_iou = float((gt & baseline).sum() / (gt | baseline).sum())
            interior = np.zeros(selected.shape, bool)
            interior[selected] = intersection
            safe = binary_erosion(interior, iterations=2)[selected]
            rendered = color * fraction[:, None]
            err = float(np.percentile(abs(rendered[safe] - expected[safe]).max(1), 95))
            report[key] = dict(
                native_pixels=int(gt.sum()),
                iou=iou,
                directional_iou=directional_iou,
                p95_linear_rgb_error=err,
                interior_pixels=int(safe.sum()),
                rear_hemisphere_pixels=int((intersection & (incident @ pose[:3, 2] < 0)).sum()),
            )
            print(name, key, report[key], flush=True)
            if iou < 0.90 or err > 0.035 or iou < directional_iou + 0.10:
                raise ValueError(f"Native dielectric disagreement on {name}/{key}")
        result = rgb.copy()
        result[selected] = srgb(color * fraction[:, None])
        Image.fromarray(result).save(out / (name + "-metric.png"))
        result[selected] = srgb(angular * fraction[:, None])
        Image.fromarray(result).save(out / (name + "-angular.png"))
        stats["views"][name] = report
    if (
        not all(stats["views"].values())
        or sum(v["rear_hemisphere_pixels"] for v in stats["views"]["edge"].values()) < 60
    ):
        raise ValueError("Insufficient off-axis or rear reflection coverage")
    rig.save_json(out / "check.json", stats)


def save_plate(packets, out, dest):
    from PIL import ImageDraw, ImageFont, ImageOps

    page = Image.new("RGB", (900, 636), "#20262b")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18, index=2)
    for j, label in enumerate(("Cycles 原生水面", "只按环境方向取色", "按近物实测距离求交")):
        draw.text((j * 300 + 12, 7), label, font=font, fill="white")
    for i, (view, label) in enumerate((("front", "前视"), ("edge", "鱼眼边缘"), ("rear", "后视"))):
        files = [
            packets / (view + "-glass.png"),
            out / (view + "-angular.png"),
            out / (view + "-metric.png"),
        ]
        with Image.open(files[0]) as photo:
            yy, xx = np.nonzero(np.asarray(photo).max(axis=2) > 20)
            box = (
                max(0, xx.min() - 8),
                max(0, yy.min() - 8),
                min(photo.width, xx.max() + 9),
                min(photo.height, yy.max() + 9),
            )
        y = 37 + i * 199
        draw.text((12, y), label, font=font, fill="white")
        for j, path in enumerate(files):
            with Image.open(path) as image:
                crop = ImageOps.contain(image.crop(box), (288, 158), Image.Resampling.LANCZOS)
            page.paste(crop, (j * 300 + (300 - crop.width) // 2, y + 27 + (158 - crop.height) // 2))
    dest.parent.mkdir(parents=True, exist_ok=True)
    page.save(dest, quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packets", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--from-cycles", type=Path)
    p.add_argument("--plate", type=Path, help="Write a labelled image of the three reflection methods")
    a = p.parse_args()
    if a.from_cycles:
        make_packets(a.from_cycles, a.packets)
    check(a.packets, a.out)
    if a.plate:
        save_plate(a.packets, a.out, a.plate)
