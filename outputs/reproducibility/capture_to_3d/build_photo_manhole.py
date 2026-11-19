"""Rectify the visible rim or lid from the source photo onto a complete manhole prior."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation


FIGURE = "https://www.mdpi.com/1424-8220/23/5/2676#fig_body_display_sensors-23-02676-f005"
IMAGE_CREDIT = "Dongping Zhang, Xuecheng Yu, Li Yang, Daying Quan, Hongmei Mi and Ke Yan. Data-Augmented Deep Learning Models for Abnormal Road Manhole Cover Detection, Sensors 23(5), 2676 (2023), Fig. 5, CC BY 4.0"
STATES = {
    "missing": {
        "base": "asphalt_missing_intact", "id": "photo_manhole_missing",
        "crop": (2423, 100, 3222, 805), "panel": "D",
        "landmarks": [(50, 321), (344, 95), (727, 350), (439, 569)]
    },
    "raised": {
        "base": "asphalt_cover_raised65", "id": "photo_manhole_raised65",
        "crop": (819, 100, 1615, 740), "panel": "B",
        "landmarks": [(91, 311), (391, 200), (717, 318), (398, 416)]
    }
}


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def inspect_photo(figure, cfg, output):
    crop = figure.crop(cfg["crop"])
    crop.save(output / "source_crop.png")
    points = np.float32(cfg["landmarks"])
    target = np.float32([[0, 512], [512, 0], [1024, 512], [512, 1024]])
    warp = cv2.getPerspectiveTransform(target, points)
    return np.array(crop.convert("RGB")), warp


def rectified_rgb(photo, warp, x, y, radius):
    pixel = np.stack([512 * (x / radius + 1), 512 * (1 - y / radius), np.ones_like(x)], axis=-1)
    sample = pixel @ warp.T
    sample = (sample[..., :2] / sample[..., 2, None]).astype(np.float32)
    return cv2.remap(photo, sample[..., 0], sample[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101).astype(np.float32) / 255


def save_texture(texture, output):
    Image.fromarray(np.uint8(np.clip(texture, 0, 1) * 255 + .5)).save(output)
    return str(output.resolve())


def add_backing_ring(mesh, material, inner, outer, elevation):
    angles = np.linspace(0, 2 * np.pi, 384, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vertices = np.c_[np.concatenate([inner * circle, outer * circle]),
                     np.full(2 * len(angles), elevation)]
    ids = np.arange(len(angles))
    nxt = (ids + 1) % len(angles)
    outer_ids = ids + len(angles)
    faces = np.concatenate([np.stack([ids, outer_ids, nxt], axis=1),
                            np.stack([nxt, outer_ids, nxt + len(angles)], axis=1)])
    mesh["faces"] = np.concatenate([mesh["faces"], faces + len(mesh["vertices"])])
    mesh["vertices"] = np.concatenate([mesh["vertices"], vertices.astype("f4")])
    mesh["uv"] = np.concatenate([mesh["uv"], vertices[:, :2].astype("f4") * 3])
    mesh["face_material"] = np.r_[mesh["face_material"], np.full(len(faces), material, dtype="i4")]


def build(figure, state, library):
    cfg = STATES[state]
    base = library / cfg["base"]
    dest = library / cfg["id"]
    dest.mkdir(parents=True, exist_ok=True)
    photo, warp = inspect_photo(figure, cfg, dest)
    mesh = dict(np.load(base / "mesh.npz"))
    materials = json.loads((base / "materials.json").read_text())
    coordinates = np.linspace(-.5, .5, 1280)
    x, y = np.meshgrid(coordinates, -coordinates)
    r = np.hypot(x, y)

    if state == "missing":
        # Blended annulus projects the photographed road patch and frame edge.
        # The opening and vertical interior retain the existing shaft geometry.
        photo_collar = rectified_rgb(photo, warp, x, y, .402)
        road_id = next(i for i, m in enumerate(materials) if m["name"] == "asphalt")
        asphalt = np.array(Image.open(materials[road_id]["color"]).convert("RGB"))
        texture_coords = .5 + x / 2.2, .5 - y / 2.2
        road = cv2.remap(asphalt, (texture_coords[0] * (asphalt.shape[1] - 1)).astype("f4"),
                        (texture_coords[1] * (asphalt.shape[0] - 1)).astype("f4"),
                        cv2.INTER_LINEAR).astype("f4") / 255
        rgb = photo_collar * np.median(road) / np.median(photo_collar[(r > .425) & (r < .49)])
        weight = smoothstep((.50 - r) / .065) * smoothstep((r - .369) / .028)
        texture = road * (1 - weight[..., None]) + rgb * weight[..., None]
        collar = dict(materials[road_id])
        collar["color"] = save_texture(texture, dest / "photo_collar.png")
        # Preserve the original road UVs and materials outside the photographic patch.
        face_centers = mesh["vertices"][mesh["faces"]].mean(axis=1)
        select = (np.hypot(*face_centers[:, :2].T) < .502) & (face_centers[:, 2] > -.018)
        select &= np.isin(mesh["face_material"], [i for i, m in enumerate(materials)
                                                if m["name"] in ("asphalt", "fracture_asphalt", "iron")])
        # Duplicate the affected triangle vertices because their original neighbors still need road UVs.
        face_uv_vertices = mesh["vertices"][mesh["faces"][select]].reshape(-1, 3)
        replacement = len(mesh["vertices"]) + np.arange(len(face_uv_vertices)).reshape(-1, 3)
        mesh["vertices"] = np.concatenate([mesh["vertices"], face_uv_vertices], axis=0)
        mesh["uv"] = np.concatenate([mesh["uv"], face_uv_vertices[:, :2] + .5])
        mesh["faces"][select] = replacement
        mesh["face_material"][select] = len(materials)
        collar["name"] = "fracture_asphalt"
        collar["normal"] = save_texture(np.full((4, 4, 3), [.5, .5, 1.]), dest / "collar_normal.png")
        materials.append(collar)
    else:
        lid_id = next(i for i, m in enumerate(materials) if m["name"] == "cover")
        x, y = x * .78, y * .78
        projected = rectified_rgb(photo, warp, x, y, .334)
        gray = projected @ [0.2126, 0.7152, 0.0722]
        preserve_color = .22 * projected + .78 * gray[..., None]
        gain = .29 / np.median(gray[(r < .39)])
        body = preserve_color * gain
        rim = smoothstep((.344 - np.hypot(x, y)) / .012)
        color = body * rim[..., None] + np.array([.22, .224, .228]) * (1 - rim[..., None])
        materials[lid_id]["color"] = save_texture(color, dest / "photo_lid.png")
        # Small embossed detail from the photographed pattern, with a modeled thickness and hinges.
        band = ndimage.gaussian_filter(gray, 1.1) - ndimage.gaussian_filter(gray, 9)
        lid_vertex_ids = np.unique(mesh["faces"][mesh["face_material"] == lid_id])
        uv = mesh["uv"][lid_vertex_ids]
        relief = ndimage.map_coordinates(band, [(1 - uv[:, 1]) * (len(x) - 1), uv[:, 0] * (len(x) - 1)],
                                        order=1, mode="nearest")
        local_xy = (uv - .5) * .78
        radius = np.hypot(*local_xy.T)
        top = -.001 + smoothstep((.342 - radius) / .024) * (.005 + np.clip(-relief, -.14, .14) * .023)
        local = np.c_[local_xy, top]
        pivot = np.array([0, .343, .012])
        rotation = Rotation.from_euler("x", -65, degrees=True).as_matrix()
        mesh["vertices"][lid_vertex_ids] = (local - pivot) @ rotation.T + pivot
        materials[lid_id]["normal"] = save_texture(np.full((4, 4, 3), [.5, .5, 1.]), dest / "lid_normal.png")

    # Close construction seams below the exposed steel ring. These darker
    # support ledges sit below the road and keep the sky from leaking through.
    iron_id = next(i for i, m in enumerate(materials) if m["name"] == "iron")
    add_backing_ring(mesh, iron_id, .378, .466, -.023)
    add_backing_ring(mesh, iron_id, .326, .367, -.084)
    np.savez_compressed(dest / "mesh.npz", **mesh)
    with np.load(base / "surface.npz") as source_surface:
        surface = dict(source_surface)
    half = float(surface["width"]) * .5
    axis = np.linspace(-half, half, surface["alpha"].shape[0])
    radius = np.hypot(axis[:, None], axis[None, :])
    surface["alpha"] = smoothstep((.88 - radius) / .20).astype("f4")
    np.savez_compressed(dest / "surface.npz", **surface)
    (dest / "materials.json").write_text(json.dumps(materials, indent=2) + "\n")
    info = {
        "state": state, "source": FIGURE, "image_credit": IMAGE_CREDIT,
        "panel": cfg["panel"], "source_crop_xyxy": cfg["crop"],
        "planar_landmarks_in_crop_xy": cfg["landmarks"], "metric_scale": "modeled",
        "open_well_diameter_m": .652, "shaft_depth_m": 1.05,
        "cover_angle_degrees": 65 if state == "raised" else None,
        "photo_transfer": "visible rim annulus" if state == "missing" else "front of metal cover and small relief",
        "completion": "The shaft, underside, thickness, and final cover pose were modeled. "
                      "The published source photo provides a single view."
    }
    (dest / "capture_metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    print(dest, flush=True)


def weight_collar(samples, state):
    with np.load(samples) as original:
        fields = dict(original)
    radius = np.hypot(*fields["local_xyz"][:, :2].T)
    if state == "missing":
        fields["road_mask"] *= smoothstep((radius - .445) / .055)
    fields["alpha"] *= smoothstep((.88 - radius) / .20)
    valid = fields["alpha"] > .008
    fields = {key: value[valid] for key, value in fields.items()}
    depth = np.maximum(0, -.10 - fields["local_xyz"][:, 2])
    fields["color"][:, :3] *= (.48 + .52 * np.exp(-depth / .29))[:, None]
    fields["photo_appearance"] = np.array(1)
    temporary = samples.with_suffix(".new.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", choices=STATES)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--weight-sampled-collar", type=Path)
    args = parser.parse_args()
    if args.weight_sampled_collar:
        weight_collar(args.weight_sampled_collar, args.state)
    elif args.figure and args.library:
        build(Image.open(args.figure.resolve()), args.state, args.library.resolve())
    else:
        parser.error("Supply --figure and --library or use --weight-sampled-collar after sampling.")
