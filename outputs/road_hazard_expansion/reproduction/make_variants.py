"""Build GLB meshes and local road cutouts from explicit road-hazard geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.spatial.transform import Rotation
import trimesh

from variant_catalog import PAVEMENTS, variants


TEXTURE_SIZE = 512
CUTOUT_SIZE = 257
SURFACE_SOURCES = ["Asphalt031", "Concrete034", "Concrete049", "ManholeCover009"]


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def faces_grid(rows, columns):
    cells = np.arange(rows * columns).reshape(rows, columns)
    a, b, c, d = cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]
    return np.stack([np.stack([a, c, b], -1), np.stack([c, d, b], -1)], -2).reshape(-1, 3)


@lru_cache(maxsize=15)
def read_map(root, material, channel, size=TEXTURE_SIZE):
    path = next((root / material).glob(f"*_{channel}.png"))
    with Image.open(path) as raw:
        im = raw.convert("RGB") if channel == "Color" else raw.convert("F")
        array = np.array(im.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
    if channel == "Color":
        return array / 255
    low, high = np.percentile(array, [1, 99])
    return np.clip((array - low) / max(high-low, 1e-4), 0, 1)


def rgb_image(rgb, quality=88):
    data = BytesIO()
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + .5)).save(data, format="JPEG", quality=quality, subsampling=0)
    data.seek(0)
    return Image.open(data)


class Asset:
    def __init__(self, name, cache):
        self.id, self.cache = name, cache
        self.parts, self.textures, self.ranges = [], {}, {}

    def texture(self, name, rgb, roughness=.92, metallic=0.):
        # Color is sampled independently from texture encoding for Gaussian export.
        self.textures[name] = (np.float32(np.clip(rgb, 0, 1)), roughness, metallic)

    def add(self, label, vertices, faces, uv, texture, road=0):
        vertices = np.asarray(vertices, dtype="f4").reshape(-1, 3)
        faces = np.asarray(faces, dtype="i4").reshape(-1, 3)
        uv = np.asarray(uv, dtype="f4").reshape(-1, 2)
        if len(faces) == 0:
            return
        used, indices = np.unique(faces, return_inverse=True)
        self.parts.append((label, vertices[used], indices.reshape(-1, 3).astype("i4"), uv[used], texture, road))

    def save(self, out, cfg, field):
        scene = trimesh.Scene()
        texture_keys = tuple(self.textures)
        materials = {}
        for key, (rgb, roughness, metallic) in self.textures.items():
            materials[key] = trimesh.visual.material.PBRMaterial(
                name=key, baseColorTexture=rgb_image(rgb), roughnessFactor=roughness,
                metallicFactor=metallic, doubleSided=False
            )
        blocks, vertices, faces, coordinates, material_ids, road_weights = {}, [], [], [], [], []
        cursor = 0
        for label, points, index, uv, key, road in self.parts:
            world = points[:, [0, 2, 1]] * [1, 1, -1]
            model = trimesh.Trimesh(vertices=world, faces=index, process=False,
                                    visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=materials[key]))
            scene.add_geometry(model, node_name=label, geom_name=label)
            blocks[label] = int(len(index))
            vertices.append(points)
            faces.append(index + cursor)
            coordinates.append(uv)
            material_ids.append(np.full(len(index), texture_keys.index(key), dtype="u1"))
            road_weights.append(np.full(len(index), road, dtype="f4"))
            cursor += len(points)

        out.mkdir(parents=True, exist_ok=True)
        mesh_file = out / "asset.glb"
        mesh_file.write_bytes(scene.export(file_type="glb"))
        np.savez_compressed(out / "surface.npz", **field)
        self.cache.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache / (self.id + ".npz"), xyz=np.concatenate(vertices), uv=np.concatenate(coordinates),
                 faces=np.concatenate(faces), material=np.concatenate(material_ids), road=np.concatenate(road_weights),
                 parameters=np.asarray([self.textures[k][1:] for k in texture_keys], dtype="f4"),
                 **{f"texture_{i}": np.uint8(np.clip(self.textures[k][0], 0, 1)*255+.5)
                    for i, k in enumerate(texture_keys)})
        cfg.update(units="metres", origin="hazard center at the local road plane", geometry_source="procedural_3d",
                   mesh_name=mesh_file.name, geometry_parts=blocks, material_sources=SURFACE_SOURCES,
                   glb_up="Y", gaussian_up="Z")
        (out / "metadata.json").write_text(json.dumps(cfg, indent=2) + "\n")
        return len(mesh_file.read_bytes()), sum(blocks.values())


def axes(extent, size):
    coords = np.linspace(-extent, extent, size, dtype="f4")
    return np.meshgrid(coords, -coords)


def resample_field(values, width=TEXTURE_SIZE):
    b = values.shape[0]
    y, x = np.meshgrid(np.linspace(0, b-1, width), np.linspace(0, b-1, width), indexing="ij")
    return map_coordinates(values, (y, x), order=1)


def environment(root, cfg, extent):
    source, tint, strength = PAVEMENTS[cfg["material"]]
    rng = np.random.default_rng(cfg["seed"])
    offset = tuple(rng.integers(0, TEXTURE_SIZE, 2))
    base = np.roll(read_map(root, source, "Color"), offset, axis=(0, 1)) * np.array(tint, dtype="f4")
    grit = np.roll(read_map(root, "Concrete049", "Color"), offset[::-1], axis=(0, 1))
    if cfg["material"] == "aged_asphalt":
        base = .88*base + .12*grit
    m = np.roll(read_map(root, source, "Displacement", CUTOUT_SIZE),
                tuple(a//2 for a in offset), axis=(0, 1))
    x, y = axes(extent, CUTOUT_SIZE)
    low = gaussian_filter(rng.normal(0, 1, x.shape), 3.2)
    h = ((m-gaussian_filter(m, 4))*.8 + low*.52)*strength
    h *= smoothstep((extent - np.hypot(x, y)) / .17)
    return base, grit, x, y, h, rng


def outline(x, y, rx, ry, rng, lean=0., rugged=.15):
    p, q = np.cos(lean)*x + np.sin(lean)*y, -np.sin(lean)*x + np.cos(lean)*y
    theta = np.arctan2(q/ry, p/rx)
    side = rng.uniform(-np.pi, np.pi, 4)
    warp = 1 + rugged*(.52*np.sin(3*theta+side[0]) + .30*np.sin(6*theta+side[1]) +
                       .19*np.sin(13*theta+side[2]) + .10*np.sin(37*theta+side[3]))
    return 1-np.hypot(p/rx, q/ry)/warp


def pit_fields(cfg, x, y, rng):
    rx, ry = cfg["width_m"]*.5, cfg["length_m"]*.5
    z = cfg["depth_design_m"]
    family = cfg["family"]
    lean = rng.uniform(-.13, .14) if family == "wheel_rut" else rng.uniform(-.49, .46)
    if family in {"joined", "cluster"}:
        a = outline(x+.36*rx, y+.11*ry, .62*rx, .74*ry, rng, lean)
        b = outline(x-.36*rx, y-.16*ry, .66*rx, .68*ry, rng, lean+.45)
        s = np.maximum(a, b)
        if family == "cluster":
            s = np.maximum(s, outline(x+.23*rx, y-.65*ry, .40*rx, .41*ry, rng, lean))
    elif family in {"patch_break", "slab_corner"}:
        theta = np.arctan2(y/ry, x/rx)
        edge = 1 + .085*np.cos(17*theta) + .045*np.sin(43*theta+rng.uniform(0, 3))
        u, v = np.cos(lean)*x + np.sin(lean)*y, -np.sin(lean)*x+np.cos(lean)*y
        p = 4 if family == "patch_break" else 6
        s = 1-((abs(u/rx)**p + abs(v/ry)**p)**(1/p))/edge
        if family == "slab_corner":
            s = np.minimum(s, .41 + .65*(x/rx + y/ry) + .075*np.sin(y*62))
    else:
        s = outline(x, y, rx, ry, rng, lean, .16 if family != "spall" else .25)
        if family == "crescent":
            inner = outline(x+.42*rx, y-.41*ry, .86*rx, .66*ry, rng, lean)
            s = np.minimum(s, -inner)
    if family == "bowled":
        h = -z*smoothstep(s/.81)
    elif family in {"spall", "wheel_rut"}:
        h = -z*(.48*smoothstep(s/.20)+.52*smoothstep(s/.66))
    else:
        h = -z*(.38*smoothstep(s/.085)+.62*smoothstep(s/.37))
    pebbles = gaussian_filter(rng.normal(0, 1, x.shape), .9)
    h += np.clip(s/.09, 0, 1)*np.clip(pebbles, -.8, .8)*min(.018, z*.31)
    cracks = np.zeros_like(h)
    for angle in rng.uniform(-np.pi, np.pi, 4):
        along = x*np.cos(angle)+y*np.sin(angle)
        side = y*np.cos(angle)-x*np.sin(angle)-.013*np.sin(along*28+angle)-.005*np.sin(along*92)
        cr = np.exp(-abs(side)/.004)*smoothstep((.32+s)/.28)*smoothstep((-s+.035)/.06)*(along>0)
        cracks = np.maximum(cracks, cr)
    h -= cracks*.005
    return s, h, cracks


def local_field(x, y, height, extent):
    return dict(height=height.astype("f4"),
                alpha=smoothstep((extent-np.hypot(x, y)) / .17).astype("f4"),
                width=np.float32(extent*2), length=np.float32(extent*2))


def build_pit(mesh, cfg, source):
    extent = max(.49, .60*max(cfg["width_m"], cfg["length_m"])+.19)
    pavement, grit, x, y, relief, rng = environment(source, cfg, extent)
    dist, pit, cracks = pit_fields(cfg, x, y, rng)
    height = relief + pit
    pit_weight = resample_field(smoothstep((dist+.025)/.18))
    hue = np.median(pavement, axis=(0, 1))
    fracture = (.62*grit + .38*pavement) * (hue/np.median(.62*grit+.38*pavement, axis=(0, 1))) * .79
    band = resample_field(smoothstep((dist+.025)/.088)*smoothstep((.25-dist)/.15))
    inner = resample_field(smoothstep(-pit/max(.011, cfg["depth_design_m"])))
    tex = (pavement*(1-pit_weight[..., None]) + fracture*pit_weight[..., None])
    tex *= (1-.22*inner-.37*resample_field(cracks))[:, :, None]
    tex += (.024*band)[:, :, None] * np.array([1, .86, .68])
    if cfg["family"] == "patch_break":
        patch = (np.abs(x) < .60*cfg["width_m"]) & (np.abs(y) < .64*cfg["length_m"])
        tex *= (1-.14*resample_field(patch.astype("f4"))*(1-pit_weight))[:, :, None]
    mesh.texture("Road and fractured aggregate", tex)
    resolution = 151 if extent < 1. else 181
    ys, xs = np.meshgrid(np.linspace(0, CUTOUT_SIZE-1, resolution),
                         np.linspace(0, CUTOUT_SIZE-1, resolution), indexing="ij")
    gx, gy = axes(extent, resolution)
    block = np.stack([gx, gy, map_coordinates(height, [ys, xs], order=1)], -1)
    uv = np.stack([gx/(2*extent)+.5, gy/(2*extent)+.5], -1)
    tri = faces_grid(resolution, resolution)
    mesh.add("Road surface, lip and pit floor", block, tri, uv, "Road and fractured aggregate", 1)
    if cfg["family"] == "waterlogged":
        level = -.54*cfg["depth_design_m"]
        pool = map_coordinates(height, [ys, xs], order=1) < level
        take = np.all(pool.ravel()[tri], axis=1)
        liquid = block.copy()
        liquid[..., 2] = level + .0004*np.sin(gx*41)*np.cos(gy*49)
        water = .11+.075*grit[:, :, :3] + np.array([.016, .029, .024])
        mesh.texture("Muddy water", water, .26, .0)
        mesh.add("Water surface", liquid, tri[take], uv, "Muddy water")
        height = np.maximum(height, np.where(height < level, level, -np.inf))
        cfg["water_level_m"] = level
    cfg["minimum_surface_z_m"] = float(np.min(block[..., 2]))
    return local_field(x, y, height, extent)


def profile_shape(cfg, theta):
    c, s = np.cos(theta), np.sin(theta)
    p = cfg["exponent"]
    return np.stack([cfg["width_m"]*.5*np.sign(c)*np.abs(c)**(2/p),
                     cfg["length_m"]*.5*np.sign(s)*np.abs(s)**(2/p)], -1)


def ring_surface(mesh, label, xy, elevations, texture, rough_map=None, road=0):
    vertices = np.concatenate([xy, np.broadcast_to(elevations, xy.shape[:-1])[..., None]], axis=-1)
    if rough_map is None:
        uv = np.stack(np.meshgrid(np.linspace(0, 1, xy.shape[1]), np.linspace(0, 1, xy.shape[0])), axis=-1)
    else:
        uv = rough_map
    mesh.add(label, vertices, faces_grid(*xy.shape[:2]), uv, texture, road)


def plane_uv(xy, extent):
    return xy / (2*extent) + .5


def disk(mesh, label, profile, theta, z, texture, radius_factor=1, transform=None, reverse=False):
    r = np.linspace(.012, radius_factor, 26)
    xy = r[:, None, None]*profile
    uv = xy / (2*np.max(np.abs(profile), axis=0)*radius_factor)*.94 + .5
    points = np.concatenate([xy, np.broadcast_to(z, xy.shape[:-1])[..., None]], axis=-1)
    center = np.array([[0., 0., float(np.asarray(z).mean())]])
    center_uv = np.array([[.5, .5]])
    faces = faces_grid(*points.shape[:2])
    next_id = np.arange(len(theta)-1)
    faces = np.r_[faces, np.c_[np.full(len(next_id), points.shape[0]*points.shape[1]), next_id, next_id+1]]
    if reverse:
        faces = faces[:, ::-1]
    vertices = np.r_[points.reshape(-1, 3), center]
    if transform:
        vertices = transform(vertices)
    mesh.add(label, vertices, faces, np.r_[uv.reshape(-1, 2), center_uv], texture)


def cast_face(root, cfg):
    x, y = axes(1, TEXTURE_SIZE)
    tx = read_map(root, "ManholeCover009", "Color")
    a = np.hypot(x, y)
    rusty = np.roll(tx, (166, 119), axis=(0, 1))
    grit = read_map(root, "Concrete049", "Color")
    base = .22*grit + .12*rusty + np.array([.15, .152, .15])
    phi = np.arctan2(y, x)
    p = cfg["exponent"]
    rr = (abs(x)**p+abs(y)**p)**(1/p)
    pattern = cfg["seed"] % 4
    if pattern == 0:
        bars = .5+.5*np.sin(69*(x+y)) * np.sin(69*(x-y))
        actual = np.clip((bars-.62)/.36, 0, 1)
    elif pattern == 1:
        web = np.sin(64*x)**2 + np.sin(63*y)**2
        actual = smoothstep((.38-web)/.34)
    elif pattern == 2:
        web = np.abs(np.sin(18*phi)) + np.abs(np.sin(31*a))
        actual = smoothstep((.86-web)/.75)
    else:
        web = np.sin(48*x)**2 + np.sin(29*y)**2
        actual = smoothstep((.63-web)/.44)
    actual *= smoothstep((.84-rr)/.08)*smoothstep((rr-.18)/.09)
    edge = np.exp(-((rr-.89)/.012)**2)*.12
    return base + (.068*actual+edge)[..., None] * np.array([1, .98, .94])


def box(mesh, label, center, dimensions, tex, transform=None):
    corner = np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    points = corner * dimensions + center
    if transform:
        points = transform(points)
    tri = np.array([[0,2,4],[2,6,4],[1,5,3],[3,5,7],
                    [0,4,1],[4,5,1],[2,3,6],[3,7,6],
                    [0,1,2],[1,3,2],[4,6,5],[5,6,7]])
    uv = (corner[:, [0, 1]] + .5)*.8+.1
    mesh.add(label, points, tri, uv, tex)


def build_manhole(mesh, cfg, source):
    extent = 1.51 if cfg.get("state") == "removed_side" else 1.11
    road, grit, x, y, relief, rng = environment(source, cfg, extent)
    phase = rng.uniform(-np.pi, np.pi)
    theta = np.linspace(0, 2*np.pi, 177)
    base_profile = profile_shape(cfg, theta)
    profile = base_profile*(1+.004*np.sin(31*theta+phase))[:, None]
    depth = cfg["depth_design_m"]
    state = cfg["state"]
    tilt = -.043*np.cos(theta+phase) if state == "settled" else 0.
    damage = np.zeros_like(theta)
    if state in {"broken_flange", "masonry"}:
        damage = (np.exp(-8*(1-np.cos(theta-phase))) + .47*np.exp(-17*(1-np.cos(theta-phase-2.5))))
    outer_profile = profile*(1.25-.10*damage + .012*np.sin(37*theta)*damage)[:, None]
    seam_height = -.003 - .027*damage + tilt
    radial = np.linspace(0, 1, 105)
    outer = (extent*.985)*np.stack([np.cos(theta), np.sin(theta)], -1)
    xy = (1-radial[:, None, None])*outer_profile+radial[:, None, None]*outer
    h = map_coordinates(relief, [(extent-xy[..., 1])/(2*extent)*(CUTOUT_SIZE-1),
                                 (xy[..., 0]+extent)/(2*extent)*(CUTOUT_SIZE-1)], order=1)
    taper = 1-smoothstep(radial/.21)
    h = h*(1-taper[:, None]) + taper[:, None]*seam_height
    x_tex, y_tex = axes(extent, TEXTURE_SIZE)
    p = cfg["exponent"]
    rn = ((abs(x_tex/(cfg["width_m"]*.5))**p)+(abs(y_tex/(cfg["length_m"]*.5))**p))**(1/p)
    collar = smoothstep((1.54-rn)/.29)*smoothstep((rn-1.04)/.17)
    pavement = road*(1-.17*collar[..., None]) + collar[..., None]*(.065*grit + np.array([.014,.010,.007]))
    mesh.texture("Road and collar bedding", pavement)
    ring_surface(mesh, "Road around the aperture", xy, h, "Road and collar bedding", plane_uv(xy, extent), 1)

    oxidized = .35*grit + .13*np.roll(read_map(source, "ManholeCover009", "Color"), (95, 84), axis=(0, 1))
    oxidized += np.array([.082, .079, .074])
    oxidized *= 1 + .14*np.sin(np.arange(TEXTURE_SIZE)[:, None, None]*.032+phase)
    mesh.texture("Oxidized frame and underside", oxidized, .81, .32)
    r = np.array([1., .995, 1.012, 1.031, 1.067, 1.17, 1.225, 1.25])[:, None]
    heights = np.array([-.107, -.033, -.018, -.013, .002, .008, .003, -.003])[:, None]
    damage_strength = np.linspace(.004, .027, len(heights))[:, None]*damage
    ring_r = r - np.linspace(0, .10, len(heights))[:, None]*damage
    ring_r += np.linspace(0, .012, len(heights))[:, None]*np.sin(37*theta)*damage
    ring_surface(mesh, "Cast ring and cover seat", ring_r[..., None]*profile,
                 heights-damage_strength+tilt, "Oxidized frame and underside")

    # Concrete or staggered brick seams on the interior wall. The wall is a real inward-facing surface.
    wall = read_map(source, "Concrete034", "Color")*.61
    zs = np.linspace(-depth, -.101, 115)
    major = ((zs+depth)*(.90 if state != "masonry" else 1.10))
    if state == "masonry" or state == "broken_flange" and cfg["seed"] % 2:
        a, b = axes(1, TEXTURE_SIZE)
        row = np.floor((b+1)*3.5)
        longitudinal = (a+1)*6 + .5*(row % 2)
        mortar_mask = (longitudinal % 1 < .07) | ((b+1)*3.5 % 1 < .075)
        brick = wall*np.array([1.07, .53, .38])*.86 + .09*grit
        wall = np.where(mortar_mask[..., None], .50*wall+np.array([.12,.108,.091]), brick)
        wall_r = 1.008 + .014*np.exp(-((major*7 % 1)/.11)**2)
    else:
        wall_r = 1.009 + .013*np.exp(-((major*2 % 1)/.055)**2)
    wall_t = np.linspace(.24, .83, TEXTURE_SIZE)[:, None, None]
    mesh.texture("Shaft masonry" if state == "masonry" else "Shaft wall", wall*wall_t)
    wall_name = "Shaft masonry" if state == "masonry" else "Shaft wall"
    wall_z = zs[:, None] + np.linspace(0, 1, len(zs))[:, None]*tilt
    ring_surface(mesh, "Inspection shaft", wall_r[:, None, None]*profile, wall_z, wall_name)
    sediment = .27*grit*np.array([.88, .92, .87])
    mesh.texture("Damp base", sediment)
    disk(mesh, "Shaft bottom", base_profile, theta, -depth, "Damp base", 1.018)
    for i, z in enumerate(np.arange(-.25, -depth+.10, -.27)):
        box(mesh, f"Access rung {i+1}", [0, .48*cfg["length_m"], z], [.25, .10, .025], "Oxidized frame and underside")
    if cfg["category"] == "raised":
        angle, dx, dy, yaw = cfg["cover_pose"]
        rz = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        rx = Rotation.from_euler("x", -angle, degrees=True).as_matrix()
        pivot = np.array([0., .525*cfg["length_m"], .015])
        translate = np.array([dx, dy, .021 if dx else 0.])
        def rotate_lid(points):
            return (points-pivot) @ rx.T @ rz.T + pivot + translate
        mesh.texture("Patterned lid face", cast_face(source, cfg), .74, .35)
        disk(mesh, "Lid face", base_profile, theta, .028, "Patterned lid face", 1.036, rotate_lid)
        disk(mesh, "Lid underside", base_profile, theta, -.016, "Oxidized frame and underside", 1.036, rotate_lid, True)
        side = np.stack([base_profile*1.036, base_profile*1.036])
        side_uv = np.stack(np.meshgrid(np.linspace(0, 1, side.shape[1]), [0., 1.]), axis=-1)
        side_points = np.concatenate([side, np.broadcast_to(np.array([[.028],[-.016]]), side.shape[:-1])[..., None]], axis=-1)
        mesh.add("Lid thickness", rotate_lid(side_points.reshape(-1, 3)), faces_grid(*side.shape[:2]),
                 side_uv, "Oxidized frame and underside")
        for i, position in enumerate(np.linspace(-.31, .31, 4)):
            span = ((1-abs(position*2)**p)**(1/p))*.82*cfg["length_m"]
            box(mesh, f"Lid rib {i+1}", [position*cfg["width_m"], 0, -.0215],
                [.022, span, .028], "Oxidized frame and underside", rotate_lid)
    path = ((abs(x/(cfg["width_m"]*.5))**p)+(abs(y/(cfg["length_m"]*.5))**p))**(1/p)
    height = np.where(path<.995, -depth, relief)
    return local_field(x, y, height, extent)


def build(job):
    cfg, root, out, cache = job
    name = cfg["id"]
    dest = out / cfg["category"] / name
    if (dest / "metadata.json").is_file() and (cache / (name + ".npz")).is_file():
        return name, "cached"
    asset = Asset(name, cache)
    if cfg["category"] == "pothole":
        footprint = build_pit(asset, cfg, root)
    else:
        footprint = build_manhole(asset, cfg, root)
    size, triangles = asset.save(dest, cfg, footprint)
    return name, size, triangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbr-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--pit-variants", type=int, default=3)
    parser.add_argument("--asset", action="append", help="Build specific IDs.")
    args = parser.parse_args()
    catalog = list(variants(args.pit_variants))
    if args.asset:
        catalog = [cfg for cfg in catalog if cfg["id"] in args.asset]
        if len(catalog) != len(set(args.asset)):
            parser.error("One or more asset IDs are unknown.")
    jobs = [(c, args.pbr_source, args.out, args.cache) for c in catalog]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for output in pool.map(build, jobs):
            print(*output, flush=True)


if __name__ == "__main__":
    main()
