"""Image sampling, glTF materials and Z-up local surface interchange."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, uniform_filter
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def source(key):
    with Image.open(ROOT / "references" / (key + ".png")) as raw:
        return np.asarray(raw.convert("RGB")).copy()


def lookup(img, col, row, scale=479., mode="nearest"):
    col, row = np.broadcast_arrays(col, row)
    coords = np.stack((row * ((img.shape[0] - 1) / scale), col * ((img.shape[1] - 1) / scale)))
    if img.ndim == 2:
        return map_coordinates(img, coords, order=1, mode=mode)
    return np.stack([map_coordinates(img[:, :, i], coords, order=1, mode=mode) for i in range(img.shape[2])], -1)


def intact_pavement(img, key, cfg):
    """Quilt measured-size patches drawn only from undamaged parts of the input."""
    n = img.shape[0]
    rows, cols = np.indices((n,n), dtype='f4') * (479. / (n-1))
    if 'outline' in cfg:
        safe = np.asarray(Image.open(ROOT/'calibration'/(key+'_outline.png'))) < 127
    else:
        cx,cy,rx,ry = cfg['outer']
        u,v=(cols-cx)/rx,(rows-cy)/ry
        r = np.hypot(u,v) if cfg['shape']=='round' else np.maximum(abs(u),abs(v))
        safe = r > cfg.get('collar',1.)+.055
        if 'lid' in cfg:
            lid=cfg['lid']
            if 'ellipse' in lid:
                cx,cy,rx,ry=lid['ellipse']
                v=(cy-rows)/ry
                safe &= np.hypot((cols-cx-lid.get('bow_x',0)*v*v)/rx,v)>1.07
            else:
                q=np.asarray(lid['quad'])
                left,right=min(cx-rx,q[:,0].min()),max(cx+rx,q[:,0].max())
                top,bottom=min(cy-ry,q[:,1].min()),max(cy+ry,q[:,1].max())
                safe &= (cols<left-12)|(cols>right+12)|(rows<top-12)|(rows>bottom+12)
    safe[[0,-1],:]=False;safe[:,[0,-1]]=False
    dist=distance_transform_edt(safe)
    size=int(min(116,dist.max()*1.25))//2*2
    if size<30:raise ValueError('Not enough intact pavement in '+key)
    margin=13
    valid=dist>size*.72+2
    if cfg.get('road_fill')=='stationary':
        # Sporadic dark repairs and directional shade are too sparse to repeat across an apron.
        # Select stable patches and preserve the photographed aggregate above the local mean.
        slow=gaussian_filter(img.astype('f4'),(8.,8.,0))
        lum=slow.mean(-1)
        avg=uniform_filter(lum,size)
        rough=np.sqrt(np.maximum(uniform_filter(lum*lum,size)-avg*avg,0))
        target=np.median(slow[valid][::13],axis=0)
        cost=rough+abs(avg-target.mean())*.35
        valid &= cost<=np.percentile(cost[valid],42)
        img=np.uint8(np.clip(target+(img.astype('f4')-slow)*.98+(slow-target)*.12,0,255))
    centers=np.argwhere(valid)
    rng=np.random.default_rng(701+sum(key.encode('utf8')))
    side=1024;canvas=np.zeros((side+size,side+size,3),'u1')
    cross=smooth((np.arange(margin,dtype='f4')+.5)/margin)
    for y0 in range(0,side,size-margin):
        for x0 in range(0,side,size-margin):
            target=canvas[y0:y0+size,x0:x0+size]
            best_error=float('inf');best=None
            for idx in rng.integers(0,len(centers),size=9):
                cy,cx=centers[idx]
                patch=img[cy-size//2:cy+size//2,cx-size//2:cx+size//2]
                if rng.random()>.5:patch=patch[:,::-1]
                if rng.random()>.5:patch=patch[::-1]
                error=0.
                if y0:error+=np.abs(patch[:margin:3,::3].astype('f4')-target[:margin:3,::3]).mean()
                if x0:error+=np.abs(patch[::3,:margin:3].astype('f4')-target[::3,:margin:3]).mean()
                if error<best_error:best_error=error;best=patch
            mix=np.ones((size,size),'f4')
            if y0:mix[:margin]*=cross[:,None]
            if x0:mix[:,:margin]*=cross[None,:]
            target[:]=np.uint8(np.round(target*(1-mix[...,None])+best*mix[...,None]))
    return np.ascontiguousarray(canvas[:side,:side])


def road_atlas(img, x, y, grain, erase=None):
    """Project observed RGB; continue unseen margins with the same road aggregate."""
    edge = np.minimum.reduce([x, 479 - x, y, 479 - y])
    original = smooth(edge / 9.)
    if erase is not None:
        original *= 1 - erase
    photo = lookup(img, x, y)
    k=(img.shape[0]-1)/479
    fill=lookup(grain,x*k,y*k,scale=grain.shape[0]-1,mode='mirror')
    return (photo * original[..., None] + fill * (1 - original[..., None])).astype("u1")


def grid_faces(rows, cols, flip=False):
    ids = np.arange(rows * cols, dtype="i4").reshape(rows, cols)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    faces = np.stack((np.stack([a, c, b], -1), np.stack([c, d, b], -1)), -2).reshape(-1, 3)
    return faces[:, [0, 2, 1]] if flip else faces


def jpeg(array, quality=95):
    stream = BytesIO()
    Image.fromarray(np.uint8(array)).save(stream, format="JPEG", quality=quality, subsampling=0)
    stream.seek(0)
    return Image.open(stream)


def luminance(img):
    return np.einsum("...c,c->...", img.astype("f4"), np.array([.2126, .7152, .0722], "f4")) / 255


def material_maps(rgb, rough, metal, bump):
    lum = luminance(rgb)
    # Fine texture slopes supplement geometry. Low-frequency source shading stays out of the normals.
    detail = gaussian_filter(lum, .65) - gaussian_filter(lum, 2.7)
    dy, dx = np.gradient(detail)
    n = np.stack([-dx * bump, dy * bump, np.ones_like(dx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = np.uint8((.5 + .5 * n) * 255 + .5)
    gr = np.clip(rough - np.clip(detail * .55, -.13, .13), .02, 1.)
    met = np.clip(metal + (lum - np.median(lum)) * min(metal, .35), 0, 1)
    orm = np.uint8(255 * np.stack([np.ones_like(gr), gr, met], -1))
    return normal, orm


class Mesh:
    def __init__(self):
        self.materials = {}
        self.parts = []
        self.lossless_color = set()
        self.normal_maps = {}

    def material(self, key, rgb, rough, metal, bump=3., *, lossless_color=False):
        self.materials[key] = (np.asarray(rgb, dtype="u1"), float(rough), float(metal), float(bump))
        if lossless_color:
            self.lossless_color.add(key)
        else:
            self.lossless_color.discard(key)
        return key

    def add(self, name, xyz, faces, uv, material, road=False):
        faces = np.asarray(faces, dtype="i4").reshape(-1, 3)
        xyz = np.asarray(xyz, dtype="f4").reshape(-1, 3)
        uv = np.asarray(uv, dtype="f4").reshape(-1, 2)
        tri = xyz[faces]
        valid = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1) > 1e-13
        used, inverse = np.unique(faces[valid], return_inverse=True)
        if not len(used):
            raise ValueError("Empty mesh part " + name)
        self.parts.append((name, xyz[used], inverse.reshape(-1, 3).astype("i4"), uv[used], material, road))

    def glb_bytes(self):
        scene = trimesh.Scene()
        mat = {}
        for key, (rgb, rough, metal, bump) in self.materials.items():
            normal, orm = material_maps(rgb, rough, metal, bump)
            if key in self.normal_maps:
                normal = self.normal_maps[key]
            mat[key] = trimesh.visual.material.PBRMaterial(
                name=key, baseColorTexture=Image.fromarray(rgb) if key in self.lossless_color else jpeg(rgb, 96), normalTexture=jpeg(normal),
                metallicRoughnessTexture=jpeg(orm, 92), roughnessFactor=1., metallicFactor=1.,
                doubleSided=False)
        for name, points, indices, coords, key, _ in self.parts:
            oriented = points[:, [0, 2, 1]] * [1, 1, -1]
            mesh = trimesh.Trimesh(oriented, indices, process=False,
                                   visual=trimesh.visual.TextureVisuals(uv=coords, material=mat[key]))
            scene.add_geometry(mesh, node_name=name, geom_name=name)
        return trimesh.exchange.gltf.export_glb(scene, include_normals=True)

    def save(self, dest, cache, meta, field):
        dest.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        (dest / "asset.glb").write_bytes(self.glb_bytes())
        keys = list(self.materials)
        xyz, faces, uv, materials, parts, road = [], [], [], [], [], []
        cursor = 0
        for i, (name, points, indices, coords, key, is_road) in enumerate(self.parts):
            xyz.append(points); uv.append(coords); faces.append(indices + cursor)
            cursor += len(points)
            materials.append(np.full(len(indices), keys.index(key), 'u1'))
            parts.append(np.full(len(indices), i, 'i2'))
            road.append(np.full(len(indices), is_road, 'u1'))
        np.savez_compressed(dest / "surface.npz", **{k:np.asarray(v, dtype="f4") for k,v in field.items()})
        np.savez(cache / (meta['id'] + ".npz"), xyz=np.concatenate(xyz), faces=np.concatenate(faces),
                 uv=np.concatenate(uv), face_part=np.concatenate(parts), part_names=np.array([p[0] for p in self.parts]),
                 material=np.concatenate(materials), road=np.concatenate(road),
                 parameters=np.asarray([self.materials[k][1:3] for k in keys], dtype="f4"),
                 **{f"texture_{i}": self.materials[k][0] for i,k in enumerate(keys)})
        meta.update(geometry_parts={p[0]: len(p[2]) for p in self.parts}, units="metres",
                    glb_up="Y", gaussian_up="Z", origin="aperture or defect centre on local road plane",
                    highest_z_m=float(np.concatenate(xyz)[:,2].max()), lowest_z_m=float(np.concatenate(xyz)[:,2].min()),
                    textures=keys, size_multiplier=1.0)
        write_json(dest / "metadata.json", meta)
