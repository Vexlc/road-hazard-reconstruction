"""Show an RGB-D array, its individually reusable GLBs at one scale, and native vehicle views."""

import argparse
from importlib import import_module
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gen2hole/downstream/sensors'))
from road_geometry import glb_triangles, read_json

Plate = import_module('render-pit-survey').Plate


DEPTH_STOPS = np.array([[224, 222, 208], [179, 211, 196], [86, 167, 164], [46, 119, 140], [37, 74, 110]])


def render_part(path, span, rim_z):
    width, height = 580, 280
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    textures = {}
    for _, xyz, faces, uv, image in glb_triangles(path, appearance=True):
        geom = scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(faces))
        textures[geom] = uv[faces], np.asarray(Image.open(BytesIO(image)).convert('RGB'))
    away = np.array([0.24, -0.72, 1.0])
    away /= np.linalg.norm(away)
    right = np.cross([0, 0, 1], away)
    right /= np.linalg.norm(right)
    up = np.cross(away, right)
    y, x = np.mgrid[:height, :width]
    eye = (
        away
        + ((x + 0.5 - width / 2) * span / width)[..., None] * right
        + ((height / 2 - y - 0.5) * span / width)[..., None] * up
    )
    rays = np.concatenate([eye, np.broadcast_to(-away, eye.shape)], axis=2).astype('f4')
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    ids = hit['geometry_ids'].numpy()
    normal = hit['primitive_normals'].numpy()
    face = hit['primitive_ids'].numpy()
    bary = hit['primitive_uvs'].numpy()
    light = np.array([-0.56, 0.65, 0.52])
    light /= np.linalg.norm(light)
    rgb = np.full((height, width, 3), [225, 230, 220], dtype='u1')
    for geom, (uv, tex) in textures.items():
        mask = ids == geom
        blend = bary[mask]
        weight = np.c_[1 - blend.sum(1), blend]
        coord = (uv[face[mask]] * weight[:, :, None]).sum(1)
        coords = (coord * tex.shape[1::-1] - 0.5)[:, ::-1].T
        albedo = np.stack(
            [map_coordinates(tex[:, :, k], coords, order=1, mode='nearest') for k in range(3)], axis=1
        )
        factor = 0.57 + 0.51 * np.maximum(normal[mask] @ light, 0)
        rgb[mask] = np.clip(albedo * factor[:, None], 0, 255).astype('u1')
    visible = np.isfinite(hit['t_hit'].numpy())
    height_z = eye[:, :, 2][visible] - hit['t_hit'].numpy()[visible] * away[2]
    mm = np.clip(1000 * (rim_z - height_z), 0, 24)
    relief = np.full_like(rgb, [225, 230, 220])
    relief[visible] = np.stack([np.interp(mm, np.arange(5) * 6, DEPTH_STOPS[:, k]) for k in range(3)], axis=1)
    return Image.fromarray(rgb), Image.fromarray(relief)


def paste(plate, image, x, y, w, h, fill=False):
    if fill:
        plate.image.paste(ImageOps.fit(image, (w, h), method=Image.Resampling.LANCZOS), (x, y))
    else:
        image = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
        plate.image.paste(image, (x + (w - image.width) // 2, y + (h - image.height) // 2))


def plate(run, output):
    key = 'P148'
    folder = run / 'evaluation' / key / 'components'
    data = read_json(folder / 'index.json')
    measure = read_json(run / 'evaluation/measurements.json')[key]
    (frame,) = measure['heldout']
    stats = measure['heldout'][frame]['exported_glb']
    spacing = np.median(data['along_row_spacing_m']) * 100
    parts = [(k, v) for k, v in data['components'].items() if 'file' in v]
    tiles = Plate(1480, 656)
    tiles.text(26, 18, '单槽网格', 32, True)
    tiles.text(203, 25, '同一比例　·　左为纹理网格，右为相对周围路面的凹深', 20)
    for i, color in enumerate(DEPTH_STOPS):
        tiles.draw.rectangle((870 + i * 72, 18, 941 + i * 72, 34), fill=tuple(color))
        tiles.text(871 + i * 72, 41, str(i * 6), 15)
    tiles.text(1238, 28, '毫米', 17)
    span = 0.65
    for i, (name, comp) in enumerate(parts):
        x, y = 24 + (i % 4) * 362, 76 + (i // 4) * 190
        image, relief = render_part(folder / comp['file'], span, comp['rim_elevation_m'])
        image.save(folder / (name + '.webp'), quality=95)
        for j, im in enumerate((image, relief)):
            paste(tiles, im.crop((121, 0, 459, 280)), x + 172 * j, y, 169, 145)

        tiles.text(x + 20, y + 150, name, 19, True)
        tiles.text(x + 112, y + 150, f'{comp["length_m"] * 100:.1f} × {comp["width_m"] * 100:.1f} cm', 18)
    tiles.save(folder / 'gallery.jpg')
    p = Plate(1480, 1538)
    p.text(25, 22, '双排槽状凹痕', 42, True)
    p.text(322, 39, 'P148　/　PothRGBD 实地采集', 23)
    p.text(
        26,
        88,
        f'一帧 RGB-D 恢复 13 个槽口，同排相邻中心的距离中位数为 {spacing:.1f} 厘米。另一帧提供独立的轮廓与深度。',
        22,
    )
    for x, title, path in [
        (24, '拟合照片与槽口编号', folder / 'source.jpg'),
        (510, '整组纹理网格 · 斜视', run / 'series/views/P148_inspection.jpg'),
        (995, '另一机位 · 原始标注与网格回投', folder / (frame + '.jpg')),
    ]:
        p.text(x, 148, title, 22, True)
        with Image.open(path) as im:
            paste(p, im, x, 188, 460, 310)
    p.text(28, 510, '两排分别有 7 个和 6 个槽口。近排最左端贴近画边，保留在整组内。', 18)
    p.text(
        835,
        510,
        f'黄线为标注，青线为回投。重合像素深度中位误差 {stats["depth_abs_median_mm"]:.1f} 毫米。',
        16,
    )
    p.image.paste(tiles.image, (0, 556))
    p.text(26, 1222, '同一整组资产迁入实拍道路', 28, True)
    for i, site in enumerate(['034', '109']):
        x = 27 + 730 * i
        camera = run / 'sensors/camera' / (site + '_148')
        view = next(
            s for s in read_json(camera / 'views.json') if s['available'] and s['camera'] == 'front_camera'
        )
        p.text(x, 1272, f'PandaSet {site}　·　前视帧 {view["frame"]:03}　·　迁入后', 18)
        with Image.open(camera / view['after_stem']) as im:
            paste(p, im, x, 1302, 395, 210)
            a, b, c, d = view['xyxy_target_in_crop']
            crop = im.crop((max(0, a - 18), max(0, b - 22), min(im.width, c + 18), min(im.height, d + 22)))
            paste(p, crop, x + 405, 1324, 297, 130)
            p.text(x + 406, 1470, '同帧槽口局部', 16)
    p.save(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plate(a.run, a.out)
