"""Retint one repair per road and raycast its actual GLB into matched PandaSet frames."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'gen2hole'), str(ROOT/'gen2hole/downstream/sensors')]
from camera_geometry import camera_hits, foreground, track
from repair_patch import export, retint_splats, write
from road_geometry import Surface, glb_triangles, sha256
from sunlight import lighting


def load_glb(path):
    (_, vertices, faces, _, blob), = glb_triangles(path, appearance=True)
    digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
    return vertices, digest, np.array(Image.open(BytesIO(blob)).convert('RGB'))


def project(vertices, pose, intrinsics, anchor):
    basis = np.asarray(anchor['basis_columns'])
    camera = (vertices @ basis.T + anchor['center_world'] - pose[:3, 3]) @ pose[:3, :3]
    if camera[:, 2].min() <= 0:
        raise ValueError('Asset behind camera')
    return camera[:, :2] / camera[:, 2:] * intrinsics.diagonal()[:2] + intrinsics[:2, 2]


def median_road(pixels):
    luminance = pixels @ [.2126, .7152, .0722]
    low, high = np.percentile(luminance, [15, 80])
    return np.median(pixels[(luminance >= low) & (luminance <= high)], axis=0)


def capture(raw, proc, site, anchor, frame, vertices, model):
    name = 'front_camera'
    _, poses, intrinsics, size = track(str(raw), site, name)
    pose = poses[frame]
    coords = project(vertices, pose, intrinsics, anchor)
    left, top = np.floor(coords.min(0)).astype(int) - 35
    right, bottom = np.ceil(coords.max(0)).astype(int) + 36
    if left < 0 or top < 0 or right > size[0] or bottom > size[1]:
        raise ValueError('Choose a complete in-frame placement')
    rect = (left, top, right, bottom)
    path = raw/site/'camera'/name/f'{frame:02d}.jpg'
    photo = np.array(Image.open(path).convert('RGB'))
    crop = photo[top:bottom, left:right]
    depth, part, _, normals = camera_hits(pose, intrinsics, rect, [model])
    rows, cols = np.indices(depth.shape, dtype='f4')
    rays = np.stack([(cols+left+.5-intrinsics[0, 2])/intrinsics[0, 0],
                     (rows+top+.5-intrinsics[1, 2])/intrinsics[1, 1], np.ones_like(rows)], -1)
    basis = np.asarray(anchor['basis_columns'])
    origin = (pose[:3, 3] - anchor['center_world']) @ basis
    dirs = rays @ pose[:3, :3].T @ basis
    plane_depth = -origin[2] / dirs[..., 2]
    protected = np.logical_or.reduce(foreground(str(raw), str(proc), None, site, name, frame, rect,
                                                np.where(part > 0, depth, plane_depth)))
    xy = (origin + dirs * np.where(part > 0, depth, plane_depth)[..., None])[..., :2]
    valid = (part > 0) & ~protected
    ring = (part == 0) & ~protected & (cv2.distanceTransform(np.uint8(part == 0), cv2.DIST_L2, 5) < 20)
    if valid.sum() < 250 or ring.sum() < 150:
        raise ValueError('Insufficient visible repair or safe road samples')
    return dict(photo=photo, rect=rect, xy=xy, part=part, normals=normals, protected=protected,
                base=median_road(crop[ring]), source_sha256=sha256(path))


def render(view, field, tint, texture, site):
    before = view['photo']
    after = before.copy()
    left, top, right, bottom = view['rect']
    original = before[top:bottom, left:right]
    xy = view['xy']
    uv = np.float32(np.stack([.5+xy[..., 0]/field['width'], .5-xy[..., 1]/field['length']], -1))
    def sample(data):
        pixels = uv * np.float32([data.shape[1]-1, data.shape[0]-1])
        return cv2.remap(data, pixels, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = (view['part'] > 0) & ~view['protected']
    alpha = valid * sample(field['alpha'])
    base = cv2.bilateralFilter(original, 11, 24, 4).astype('f4')
    intrinsic = sample(texture).astype('f4') / np.maximum(tint, 1)
    light = lighting()['sites'][site]
    sun = np.array(light['sun_world'])
    strength = .24 if light['direct_light_used'] else .08
    cosine = np.maximum(0, view['normals'] @ sun)
    reference = np.median(cosine[alpha > .95])
    shading = 1 + strength * (cosine - reference)
    rgb = base * intrinsic * shading[..., None] + .55*(original-base)
    composite = np.uint8(np.clip(original + alpha[..., None]*(rgb-original), 0, 255).round())
    after[top:bottom, left:right] = composite
    seam = sample(field['joint'])
    body = (alpha > .95) & (seam < .05)
    core = (alpha > .95) & (seam > .55)
    if not core.any() or not body.any():
        raise ValueError(f'No visible body or seam core in {site}')
    weights = [.2126, .7152, .0722]
    gain = (composite @ weights) / np.maximum(original @ weights, 1)
    def lab(image):
        return cv2.cvtColor(image.astype('f4')/255, cv2.COLOR_RGB2LAB)
    delta = np.linalg.norm(lab(composite) - lab(original), axis=2)
    stats = dict(host_median_rgb=tint.tolist(), body_delta_E76_median=float(np.median(delta[body])),
                 joint_luminance_ratio=float(np.median(gain[core])), body_luminance_ratio=float(np.median(gain[body])),
                 visible_joint_pixels=int(core.sum()), visible_body_pixels=int(body.sum()),
                 outside_edits=int(np.any(composite != original, axis=2)[view['part'] == 0].sum()),
                 protected_edits=int(np.any(composite != original, axis=2)[view['protected']].sum()),
                 source_sha256=view['source_sha256'])
    if (stats['outside_edits'] or stats['protected_edits'] or
            stats['body_delta_E76_median'] > 5 or stats['joint_luminance_ratio'] > stats['body_luminance_ratio']-.10):
        raise ValueError(f'Colour, support or joint regression in {site}: {stats}')
    return after, stats


def splat_shape(path):
    from plyfile import PlyData

    vertices = PlyData.read(path)['vertex'].data
    digest = hashlib.sha256()
    for name in vertices.dtype.names:
        if not name.startswith(('f_dc_', 'f_rest_')):
            digest.update(vertices[name].tobytes())
    return digest.hexdigest()


def review_plate(dest, field, previews):
    page = Image.new('RGB', (1480, 954), '#f4f2ed')
    draw = ImageDraw.Draw(page)
    def label(pos, title, size, color='#203430'):
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size, index=2)
        draw.text(pos, title, font=font, fill=color)
    def panel(image, pos):
        page.paste(image.resize((700, 185), Image.Resampling.LANCZOS), pos)
    label((32, 13), 'P202   横向修补带与断续接缝', 37)
    label((33, 72), '01   原始截图 · 前景裁切', 21)
    label((744, 72), '02   设计高程 · 蓝色下凹，土黄色凸起', 21)
    panel(Image.open(dest/'reference-crop.jpg'), (32, 114))
    height = field['height']
    level = np.clip(height/.019, -1, 1)[..., None]
    neutral, low, high = np.array([190, 190, 179]), np.array([20, 68, 102]), np.array([192, 137, 59])
    relief = neutral + np.maximum(level, 0)*(high-neutral) + np.maximum(-level, 0)*(low-neutral)
    relief[~field['defect_mask']] = [234, 232, 225]
    relief = Image.fromarray(relief.astype('u1'))
    relief.thumbnail((700, 185), Image.Resampling.LANCZOS)
    draw.rectangle((744, 114, 1443, 298), fill='#eae8e1')
    page.paste(relief, (744+(700-relief.width)//2, 114))
    for row, (site, before, after) in enumerate(previews):
        start = 318 + row*285
        name = '背阴路面' if site == '034' else '日照与原有投影'
        label((32, start), f'PandaSet {site}   /   {name}   ·   迁移前', 23)
        label((744, start), '迁移后   /   同机位、同裁切', 23)
        panel(before, (32, start+46))
        panel(after, (744, start+46))
    label((32, 871), '车载对照为原始前视第 29 帧局部。两个路段共用同一网格；只改变沥青基色与受光。', 22)
    label((32, 912), '单张 RGB 无绝对尺度：平面尺寸及毫米级凹凸为设计假设，不是实测。', 17, '#52645d')
    page.save(dest/'comparison.jpg', quality=95, subsampling=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dest', type=Path)
    parser.add_argument('--anchors', type=Path, default=ROOT/'gen2hole/examples/transverse-repair/anchors.json')
    parser.add_argument('--config', type=Path, default=ROOT/'gen2hole/config/local.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    dest = args.dest
    with np.load(dest/'field.npz') as data:
        field = dict(data)
    vertices, reference, _ = load_glb(dest/'asset.glb')
    splat_reference = splat_shape(dest/'asset.ply') if (dest/'asset.ply').exists() else None
    result, previews = {}, []
    for site, anchor in anchors.items():
        out = dest/site
        out.mkdir(exist_ok=True)
        raw, proc = Path(cfg['pandaset_raw']), Path(cfg['pandaset_processed'])
        def at(frame, surface):
            return capture(raw, proc, site, anchor, frame, vertices, (anchor, surface, field))
        tint = at(29, Surface(dest/'asset.glb'))['base']
        export(dest, out/'asset.glb', tint, cfg)
        _, digest, texture = load_glb(out/'asset.glb')
        if digest != reference:
            raise ValueError('Retint changed vertices or triangle indices')
        surface = Surface(out/'asset.glb')
        stats = {}
        for frame in (25, 29):
            view = at(frame, surface)
            after, stats[frame] = render(view, field, tint, texture, site)
            Image.fromarray(after).save(out/f'front-{frame:02d}-after.webp', lossless=True)
            if frame == 29:
                before = Image.fromarray(view['photo'])
                before.save(out/'front-29-before.webp', lossless=True)
                left, top, right, bottom = view['rect']
                width = max(right-left, int((bottom-top)*700/185))
                cx, cy = (left+right)//2, (top+bottom)//2
                box = (cx-width//2, cy-round(width*185/1400), cx+width//2, cy+round(width*185/1400))
                previews.append((site, before.crop(box), Image.fromarray(after).crop(box)))
        result[site] = dict(geometry_sha256=digest, mesh_sha256=sha256(out/'asset.glb'), anchor=anchor, frames=stats)
        if splat_reference:
            retint_splats(dest, out/'asset.ply', tint)
            if splat_shape(out/'asset.ply') != splat_reference:
                raise ValueError('Retint changed non-colour Gaussian properties')
            result[site]['splat_shape_sha256'] = splat_reference
    review_plate(dest, field, previews)
    write(dest/'transfer.json', dict(original_geometry_sha256=reference, sites=result))


if __name__ == '__main__':
    main()
