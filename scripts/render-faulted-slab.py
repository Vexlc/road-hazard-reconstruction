"""Export the observed detached block and render registered source, solid and metric section plates."""

import argparse
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont, ImageOps
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gen2hole/recipes' / n) for n in ('common', 'series')]
from render_mesh import prepare_view  # noqa: E402

INK = '#243b33'
MUTED = '#53665b'
PAPER = '#f8f6ef'
COLORS = {'top': '#246c65', 'edge': '#995934', 'base': '#927549', 'road': '#57615b'}
LABELS = {'P144': '原位', 'P145': '凸起', 'P146': '下沉', 'P147': '脱落'}


def read(p):
    return json.loads(p.read_text())


def text(d, x, y, msg, size=24, bold=False, color=INK, anchor='lt'):
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-' + ('Bold' if bold else 'Regular') + '.ttc'
    d.text((x, y), msg, fill=color, font=ImageFont.truetype(font, size, index=2), anchor=anchor)


def rgb(p):
    with Image.open(p) as im:
        return im.convert('RGB')


def extract(asset, output):
    scene = trimesh.load(asset, force='scene', process=False)
    piece = trimesh.Scene()
    for n in sorted(scene.graph.nodes_geometry):
        if n.startswith('Detached pavement '):
            matrix, g = scene.graph[n]
            # Keep already decoded JPEG colour and normal texels exact when building a new GLB.
            material = scene.geometry[g].visual.material
            for prop in ('baseColorTexture', 'normalTexture', 'metallicRoughnessTexture'):
                getattr(material, prop).format = 'PNG'
            piece.add_geometry(scene.geometry[g], geom_name=n, node_name=n, transform=matrix)
    output.write_bytes(trimesh.exchange.gltf.export_glb(piece, include_normals=True))
    saved = trimesh.load(output, force='scene', process=False)
    if len(saved.graph.nodes_geometry) != 3:
        raise ValueError('Isolated block lost geometry')
    for name in saved.graph.nodes_geometry:
        a_matrix, a = scene.graph[name]
        b_matrix, b = saved.graph[name]
        first, last = scene.geometry[a], saved.geometry[b]
        if not (
            np.array_equal(a_matrix, b_matrix)
            and np.array_equal(first.vertices, last.vertices)
            and np.array_equal(first.faces, last.faces)
            and np.allclose(first.visual.uv, last.visual.uv, atol=1e-6, rtol=0)
        ):
            raise ValueError('Isolated block differs from its parent: ' + name)
        for prop in ('baseColorTexture', 'normalTexture', 'metallicRoughnessTexture'):
            if not np.array_equal(
                np.asarray(getattr(first.visual.material, prop)),
                np.asarray(getattr(last.visual.material, prop)),
            ):
                raise ValueError('Isolated block material differs: ' + name)


def render_block(bundle, folder, dest):
    scene, _ = prepare_view(folder, bundle / 'mesh_cache', 'inspection', 840, 520)
    for ob in list(scene.objects):
        if ob.type == 'MESH' and not ob.name.startswith('Detached pavement '):
            bpy.data.objects.remove(ob, do_unlink=True)
    away = Vector((0.94, -0.38, 0.52)).normalized()
    center = Vector((-0.005, -0.008, -0.023))
    cam = scene.camera
    cam.location = center + away * 3
    cam.rotation_euler = (-away).to_track_quat('-Z', 'Y').to_euler()
    cam.data.ortho_scale = 0.56
    scene.render.filepath = str(dest.resolve())
    bpy.ops.render.render(write_still=True)


def section(asset, meta, dest):
    scene = trimesh.load(asset, force='scene', process=False)
    n = meta['intervals']
    pitch = meta['source_pitch_m']
    # The source-column offset avoids cuts exactly along shared triangle edges.
    x = (meta['slab']['seed_col_row_in_crop'][0] + 0.37 - n / 2) * pitch
    im = Image.new('RGB', (1050, 296), PAPER)
    d = ImageDraw.Draw(im)
    left, right = 104, 1016
    def coord(y, z):
        return left + (y + 0.36) / 0.72 * (right - left), 68 + (0.042 - z) / 0.14 * 174

    for mm in (35, 0, -25, -50, -85):
        yy = coord(0, mm / 1000)[1]
        d.line((left, yy, right, yy), fill='#cdd3c7', width=1)
        text(d, left - 12, yy, f'{mm:+d}', 20, color=MUTED, anchor='rm')
    text(d, 18, 12, '高程 / mm', 19, color=MUTED)
    for cm in (-30, -15, 0, 15, 30):
        xx = coord(cm / 100, 0)[0]
        d.line((xx, 58, xx, 233), fill='#dce0d6', width=1)
        text(d, xx, 239, str(cm), 19, color=MUTED, anchor='mt')
    text(d, right, 267, '沿截线距离 / cm', 17, color=MUTED, anchor='rt')
    order = sorted(scene.graph.nodes_geometry, key=lambda s: s.startswith('Detached'))
    for name in order:
        trans, g = scene.graph[name]
        mesh = scene.geometry[g].copy()
        mesh.apply_transform(trans)
        seg = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [x, 0, 0])
        role = (
            'top'
            if name == 'Detached pavement top'
            else 'edge'
            if name.startswith('Detached') or name == 'Slab socket wall'
            else 'base'
            if 'foundation' in name or 'bearing' in name
            else 'road'
        )
        for a, b in seg:
            # glTF is Y-up. Source pavement local Y is minus glTF Z.
            d.line(
                (*coord(-a[2], a[1]), *coord(-b[2], b[1])),
                fill=COLORS[role],
                width=4 if role == 'top' else 3,
            )
    im.save(dest, quality=95, subsampling=0)


def figure(run, out):
    bundle = run / 'series'
    ev = run / 'evaluation'
    records = read(ev / 'measurements.json')
    first = 'P144'
    folder = bundle / 'assets/pothole' / first
    meta = read(folder / 'metadata.json')
    data = meta['slab']
    n = meta['intervals']
    pitch = meta['source_pitch_m']
    ref = rgb(bundle / meta['source_image'])
    overlay = ImageDraw.Draw(ref)
    pts = np.asarray(data['boundary_xy']) / pitch * [1, -1] + n / 2
    coords = [tuple(x) for x in pts]
    overlay.line([*coords, coords[0]], fill='#3ddac0', width=4)
    cx = data['seed_col_row_in_crop'][0] + 0.37
    for y in range(0, ref.height, 18):
        overlay.line((cx, y, cx, min(y + 9, ref.height)), fill='#ffc158', width=3)
    ref.save(ev / 'outline.jpg', quality=95)
    extract(folder / 'asset.glb', ev / 'block.glb')
    render_block(bundle, folder, ev / 'block.jpg')
    full = Image.new('RGB', (1610, 1220), PAPER)
    d = ImageDraw.Draw(full)

    def paste(im, x, y, w, h):
        im = ImageOps.contain(im, (w, h), method=Image.Resampling.LANCZOS)
        full.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))

    text(d, 34, 23, '一块实拍碎板，从闭合裂缝到四种路面状态', 43, True)
    text(d, 36, 89, '混凝土扫描　/　同一顶面与纹理　/　原位、凸起、下沉和脱落', 24, color=MUTED)
    paste(ref, 34, 149, 300, 265)
    paste(rgb(ev / 'block.jpg'), 362, 149, 470, 265)
    text(d, 62, 425, '扫描顶面与提取边界', 22, True)
    text(d, 455, 425, '独立碎板实体', 22, True)
    text(d, 878, 155, '从原图恢复了什么', 30, True)
    area = records[first]['source_area_m2']
    extent = data['grid_xy_extent_m']
    text(
        d,
        880,
        216,
        f'闭合区域 {area * 10000:.0f} cm²，板块约 {extent[0] * 100:.1f} × {extent[1] * 100:.1f} cm。',
        23,
    )
    text(d, 880, 267, '顶面取自扫描的颜色与位移，青线沿原图闭合裂缝。', 22)
    text(d, 880, 315, '厚度、高差和基层采用设计值，黄虚线为下方截线。', 22)
    text(d, 880, 366, '图中高度以局部路面为零，凸起为正、下沉为负。', 21, color=MUTED)
    for i, key in enumerate(LABELS):
        folder = bundle / 'assets/pothole' / key
        dst = ev / key
        dst.mkdir(exist_ok=True)
        section(folder / 'asset.glb', read(folder / 'metadata.json'), dst / 'section.jpg')
        x = 34 + i * 393
        y = 485
        d.line((x, y, x + 366, y), fill='#b9c4b9', width=2)
        text(d, x, y + 17, LABELS[key] + '　' + key, 29, True)
        rec = records[key]
        note = (
            f"顶面整体移动 {rec['vertical_difference_mm']:+.0f} mm" if key != 'P147' else '露出基层　−85 mm'
        )
        text(d, x, y + 69, note, 22, color=MUTED)
        paste(rgb(bundle / 'views' / (key + '_inspection.jpg')), x, y + 111, 365, 247)
        paste(rgb(bundle / 'views' / (key + '_reverse.jpg')), x, y + 363, 365, 200)
        paste(rgb(dst / 'section.jpg'), x, y + 576, 365, 104)
    text(
        d,
        35,
        1180,
        '各栏为同一米制场景的两个斜视方向。截面取自导出网格，绿色为碎板顶面，棕色为切面与基层。',
        20,
        color=MUTED,
    )
    full.save(out, quality=95, subsampling=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    figure(args.run, args.out)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
