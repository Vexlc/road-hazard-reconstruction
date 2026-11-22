"""Offline indexes for synchronized road crops. Load observations for one physical anchor at a time."""

import json
from pathlib import Path


def reference_image(row):
    """Use a labelled source photo where available, including the dry input to simulated water."""
    for view in row.get('sourceViews', []):
        if view['label'].startswith(('实拍原图', '主照片', '拟合帧', '原始颜色')):
            return {'label': view['label'], 'image': view['image']}
    label = '生成参考图' if row['method'] == '生成 RGB' else '重建参考图'
    return {'label': label, 'image': row['reference']}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def write_catalog(dest, groups):
    overview = {}
    for key, group in groups.items():
        assets = {
            asset: {k: v for k, v in row.items() if k != 'views'} for asset, row in group['assets'].items()
        }
        overview[key] = dict(group, assets=assets)
        views = {asset: row['views'] for asset, row in group['assets'].items()}
        (dest / key / 'views.js').write_text(
            'window.roadViewData(' + compact(key) + ',' + compact(views) + ');\n'
        )
    (dest / 'groups.js').write_text('window.matchedRoad=' + compact(overview) + ';\n')


def read_catalog(dest):
    path = Path(dest)
    raw = (path / 'groups.js').read_text()
    groups = json.loads(raw.removeprefix('window.matchedRoad=')[:-2])
    for key, group in groups.items():
        raw = (path / key / 'views.js').read_text()
        name, views = json.loads('[' + raw.removeprefix('window.roadViewData(')[:-3] + ']')
        if name != key or set(views) != set(group['assets']):
            raise ValueError('View index and asset catalog disagree at ' + key)
        for asset, row in group['assets'].items():
            row['views'] = views[asset]
    return groups
