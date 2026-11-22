"""Create native Gaussian previews with the original PLY and a reproducible metric camera."""

import argparse
import fcntl
from hashlib import sha256
import inspect
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import gsplat
import numpy as np
from PIL import Image
import torch

from atlas_gaussians import ATLAS, BACKGROUND, ROOT, catalog, digest, geometry_frame, load, projection, render


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument('--out', type=Path, default=ROOT / 'runs/atlas-gaussians')
    cmd.add_argument('--ids', nargs='+')
    args = cmd.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.render.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        render_batch(args)


def render_batch(args):
    registry = args.out / 'views.json'
    rows = catalog()
    ids = args.ids or list(rows)
    views = json.loads(registry.read_text()) if registry.is_file() else {}
    torch.set_num_threads(4)
    w, h = 640, 400
    views = {k: v for k, v in views.items() if k in rows}
    spec = str((gsplat.__version__, BACKGROUND.tolist(), w, h))
    signature = sha256(
        (spec + ''.join(inspect.getsource(fn) for fn in (load, projection, render))).encode()
    ).hexdigest()
    for i, key in enumerate(ids):
        row = rows[key]
        path = args.out / (key + '.webp')
        sha = digest(ATLAS / row['ply'])
        glb_hash = digest(ATLAS / row['glb'])
        camera = geometry_frame(row, w / h)
        previous = views.get(key, {})
        if (
            previous.get('ply_sha256') == sha
            and previous.get('render_sha256') == signature
            and previous.get('camera') == camera
            and path.is_file()
            and previous.get('image_sha256') == digest(path)
        ):
            previous['glb_sha256'] = glb_hash
            continue
        rgb, alpha, _ = render(load(row), camera, w, h)
        with NamedTemporaryFile(dir=args.out, delete=False) as stream:
            draft = Path(stream.name)
        try:
            Image.fromarray(rgb).save(draft, format='WEBP', quality=92, method=4)
            draft.replace(path)
        finally:
            draft.unlink(missing_ok=True)
        views[key] = dict(
            render_sha256=signature,
            ply_sha256=sha,
            glb_sha256=glb_hash,
            image_sha256=digest(path),
            count=row['gaussians'],
            width=w,
            height=h,
            camera=camera,
            foreground_pixels=int(np.count_nonzero(alpha > 0.5)),
        )
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(ids)}', flush=True)
    with NamedTemporaryFile(mode='w', dir=args.out, delete=False) as stream:
        json.dump(views, stream, indent=2, allow_nan=False)
        stream.write('\n')
        draft = Path(stream.name)
    draft.replace(registry)
    print(f'{len(views)} Gaussian previews in {args.out}')


if __name__ == '__main__':
    main()
