"""Draw the photographed pothole and two exported mesh profiles on the atlas sampling grid."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, PAPER = '#1d5071', '#ab602d', '#243234', '#f7f5ee'


def profile(run, key):
    raw = (ROOT / 'tools/asset_atlas/relief' / (key + '.js')).read_text()
    name, data = json.loads('[' + raw[len('window.roadRelief(') : -3] + ']')
    assets = run / 'series/assets/pothole' / key
    compare = run / 'series/comparison' / key
    for path, digest in [
        (assets / 'asset.glb', 'glb_sha256'),
        (compare / 'range-only.glb', 'range_only_glb_sha256'),
        (compare / 'support.glb', 'support_glb_sha256'),
    ]:
        if key != name or hashlib.sha256(path.read_bytes()).hexdigest() != data[digest]:
            raise ValueError('Rebuild the atlas profile from this mesh before rendering ' + key)
    n = data['n']

    def values(key):
        q = np.frombuffer(base64.b64decode(data[key]), '<u2').astype(float).reshape(n, n)
        return np.where(q > 0, data['low'] + (q - 1) * data['step'], np.nan).astype('f4')

    top, baseline = values('top'), values('rangeOnly')
    source = np.frombuffer(base64.b64decode(data['support']), 'u1').reshape(n, n)
    # Match the browser peak selection and retain its row on the photographed texture.
    delta = top.astype(float) - baseline
    i, j = np.unravel_index(np.where(source == 2, abs(delta), -1).argmax(), source.shape)
    with np.load(assets / 'surface.npz') as field:
        width = float(field['width']) * 1000
    with Image.open(run / 'series/references' / (key + '_ortho.png')) as image:
        photo = np.array(image.convert('RGB'))
    return (
        1000 * np.linspace(*data['x'], n),
        1000 * np.linspace(*data['y'], n),
        1000 * top,
        1000 * baseline,
        source,
        i,
        j,
        width,
        photo,
    )


def draw(run, output):
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font)
    plt.rcParams.update(
        {
            'font.family': FontProperties(fname=font).get_name(),
            'axes.unicode_minus': False,
            'text.color': INK,
            'axes.labelcolor': INK,
            'xtick.color': INK,
            'ytick.color': INK,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), dpi=155, gridspec_kw={'height_ratios': [1.33, 1]})
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.824, bottom=0.18, wspace=0.16, hspace=0.22)
    fig.text(0.075, 0.965, '同一条坑壁截线，两种重建高度', fontsize=24, weight='bold', va='top')
    fig.text(
        0.076,
        0.898,
        '黑圈标出彩照单独支持的格点中，两种模型高差最大的位置。下排截面沿照片中的黄线读取。',
        fontsize=12,
    )
    specs = json.loads((run / 'series/calibration/survey.json').read_text())
    cases = {spec['id']: spec for spec in specs}
    for c, key in enumerate(('P142', 'P143')):
        spec = cases[key]
        x, y, top, base, source, i, j, width, photo = profile(run, key)
        yy, xx = np.nonzero(source < 3)
        xlim = (x[xx].min() - 6, x[xx].max() + 6)
        ylim = (y[yy].min() - 5, y[yy].max() + 5)
        a = axes[0, c]
        a.imshow(photo, extent=[-width / 2, width / 2, -width / 2, width / 2], origin='upper')
        a.axhline(y[i], color='#ffda73', lw=1.8)
        a.plot(x[j], y[i], 'o', ms=9, mfc='none', mec='black', mew=2.8)
        a.plot(x[j], y[i], 'o', ms=7, mfc='none', mec='#fff', mew=1)
        a.set_title(key + '   ' + spec['label'], loc='left', fontsize=15, pad=7)
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.axis('off')
        b = axes[1, c]
        b.set_facecolor(PAPER)
        b.spines[['right', 'top']].set_visible(False)
        b.axhline(0, color='#829084', lw=0.9)
        b.plot(x, top[i], color=BLUE, lw=2.4)
        b.plot(x, base[i], color=ORANGE, dashes=(4, 3), lw=2.2)
        b.axvline(x[j], color='#657270', lw=0.85)
        b.scatter([x[j], x[j]], [top[i, j], base[i, j]], c=[BLUE, ORANGE], zorder=3, s=26)
        b.set_xlim(xlim)
        b.set_ylim(-50, 7)
        b.set_xlabel('局部横向位置（毫米）')
        if c == 0:
            b.set_ylabel('相对路面高程（毫米）')
        b.set_title(
            '截线 Y = {:+.1f} 毫米    ·    标记处高差 {:+.1f} 毫米'.format(y[i], top[i, j] - base[i, j]),
            fontsize=11,
            loc='left',
        )
        b.grid(axis='y', lw=0.45, alpha=0.35)
    legend = [Line2D([0], [0], color=BLUE, lw=2.4), Line2D([0], [0], color=ORANGE, lw=2.2, dashes=(4, 3))]
    fig.legend(
        legend,
        ['距离图与彩照补深', '仅用距离图'],
        bbox_to_anchor=(0.075, 0.10),
        loc='center left',
        frameon=False,
        ncol=2,
        fontsize=12,
    )
    fig.text(
        0.076,
        0.032,
        '高差为两份网格在同一位置的竖直高度差。与实测深度的误差另由留出机位检验。',
        fontsize=11,
    )
    fig.savefig(output, dpi=155, facecolor=PAPER)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/pit-depth-completion')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/figures/pit-depth-sections.jpg')
    args = parser.parse_args()
    draw(args.run, args.out)
