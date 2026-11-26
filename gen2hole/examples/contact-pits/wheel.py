"""Sample a rigid transverse cylinder on a surveyed height grid, retaining gaps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def roll(height, supported, x, y, row, radius, width):
    dx = x[1] - x[0]
    half = int(np.floor(radius / dx + 1e-8))
    offsets = np.arange(-half, half + 1)
    band = abs(y - y[row]) <= width / 2 + 1e-9
    # The full tread must fit inside the raster, even when every in-raster cell is valid.
    side_covered = y.max() - y[row] >= width / 2 - 1e-9 and y[row] - y.min() >= width / 2 - 1e-9
    crown = np.max(height[band], axis=0)
    line = sliding_window_view(np.pad(crown, half, constant_values=-np.inf), len(offsets))
    known = sliding_window_view(np.pad(supported[band].all(axis=0), half), len(offsets)).all(axis=1)
    known &= side_covered
    # Subtract R to express centre height relative to the same wheel on z=0.
    arc = np.sqrt(np.maximum(radius**2 - (offsets * dx) ** 2, 0)) - radius
    axle = np.where(known, (line + arc).max(axis=1), np.nan)
    ground = np.where(supported[row], height[row], np.nan)
    return ground, axle, known


def check_reference():
    xy = np.linspace(-1, 1, 401)
    h = np.zeros((401, 401))
    support = np.ones_like(h, bool)
    support[:, 320] = False
    h[:, 190:211] = -0.10
    ground, axle, known = roll(h, support, xy, xy, 200, 0.30, 0.18)
    assert np.isclose(ground[200], -0.10)
    assert np.isclose(axle[200], np.sqrt(0.30**2 - 0.055**2) - 0.30)
    assert np.isclose(axle[120], 0.0) and not known[320] and not known[20]
    assert not roll(h, support, xy, xy, 5, 0.30, 0.18)[2].any()


def build(assets, output):
    check_reference()
    output.mkdir(parents=True, exist_ok=True)
    survey = json.loads(Path(__file__).with_name("survey.json").read_text())
    fig, axes = plt.subplots(len(survey["states"]), 2, figsize=(11, 9), constrained_layout=True)
    rows = {}
    for rec, (plan, section) in zip(survey["states"], axes):
        key = rec["id"]
        with np.load(assets / key / "surface.npz") as field:
            z = field["height"]
            support = field["observed_footprint"].astype(bool)
            defect = field["defect_mask"].astype(bool)
            width, length = float(field["width"]), float(field["length"])
        x = np.linspace(-width / 2, width / 2, z.shape[1])
        y = np.linspace(length / 2, -length / 2, z.shape[0])
        row_ids = np.rint(np.quantile(np.nonzero(defect)[0], [0.25, 0.50, 0.75])).astype(int)
        plan.imshow(
            np.ma.array(-z * 100, mask=~support),
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap="cividis",
            vmin=0,
            aspect="equal",
        )
        plan.set(
            title=key + " | survey and test tracks",
            xlabel="Longitudinal (m)",
            ylabel="Transverse (m)",
        )
        section.set(
            title="Centreline surface and rolling wheel",
            xlabel="Longitudinal (m)",
            ylabel="Height (cm)",
        )
        section.grid(alpha=0.2)
        ground, axle, valid = [], [], []
        rows[key] = []
        for idx, col in zip(row_ids, ["#d77b21", "#257995", "#965697"]):
            g, a, v = roll(z, support, x, y, idx, 0.30, 0.18)
            if not v.any() or not np.all(a[v] + 1e-7 >= g[v]):
                raise ValueError(f"{key}: incomplete or inconsistent wheel track")
            ground.append(g)
            axle.append(a)
            valid.append(v)
            plan.axhline(y[idx], color=col, lw=0.9)
            section.plot(x, g * 100, ":", color=col, lw=0.9)
            section.plot(x, a * 100, color=col, lw=1.6, label=f"y={y[idx]:+.2f} m")
            rows[key].append(
                dict(
                    y_m=float(y[idx]),
                    supported_positions=int(v.sum()),
                    max_centreline_depth_on_comparable_samples_m=float(max(0, -g[v].min())),
                    max_axle_drop_m=float(max(0, -a[v].min())),
                    max_wheel_to_centreline_clearance_m=float((a[v] - g[v]).max()),
                )
            )
        section.legend(fontsize=8)
        np.savez_compressed(
            output / (key + "-wheel.npz"),
            x_m=x,
            y_m=y[row_ids],
            centreline_z_m=ground,
            axle_minus_radius_m=axle,
            full_support=valid,
            radius_m=0.30,
            tread_width_m=0.18,
        )
    (output / "wheel.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig.savefig(output / "wheel.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    build(**vars(parser.parse_args()))
