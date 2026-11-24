"""Check whether a mark inside the original pavement leaks into the inserted native Gaussian colors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "streetcrafter/joint"), str(ROOT / "gen2hole/downstream")]
from pavement_layout import PavementLayout
from pit_pairs import C0, RIG, digest, load_rig, road_plate, terrain_asset
from road_geometry import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = ROOT / "gen2hole/examples/asphalt-breakouts"
    p.add_argument("--layout", type=Path, default=src / "layout.json")
    p.add_argument("--anchor", type=Path, default=src / "kitti.json")
    p.add_argument("--assets", type=Path, default=ROOT / "runs/asphalt-breakouts")
    p.add_argument(
        "--out", type=Path, default=ROOT / "outputs/asphalt-breakouts/native-paint.json"
    )
    a = p.parse_args()
    layout = PavementLayout(a.layout, a.assets)
    anchor = json.loads(a.anchor.read_text())
    layout.bind(anchor)
    split, cameras = load_rig(RIG)
    summary = {}
    for seat, (_, folder) in zip(anchor["instances"], layout.members):
        center = np.array(seat["anchor"]["center_world"])
        basis = np.array(seat["anchor"]["basis_columns"])
        axis, plate = road_plate(center, basis, cameras, split)
        with np.load(folder / "surface.npz") as field:
            cut = field["defect_mask"].astype(bool)
            size = np.array([float(field["width"]), float(field["length"])])
        xy = np.stack(np.meshgrid(axis, axis), axis=-1)
        uv = xy / size * [1, -1] + 0.5
        idx = np.rint(np.clip(uv, 0, 1) * (np.array(cut.shape[::-1]) - 1)).astype(int)
        inside = (uv >= 0).all(-1) & (uv <= 1).all(-1) & cut[idx[:, :, 1], idx[:, :, 0]]
        clearance = distance_transform_edt(inside) * (axis[1] - axis[0])
        y, x = np.unravel_index(clearance.argmax(), clearance.shape)
        stripe = (
            (clearance > 0.075)
            & (abs(xy[..., 1] - axis[y]) < 0.09)
            & (abs(xy[..., 0] - axis[x]) < clearance[y, x] * 0.8)
        )
        painted = plate.copy()
        painted[stripe] = [0.87, 0.86, 0.83]
        case = {
            "model": str((folder / "asset.glb").resolve()),
            "label": seat["id"],
            "outline_area_m2": float(inside.sum() * (axis[1] - axis[0]) ** 2),
            "native_hashes": {"asset.ply": digest(folder / "asset.ply")},
            "source_ply": seat["asset"] + "/asset.ply",
        }
        row = {"road_texels_overpainted": int(stripe.sum())}
        for flag, label in [(False, "image_context"), (True, "registered")]:
            states = []
            for image in (plate, painted):
                _, components, _, _ = terrain_asset(
                    seat["asset"],
                    case,
                    (center, basis, {}),
                    (axis, image),
                    None,
                    voxel=0.009,
                    apron_exposure=flag,
                )
                states.append(components[0])
            plain, altered = states
            if not np.array_equal(plain["_xyz"], altered["_xyz"]):
                raise ValueError("Mark changed native geometry")
            delta = (
                np.abs(plain["_features_dc"].numpy() - altered["_features_dc"].numpy())
                * C0
                * 255
            )
            row[label] = {
                "changed_above_one_level": int(
                    np.count_nonzero(delta.max(axis=(1, 2)) > 1)
                ),
                "max_srgb_delta": float(delta.max()),
            }
        if (
            row["image_context"]["changed_above_one_level"] < 20
            or row["registered"]["max_srgb_delta"] > 0
        ):
            raise ValueError("Native road marking survived the registered aperture")
        print(seat["asset"], row, flush=True)
        summary[seat["asset"]] = row
    save_json(a.out, summary)


if __name__ == "__main__":
    main()
