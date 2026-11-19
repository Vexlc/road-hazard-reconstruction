"""Create mixed, non-overlapping, multi-asset placement batches for PandaSet."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path

import numpy as np

CURATED = [
    ("055", [
        "model_missing_square_broken_flange_fine_asphalt", "rgbd_pothole_20250305_123822",
        "model_raised_round_raised62_aged_asphalt", "model_pothole_deep_break_fine_asphalt_02",
        "model_raised_rectangular_removed_side_chip_seal", "stereo_pothole_fan_1_01",
        "model_missing_round_masonry_aged_asphalt", "model_pothole_waterlogged_aged_asphalt_02",
    ]),
    ("056", [
        "model_pothole_slab_corner_concrete_03", "model_missing_rectangular_settled_concrete",
        "model_raised_square_propped43_concrete", "model_pothole_joined_concrete_02",
        "model_raised_oval_upright82_concrete", "rgbd_pothole_20250305_135046",
        "model_missing_oval_flush_concrete", "stereo_pothole_fan_2_28",
    ]),
    ("055", [
        "model_pothole_wheel_rut_chip_seal_02", "model_raised_oval_skewed_fine_asphalt",
        "rgbd_pothole_20250227_135438", "model_missing_round_settled_fine_asphalt",
        "model_raised_square_slid_aged_asphalt", "model_pothole_crescent_aged_asphalt_03",
        "stereo_pothole_fan_3_05", "model_missing_rectangular_broken_flange_chip_seal",
    ]),
    ("056", [
        "model_raised_rectangular_ajar14_concrete", "model_missing_square_masonry_concrete",
        "model_pothole_wheel_rut_concrete_03", "model_raised_round_tilted29_concrete",
        "model_pothole_waterlogged_concrete_03", "model_missing_oval_broken_flange_concrete",
        "rgbd_pothole_20250305_142220", "stereo_pothole_fan_2_40",
    ]),
]

SLOTS = {
    "055": ["082", "110", "141", "172", "201", "230", "262", "291"],
    "056": ["081", "112", "141", "170", "201", "232", "261", "292"],
}

def make_plan(library, work):
    rows = json.loads((library/"asset_manifest.json").read_text())
    meta = {row["id"]: row for row in rows}
    slots = {}
    for site, keys in SLOTS.items():
        estimates = json.loads((work/f"{site}_probe.json").read_text())
        slots[site] = [dict(estimates[key], slot=key) for key in keys]
        for anchor in slots[site]:
            assert anchor["rms_m"] < .032 and anchor["points"] > 130 and anchor["neighborhood_coverage"] > .70, anchor
        centers = np.array([a["center_world"][:2] for a in slots[site]])
        assert min(np.linalg.norm(centers[1:]-centers[:-1],axis=1)) > 2.95
    variants = [{"site": site, "assets": items, "showcase": True} for site, items in CURATED]
    used = {name for v in variants for name in v["assets"]}
    assert len(used)==sum(len(v["assets"]) for v in variants) and used<=meta.keys(), used-meta.keys()
    rng = np.random.default_rng(130728)
    pools = {}
    for category in ("raised", "missing", "pothole", "stereo", "rgbd"):
        names = sorted(n for n,m in meta.items() if m["category"]==category and n not in used)
        pools[category] = deque(rng.permutation(names).tolist())
    catalogue = []
    for category, pool in pools.items():
        total=len(pool)
        catalogue.extend((float((i+.27+float(rng.random())*.4)/max(1,total)),name) for i,name in enumerate(pool))
    names = [name for p,name in sorted(catalogue)]
    while names:
        items, names = names[:8], names[8:]
        concrete = sum(meta[n]["material"] == "concrete" for n in items)
        site = "056" if concrete >= 2 or len(variants)%2 else "055"
        variants.append({"site":site, "assets":items,"showcase":False})
    for i, scene in enumerate(variants,1):
        scene["id"]=f"{i:03d}"
    assert Counter(n for v in variants for n in v["assets"]) == Counter(meta.keys())
    return {"slots_by_site":slots,"scenes":variants,"total_assets":len(meta)}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library",type=Path)
    parser.add_argument("work",type=Path)
    args=parser.parse_args()
    plan=make_plan(args.library,args.work)
    (args.work/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    print(len(plan["scenes"]),plan["total_assets"])
