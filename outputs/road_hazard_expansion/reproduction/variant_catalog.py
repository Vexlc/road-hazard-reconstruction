"""Design families and deterministic dimensions for the road asset library."""
import itertools

PAVEMENTS = {
    "fine_asphalt": ("Asphalt031", (0.73, 0.75, 0.78), .004),
    "aged_asphalt": ("Asphalt031", (1.02, 1.00, .95), .006),
    "chip_seal": ("Concrete049", (.65, .68, .68), .008),
    "concrete": ("Concrete034", (.88, .90, .91), .003),
}
PITS = {
    "spall":       (.28, .25, .025),
    "bowled":      (.60, .46, .073),
    "deep_break":  (.92, .70, .148),
    "crescent":    (.75, .51, .063),
    "wheel_rut":   (.47, 1.35, .086),
    "joined":      (1.02, .62, .104),
    "cluster":     (.82, .78, .066),
    "patch_break": (.98, .72, .109),
    "slab_corner": (1.13, .76, .098),
    "waterlogged": (.83, .63, .102),
}
SHAFTS = {
    "round": (.64, .64, 2.0),
    "oval": (.57, .78, 2.0),
    "square": (.65, .65, 7.0),
    "rectangular": (.61, .84, 9.0),
}
MISSING = ["flush", "masonry", "broken_flange", "settled"]
LID_POSES = {
    "ajar14": (14, 0., 0., 0.),
    "tilted29": (29, 0., 0., 0.),
    "propped43": (43, 0., 0., 0.),
    "raised62": (62, 0., 0., 0.),
    "upright82": (82, 0., 0., 0.),
    "slid": (0, .38, .03, 4.),
    "skewed": (23, .24, .02, 27.),
    "removed_side": (0, .91, .15, -16.),
}


def variants(pit_variants=3):
    for fi, (family, (w, l, depth)) in enumerate(PITS.items()):
        for pi, surface in enumerate(PAVEMENTS):
            for size in range(1, pit_variants + 1):
                factor = (.75, 1., 1.22)[(size - 1) % 3] * (1 + .025 * ((size - 1) // 3))
                key = f"model_pothole_{family}_{surface}_{size:02}"
                yield dict(id=key, category="pothole", family=family, material=surface,
                           width_m=round(w*factor, 3), length_m=round(l*factor, 3),
                           depth_design_m=round(depth*factor*(.96+.032*pi), 3),
                           seed=14791 + fi*770 + pi*118 + size*1591)
    for fi, (family, (w, l, exponent)) in enumerate(SHAFTS.items()):
        for pi, surface in enumerate(PAVEMENTS):
            for si, (category, state) in enumerate(itertools.chain(
                (("missing", s) for s in MISSING), (("raised", s) for s in LID_POSES)
            )):
                key = f"model_{category}_{family}_{state}_{surface}"
                cfg = dict(id=key, category=category, family=family, state=state, material=surface,
                           width_m=round(w*(.94+.042*((pi+si)%4)), 3),
                           length_m=round(l*(.94+.042*((pi+si)%4)), 3),
                           exponent=exponent, depth_design_m=round(.83 + .09*((fi+pi+si)%6), 2),
                           seed=34123 + fi*821 + pi*97 + si*355)
                if category == "raised":
                    cfg["cover_pose"] = LID_POSES[state]
                yield cfg
