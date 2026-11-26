"""Place the visible stripe on the already painted lane dash in a PandaSet exposure."""

import cv2
import numpy as np
from PIL import Image

from mesh_transfer import fit_road, unit


def anchor_from_stripe(
    raw, shot, model, poses, drive, record, source_line, source_tip=None
):
    if (
        raw.name != record["scene"]
        or shot != record["frame"]
        or list(model[0]) != record["image_size"]
    ):
        raise ValueError("Stripe annotation does not match driving exposure")
    _, project, unproject = model
    seed, points, heading = drive.host_patch(
        raw, shot, project, poses, record["seed_px"]
    )
    grade = fit_road(points, seed, heading, up_world=[0, 0, 1], radius_m=1.05)
    normal = np.asarray(grade["basis_columns"])[:, 2]
    cam = drive.pose_matrix(poses[shot])
    eye = cam[:3, 3]

    def ground(x, y):
        r, ok = unproject(int(x), int(y), 1, 1)
        if not ok[0, 0]:
            raise ValueError("Pixel outside marked lane")
        d = cam[:3, :3] @ r[0, 0]
        return eye + d * ((np.asarray(grade["center_world"]) - eye) @ normal) / (
            d @ normal
        )

    raw_image = raw / "camera/front_camera" / f"{shot:02}.jpg"
    gray = np.array(Image.open(raw_image).convert("L")).astype("f4")
    a, b = record["row_range"]
    az, bz = record["center_x_hint"]
    samples = []
    image_edges = []
    for row in range(a, b + 1, 2):
        x = round(az + (bz - az) * (row - a) / (b - a))
        half = record["search_half_width_px"]
        profile = cv2.GaussianBlur(
            gray[row - 1 : row + 2, x - half : x + half + 1].mean(0).reshape(1, -1),
            (0, 0),
            1.2,
        )[0]
        background = np.median(np.r_[profile[:5], profile[-5:]])
        lit = profile > background + (np.percentile(profile, 90) - background) * 0.5
        ix = np.flatnonzero(lit)
        if not len(ix) or np.percentile(profile, 90) - background < 35:
            continue
        pieces = np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)
        section = min(pieces, key=lambda ids: abs(np.median(ids) - half))
        if section[0] == 0 or section[-1] == len(profile) - 1 or len(section) < 5:
            continue
        left, right = x - half + section[[0, -1]]
        samples.append((ground(left, row), ground(right, row), row))
        image_edges.append([row, left, right])
    if len(samples) < 15:
        raise ValueError("Lane dash not resolved")
    ends = np.array([v[:2] for v in samples])
    center = ends.mean(1)
    fit = center[::2]
    origin = fit.mean(0)
    _, _, vt = np.linalg.svd(fit - origin)
    along = unit(vt[0] - normal * (vt[0] @ normal))
    if along @ (center[0] - center[-1]) < 0:
        along = -along
    across = unit(np.cross(along, normal))
    local = np.asarray(source_line)
    _, _, lv = np.linalg.svd(local - local.mean(0))
    d = lv[0] if lv[0] @ (local[0] - local[-1]) > 0 else -lv[0]
    local_basis = np.array([[d[0], d[1], 0], [d[1], -d[0], 0], [0, 0, 1]])
    # local_basis rows are along, right and up. Reflect both right axes to keep det +1.
    basis = np.column_stack([along, across, normal]) @ local_basis
    nearest = local.mean(0) - d * (local.mean(0) @ d)
    halfwidth_offset = record.get("center_offset_halfwidths", 0)
    if halfwidth_offset:
        # A partial source stripe sometimes provides only one measured edge.
        width = np.median(abs((ends[:, 1] - ends[:, 0]) @ across))
        nearest = nearest + halfwidth_offset * 0.5 * width * np.array([d[1], -d[0]])
    world = origin - basis @ [*nearest, 0]
    tip_record = {}
    if source_tip is not None:
        rail = np.asarray(image_edges)
        y0, y1 = record["tip_search_rows"]
        rows = np.arange(y0, y1 + 1)
        ends_px = np.column_stack(
            [np.polyval(np.polyfit(rail[:, 0], rail[:, i], 1), rows) for i in (1, 2)]
        )
        terminals = []
        for t in np.linspace(*record["tip_clear_width_fraction"], 33):
            col = np.rint((1 - t) * ends_px[:, 0] + t * ends_px[:, 1]).astype(int)
            sample = gray[rows, col]
            bright = np.median(sample[:4])
            dark = np.median(sample[-4:])
            if bright - dark < 35:
                continue
            selected = np.flatnonzero(sample > dark + 0.5 * (bright - dark))
            if not len(selected) or selected[-1] >= len(rows) - 2:
                continue
            j = selected[-1]
            terminals.append((int(col[j]), int(rows[j])))
        terminals = list(dict.fromkeys(terminals))
        if len(terminals) < 7:
            raise ValueError("Insufficient observed stripe-end pixels")
        tips = np.asarray([ground(x, y) for x, y in terminals])
        old_tip = world + basis @ [*source_tip, 0]
        shift = float(np.median((tips[::2] - old_tip) @ along))
        world = world + along * shift
        fitted_tip = world + basis @ [*source_tip, 0]
        tip_record = dict(
            source_tip_xy=source_tip,
            host_edge_px=terminals,
            held_end_p90_longitudinal_error_m=float(
                np.percentile(abs((tips[1::2] - fitted_tip) @ along), 90)
            ),
            stripe_axis_transverse_error_m=float((fitted_tip - origin) @ across),
        )
    holdout = abs((center[1::2] - origin) @ across)
    if np.max(holdout) > 0.025 or abs(np.linalg.det(basis) - 1) > 1e-5:
        raise ValueError("Lane registration failed")
    return dict(
        grade,
        center_world=world.tolist(),
        basis_columns=basis.tolist(),
        stripe_reference=record,
        **({"paint_tip_registration": tip_record} if tip_record else {}),
        stripe_grade_line=dict(center_xy=nearest.tolist(), direction=d.tolist()),
        stripe_profile={
            "fit_rows": len(fit),
            "check_rows": len(holdout),
            "max_center_error_m": float(max(holdout)),
            "width_median_m": float(np.median(abs((ends[:, 1] - ends[:, 0]) @ across))),
            "interval_length_m": float(np.ptp((center - origin) @ along)),
        },
    )


def match_marking_width(mesh, meta, anchor):
    """Widen the instance's painted band by transferring low-frequency colour over its existing roughness."""
    from pavement_texture import pyramid
    from hashlib import sha256

    target = anchor["stripe_profile"]["width_median_m"]
    reference = meta["source_stripe_widths_m"]["ground"]
    factor = target / reference
    box = np.array(meta["stripe_texture_box"])
    width, height = box[2:] - box[:2]
    records = meta["surface_stripe_rows"]
    painted = None
    for gid, (part, tri, uv, tex) in mesh.surfaces.items():
        if part not in (13, 14, 104, 105, 106, 107) or tex.shape[:2] != (height, width):
            continue
        if painted is None:
            painted = tex.copy()
            base = cv2.GaussianBlur(tex.astype("f4"), (0, 0), 2)
            for region in ("cap", "ramp"):
                samples = np.asarray(records[region]) - box[:2]
                ys = samples[:, 0, 1]
                sides = samples[:, :, 0]
                slope, offset = np.polyfit(ys, sides, 1)
                extra = np.median(np.diff(ys))
                lo = max(0, int(ys[0] - 2.5 * extra))
                hi = min(height, int(ys[-1] + 1.6 * extra))
                for y in range(lo, hi):
                    l, r = slope * y + offset
                    center = (l + r) / 2
                    hw = (r - l) / 2
                    x = np.arange(
                        max(0, int(center - max(factor, 1) * hw * 1.32)),
                        min(width, int(center + max(factor, 1) * hw * 1.32 + 1)),
                    )
                    donor = center + (x - center) / factor
                    baseline = base[y, x]
                    virtual = np.stack(
                        [np.interp(donor, x, baseline[:, i]) for i in range(3)], axis=1
                    )
                    # Keep the original aggregate and pattern relief at their real coordinates.
                    edge = np.clip(
                        (1.30 - abs(x - center) / (max(factor, 1) * hw)) / 0.16, 0, 1
                    )
                    delta = virtual - baseline
                    if factor >= 1:
                        delta = np.maximum(delta, 0)
                    else:
                        delta = np.minimum(delta, 0)
                    painted[y, x] = np.uint8(
                        np.clip(tex[y, x] + edge[:, None] * delta, 0, 255)
                    )
            changed = np.any(painted != tex, axis=2)
            anchor["stripe_appearance"] = dict(
                native_ground_width_m=reference,
                host_ground_width_m=target,
                width_ratio=factor,
                changed_atlas_texels=int(changed.sum()),
                native_rgb_sha256=sha256(tex.tobytes()).hexdigest(),
                host_rgb_sha256=sha256(painted.tobytes()).hexdigest(),
            )
        mesh.surfaces[gid] = (part, tri, uv, painted)
        xyz, _ = mesh.filtered_textures[gid]
        mesh.filtered_textures[gid] = (xyz, pyramid(painted))
    if painted is None:
        raise ValueError("Registered paint profile did not find its source texture")
    return painted
