"""Stateless, addressable recipes: requesting item N never depends on N-1."""
import hashlib
import json
import random

VERSION = 'physical-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def recipe(seed, index, catalog):
    if index < 0:
        raise ValueError('index must be nonnegative')
    category = ('pothole', 'missing', 'raised')[index % 3]
    digest = hashlib.sha256(f'{VERSION}:{seed}:{index}'.encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    candidates = [dict(x) for x in catalog if x['category'] == category]
    # Stratify a stream instead of independently drawing duplicate families/materials.
    slot = index // 3
    def cycle(key):
        values = sorted({x[key] for x in candidates})
        random.Random(f'{VERSION}:{seed}:{category}:{key}').shuffle(values)
        return values[slot % len(values)]
    chosen = [x for x in candidates if x['family'] == cycle('family')
              and x['material'] == cycle('material')
              and ('state' not in x or x['state'] == cycle('state'))]
    base = rng.choice(chosen)
    factor = rng.uniform(.82, 1.19)
    base['width_m'] = round(base['width_m'] * factor, 4)
    base['length_m'] = round(base['length_m'] * factor * rng.uniform(.93, 1.07), 4)
    base['depth_design_m'] = round(base['depth_design_m'] * rng.uniform(.83, 1.17), 4)
    base['seed'] = int(digest[16:24], 16)
    if category == 'raised':
        angle, dx, dy, yaw = base['cover_pose']
        base['cover_pose'] = [round(max(0., min(86., angle + rng.uniform(-5, 5))), 2)
                              if angle else 0., round(dx * factor, 4),
                              round(dy * factor, 4), round(yaw + rng.uniform(-5, 5), 2)]
        base['lid_relief_m'] = round(rng.uniform(.0014, .0032), 5)
        base['lid_wear'] = round(rng.uniform(.12, .68), 4)
    base.update(recipe_version=VERSION, stream_seed=seed, stream_index=index,
                scale_kind='physical_design', size_multiplier=1.,
                provenance_kind='synthetic_geometry_with_scanned_materials',
                limitations=['Designed metric dimensions, not a measurement of a real defect.',
                             'Appearance is baked into degree-1 SH; lighting is not fully relightable.'],
                gaussian_sampling_density_m2=100000, gaussian_sampling_min=100000,
                gaussian_sampling_max=500000,
                gaussian_detail_weights={'Lid face': 6., 'Cast ring and cover seat': 2.5,
                                         'Shaft bottom': .5, 'Inspection shaft': .85})
    base.pop('id', None)
    key = hashlib.sha256(canonical(base).encode()).hexdigest()[:16]
    base['id'] = f"factory_{category}_{seed}_{index:06d}_{key}"
    return base
