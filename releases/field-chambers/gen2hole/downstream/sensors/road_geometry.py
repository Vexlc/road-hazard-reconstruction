"""Geometry shared by the ray replay and the vehicle surface queries (metres, Z up)."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import open3d as o3d

PACKAGE_NAMES = ('道路病害图生三维', '道路病害图生三维_新增100')
PARTS = ('background', 'adjoining road', 'fracture bed and lip', 'fracture sidewall',
         'water surface', 'cast ring and seat', 'shaft wall', 'shaft bottom',
         'concrete collar', 'lid face', 'lid underside', 'lid edge', 'hinge',
         'pavement block top', 'pavement block edge', 'pavement block underside', 'subgrade', 'socket wall', 'visible debris face', 'opening inner rim', 'central bearing beam',
         'concrete barrier', 'hydrant body', 'hydrant cap', 'hydrant chain', 'traffic cone', 'tyre', 'wheel rim', 'crate', 'cement sack',
         'shipping carton', 'tool chest', 'garden sack', 'wood box')
EXACT_PARTS = {
    "Setts at the isolated asphalt loss": 178, "Setts at the connected asphalt loss": 179,
    "Photographed corroded chequer plate": 163, "Rust perforation cut edge": 164,
    'Printed wrapper caught in grate': 156, 'Snagged grate rod': 157, 'Cover lifting staple': 158, 'Photographed gutter lip': 98,
    'Pavement strip between paired cracks': 160, 'First surveyed crack bed': 161, 'Second surveyed crack bed': 162,
    'Enclosed pavement island': 153, 'Measured rim shelf 01': 154, 'Companion pothole bed': 155, 'Measured trough within pit bed': 159,
    'Parted main concrete cover': 147, 'Parted corner of concrete cover': 148,
    'Exposed transverse grate blade': 149, 'Displaced grate edge rail': 150,
    'Recess on photographed bearing': 16, 'Photographed recess back': 6, 'Photographed recessed bed': 7, 'Photographed soffit': 10,
    'Fractured inlet surround': 151, 'Broken inlet sill': 152, 'Rubble at inlet mouth': 18,
    'Broken cover facet': 14, 'Grate blade flank': 11, 'Hidden broken cover facet': 14, 'Hidden grate blade flank': 11,
    'Embedded curb end left': 1, 'Embedded curb end right': 1, 'Embedded curb rear': 1,
    'Sidewalk beside cap left': 97, 'Sidewalk beside cap right': 97,
    'Curb beside cap left': 110, 'Curb beside cap right': 110,
    'Scoured curb crown': 110, 'Overhanging concrete cap': 13, 'Exposed curb break face': 14,
    'Overhanging curb underside': 15, 'Recessed inlet back': 6, 'Scoured gutter floor': 2,
    'Curb bearing left': 110, 'Curb bearing right': 110, 'Curb return left': 6, 'Curb return right': 6,
    'Gutter shoulder left': 98, 'Gutter shoulder right': 98,
    'Concrete slab side joint left': 125, 'Concrete slab side joint right': 125,
    'Concrete slab rear joint slope': 125, 'Concrete slab rear joint rise': 125,
    'Surveyed road marking': 138, 'Crack crossing road paint': 139,
    'Surveyed pit bed 1': 134, 'Surveyed pit bed 2': 135, 'Surveyed pit bed 3': 136, 'Surveyed inter-pit road': 137,
    'Low foliage in the joint': 130, 'Joint plant leaf blades': 131, 'Joint plant stems': 132,
    'Vegetated joint soil': 133, 'Bridge joint header': 123, 'Recessed joint base': 124, 'Joint interior face': 125,
    'Longitudinal joint remnant': 126, 'Asphalt patch': 128, 'Patch broken edge': 128,
    'Black insulated cables': 115, 'Grey insulated cables': 116, 'Tracer wire': 117,
    'Concealed timber bearing': 122, 'Lid edge': 11, 'Wall cable fitting': 118, 'Wall fitting face': 118,
    **{f'Electrical conduit {i} {part}': code for i in (1, 2, 3)
       for part, code in [('jacket', 119), ('bore', 120), ('end', 121)]},
    'Measured pavement around crack': 113, 'Textured road marking edge': 114,
    'Embedded steel step': 111, 'Cast iron throat skirt': 112,
    'Slate paving stone': 13,
    'Patterned twin-inlet deck': 108, 'Circular cover rebate': 5,
    'Twin arched inlet face': 109, 'Twin inlet reveal': 96, 'Twin-inlet cast underside': 96,
    'Broken concrete inlet sill': 8, 'Sill crown': 8, 'Coursed concrete chamber back': 6,
    'Chamber joint recess': 6, 'Casting end supports': 110, 'Curb block crowns': 110,
    'Outer curb ends': 110, 'Casting end skirt': 96, 'Casting rear edge': 96,
    'Chequered access plate': 104, 'Slotted casting face': 105, 'Casting pocket floor': 106,
    'Painted descending slab': 107, 'Painted descending slab painted edge': 14, 'Painted descending slab edge': 14, 'Painted descending slab underside': 15,
    'Photographed concrete riser': 102, 'Inferred concrete riser': 103,
    'Low detached asphalt': 13, 'Low detached asphalt edge': 14, 'Low detached asphalt underside': 15,
    'Uplifted asphalt cap edge': 14, 'Uplifted asphalt cap painted edge': 14, 'Uplifted asphalt cap underside': 15,
    'Upper painted concrete course': 102, 'Lower concrete shaft course': 103,
    'Uplifted asphalt cap': 13, 'Ruptured asphalt cap edge': 14, 'Ruptured asphalt cap underside': 15,
    'Left torn asphalt flap': 13, 'Right torn asphalt flap': 13,
    'Left torn asphalt flap edge': 14, 'Right torn asphalt flap edge': 14,
    'Left torn asphalt flap underside': 15, 'Right torn asphalt flap underside': 15,
    'Registered front casting': 5, 'Casting perimeter wall': 5, 'Lower casting flange': 5,
    'Lower casting flange edge': 5, 'Front casting gussets': 147, 'Level-setting shims': 148,
    'Outlet bevel': 99, 'Outlet bore': 100, 'Outlet continuation': 101,
    'Measured high spot on pit bed': 144, 'Photographed asphalt fissure': 145, 'Photographed bed fragment': 146, 'Photographed verge': 2,
    'Trapped leaf and silt': 142, 'Trapped leaf and silt edge': 142,
    'Draped litter sheet': 143, 'Draped litter sheet edge': 143,
    'Recessed grate crossbars': 95, 'Curb throat casting': 96, 'Curb hood top': 96, 'Kerb and sidewalk': 97, 'Concrete gutter pan': 98,
    'Cavity inferred closure': 3, 'Fracture inset rim': 19, 'Road around the aperture': 1, 'Road adjoining the fracture': 1,
    'Fracture lip and exposed bed': 2, 'Fracture sidewall in surface coordinates': 3,
    'Water surface': 4, 'Cast ring and cover seat': 5, 'Cast seat riser and ledge': 5, 'Inspection shaft': 6,
    'Shaft bottom': 7, 'Chipped concrete collar': 8, 'Lid face': 9,
    'Lid underside with casting ribs': 10, 'Lid rolled edge': 11,
    'Lid underside': 10, 'Lid aperture walls and perimeter': 11,
    'Lid fragment face': 9, 'Lid fragment underside': 10, 'Lid fragment aperture walls and perimeter': 11,
    'Left lid face': 9, 'Right lid face': 9, 'Left lid underside': 10, 'Right lid underside': 10,
    'Left lid aperture walls and perimeter': 11, 'Right lid aperture walls and perimeter': 11,
    'Central bearing beam': 20,
    'Detached pavement top': 13, 'Detached pavement edges': 14, 'Detached pavement underside': 15,
    'Exposed slab foundation': 16, 'Subgrade bearing': 16, 'Subgrade bearing surface': 16, 'Slab socket wall': 17,
}
PROP_PREFIXES = [('Tyre ', 26), ('Wheel rim ', 27), ('Road crate ', 28), ('Cement sack ', 29),
                         ('Shipping carton ', 30), ('Tool chest ', 31), ('Garden sack ', 32), ('Wood box ', 33),
                         ('Tied refuse sack ', 34), ('Upturned basket ', 35), ('Roadworks drum ', 36),
                         ('Dropped wrench ', 37), ('Cross pein hammer ', 38), ('Bent crowbar ', 39),
                         ('Sledgehammer ', 40), ('Wooden pail ', 41), ('Forked branch ', 42),
                         ('Road rock ', 43), ('Loose stone ', 44), ('Dry branch ', 45),
                         ('Metal jerrycan ', 46), ('Plastic jerrycan ', 47), ('Oil tin ', 48),
                         ('Folding stool ', 49), ('Fallen ladder ', 50), ('Galvanized bin ', 51),
                         ('Detached lid ', 52), ('Rusty can ', 53), ('Tape measure ', 54),
                         ('Hand truck ', 55), ('Storage cart ', 56), ('Tool cart ', 57),
                         ('Empty wheelchair ', 58), ('Steel pipe ', 59), ('Exposed field conduit ', 59), ('Steel tee ', 60),
                         ('Steel cross ', 61), ('PVC elbow ', 62), ('Offset tube ', 63),
                         ('Return bend ', 64), ('Valve body ', 65), ('PVC reducer ', 66),
                         ('Work boot ', 67), ('Junction box ', 68), ('Double pipe bracket ', 69),
                         ('Worn football ', 72), ('Panel football ', 73), ('Soft football ', 74),
                         ('Stitched baseball ', 75), ('Gridiron ball ', 76), ('Life ring ', 77),
                         ('Flotation vest ', 78), ('Travel case ', 79), ('Garden gloves ', 80),
                         ('Monobloc chair ', 81),
                         ('Long spade ', 82), ('Hand trowel ', 83), ('Dustpan ', 84),
                         ('Wood broom ', 85), ('Plastic broom ', 86), ('Watering can ', 87),
                         ('Garden sprinkler ', 88), ('Seed tray ', 89), ('Clay planter ', 90),
                         ('Bolt cutters ', 91), ('Panel saw ', 92), ('Woven basket ', 93),
                         ('Basket lid ', 94)]
# The scan verifier indexes this tuple with the stored numeric part ID.
_part_labels = {code: prefix.strip().lower() for prefix, code in PROP_PREFIXES}
_part_labels.update({129: 'broken concrete fragment', 127: 'exposed reinforcing steel', 70: 'bore wall', 71: 'bore lip'})
for _name, _code in EXACT_PARTS.items():
    _part_labels.setdefault(_code, _name.lower())
_part_labels.update({96: 'inlet casting edges and reveal', 110: 'curb support blocks',
                     118: 'wall cable fitting', 119: 'electrical conduit jacket',
                     120: 'electrical conduit bore', 121: 'electrical conduit termination',
                     140: 'grate twig', 141: 'grate leaf'})
_part_labels.update({165: "blue instrument enclosure", 166: "pale junction housing",
                     167: "white rounded enclosure", 168: "exposed metal spindle",
                     169: "pipe fitting", 170: "yellow insulated lead", 171: "grey insulated lead",
                     172: "dry stem", 173: "leaf litter", 174: "folding manhole guard",
                     175: "ribbed road insert", 176: "intersection delineator and ballast",
                     177: "yellow braided line"})
PARTS += tuple(_part_labels[code] for code in range(len(PARTS), max(_part_labels) + 1))

CLASSES = {'pothole': 1, 'missing': 2, 'raised': 3}
SITES = ('034', '109', '110')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for buf in iter(lambda: stream.read(1048576), b''):
            digest.update(buf)
    return digest.hexdigest()


def load_inventory(parent):
    assets, scenes, anchors = {}, {}, {}
    parent = Path(parent)
    # Each gen2hole run declares its editions. Historical releases use the original folder names.
    marker = parent / 'run.json'
    packages = read_json(marker)['editions'] if marker.is_file() else PACKAGE_NAMES
    for name in packages:
        root = Path(parent) / name
        for meta in root.glob('assets/*/*/metadata.json'):
            key = meta.parent.name
            assert key not in assets
            assets[key] = (meta.parent, read_json(meta))
        for path in sorted(root.glob('scenes/*/scene.json')):
            info = read_json(path)
            case = path.parent.name
            assert case not in scenes
            scenes[case] = (path, info)
            for pl in info['placements']:
                group = (info['site'], pl['slot'])
                center, basis = np.array(pl['center_world']), np.array(pl['basis_columns'])
                assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-5) and np.linalg.det(basis) > .99999
                assert pl['size_multiplier'] == 1.
                if group in anchors:
                    old_center, old_basis = anchors[group]
                    assert np.allclose(old_center, center, rtol=0, atol=1e-7) and np.allclose(old_basis, basis, rtol=0, atol=1e-7)
                anchors[group] = center, basis
    if not assets:
        raise ValueError(f"No rebuilt or published asset packages in {parent}")
    return assets, scenes, anchors


def part_id(name):
    if name.startswith("Exposed asphalt wall "):
        return 180
    if name.startswith("Inferred wall ") or name == "Optical pocket":
        return 181
    contents = {"Black sack": 182, "Folded cardboard": 183, "Trapped cone": 184}
    base_name = name.removesuffix(" reverse").removesuffix(" back").removesuffix(" side").removesuffix(" continuation")
    if base_name in contents:
        return contents[base_name]
    if name.startswith("Delineator "):
        return 176
    if name == "Yellow braided line":
        return 177
    if name == "Photographed ribbed road insert":
        return 175
    verge_parts = {"Blue instrument enclosure": 165, "Pale junction housing": 166,
                  "White rounded enclosure": 167, "Exposed metal spindle": 168,
                  "Pipe coupling": 169, "Pipe elbow": 169, "Yellow insulated lead": 170,
                  "Grey insulated lead": 171, "Crossing dry stem": 172, "Right-hand dry stem": 172,
                  "Leaf litter in chamber": 173, "Concrete frame on verge": 8,
                  "Grass and soil beside chamber": 1}
    if name.removesuffix(" side") in verge_parts:
        return verge_parts[name.removesuffix(" side")]
    if name.startswith('Snagged grate cross tie '):
        return 95
    if name.startswith('Snagged grate rod '):
        return 157
    if name.startswith('Companion pothole bed '):
        return 155
    if name.startswith('Enclosed pavement island '):
        return 153
    if name.startswith('Exposed transverse grate blade '):
        return 149
    if name.startswith('Photographed bed fragment '):
        return 146
    if name.startswith('Photographed asphalt fissure '):
        return 145
    if name.startswith('Grate twig '):
        return 140
    if name.startswith('Grate leaf '):
        return 141
    if name.startswith(('Surveyed bowed grate bar ', 'Inlet strip ', 'Inlet round bar ')):
        return 95
    if name.startswith('Vegetated joint asphalt plug '):
        return 128
    if name.startswith('Vegetated joint wall '):
        return 3
    if name.startswith('Parallel propagation crack '):
        return 2
    if name.startswith('Spall inspection apron '):
        return 1
    if name.startswith('Broken concrete fragment '):
        return 129
    if name.startswith('Exposed reinforcing bar '):
        return 127
    if name in {'Road adjoining the fracture / 2', 'Bridge joint header / 2', 'Joint interior face / 2'}:
        name = name.split(' / ')[0]
    if " [bore] " in name:
        return 70
    if " [lip] " in name:
        return 71
    for prefix, code in PROP_PREFIXES:
        if name.startswith(prefix):
            return code
    if name.startswith('Folding manhole guard panel '):
        return 174
    if name.startswith('Traffic cone '):
        return 25
    if name.startswith('Barrier '):
        return 21
    if name.startswith('Hydrant '):
        return 24 if 'chain' in name else 23 if 'cap_' in name else 22
    if name.startswith('Fracture debris '):
        return 18
    return 12 if name.startswith('Hinge ') else EXACT_PARTS[name]


def dielectric_spec(material):
    """Tinted dielectric reflectance at normal incidence for untextured glTF specular factors."""
    ext = material.get("extensions", {})
    spec = ext.get("KHR_materials_specular")
    pbr = material["pbrMetallicRoughness"]
    if not spec or pbr.get("metallicFactor", 1) != 0 or any(k.endswith("Texture") for k in spec):
        return None
    ior = ext.get("KHR_materials_ior", {}).get("ior", 1.5)
    f0 = ((ior - 1) / (ior + 1))**2 * np.array(spec.get("specularColorFactor", [1.,1.,1.]))
    return np.float32(np.clip(f0 * spec.get("specularFactor", 1.), 0, 1)), pbr.get("roughnessFactor", 1.)


def glb_triangles(path, *, names=False, appearance=False, normal_map=False, packed_material=False):
    """Read the flat, uncompressed indexed meshes exported by the two reconstruction packages."""
    data = Path(path).read_bytes()
    magic, version, total, jsize, tag = struct.unpack_from('<4sIIII', data)
    if (magic, version, total, tag) != (b'glTF', 2, len(data), 0x4e4f534a):
        raise ValueError(f'Not a glTF 2 binary file: {path}')
    end = 20 + jsize
    gltf = json.loads(data[20:end])
    blen, btag = struct.unpack_from('<II', data, end)
    assert btag == 0x004e4942 and end+8+blen == len(data)
    binary = memoryview(data)[end+8:]

    def accessor(i):
        a = gltf['accessors'][i]
        view = gltf['bufferViews'][a['bufferView']]
        dtype = {5121:'u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[a['componentType']]
        size = {'VEC4': 4, 'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
        if view.get('byteStride', np.dtype(dtype).itemsize*size) != np.dtype(dtype).itemsize*size or 'sparse' in a:
            raise ValueError('Unexpected non-packed asset accessor')
        return np.frombuffer(binary, dtype=dtype, offset=view.get('byteOffset',0)+a.get('byteOffset',0),
                             count=a['count']*size).reshape(-1, size)

    for i in gltf['scenes'][gltf.get('scene', 0)]['nodes']:
        node = gltf['nodes'][i]
        if set(node) - {'name','mesh','extras'}:
            raise ValueError('Expected world-aligned exported nodes')
        mesh = gltf['meshes'][node['mesh']]
        for primitive in mesh['primitives']:
            assert primitive.get('mode', 4) == 4
            p = accessor(primitive['attributes']['POSITION'])
            v = np.stack((p[:,0], -p[:,2], p[:,1]), axis=-1)
            f = accessor(primitive['indices']).reshape(-1, 3).astype(np.uint32)
            part = (mesh['name'] if names else part_id(mesh['name']), np.ascontiguousarray(v), f)
            if appearance or normal_map or packed_material:
                uv = accessor(primitive['attributes']['TEXCOORD_0'])
                material = gltf['materials'][primitive['material']]
                tex = material['normalTexture'] if normal_map else material['pbrMetallicRoughness']['baseColorTexture']
                source = gltf['textures'][tex['index']]['source']
                view = gltf['bufferViews'][gltf['images'][source]['bufferView']]
                start = view.get('byteOffset', 0)
                part += (uv, binary[start:start + view['byteLength']])
                if packed_material:
                    src = gltf['textures'][material['pbrMetallicRoughness']['metallicRoughnessTexture']['index']]['source']
                    packed = gltf['bufferViews'][gltf['images'][src]['bufferView']]
                    offset = packed.get('byteOffset', 0)
                    normals = accessor(primitive['attributes']['NORMAL'])
                    normals = np.stack((normals[:,0], -normals[:,2], normals[:,1]), axis=-1)
                    part += (normals, binary[offset:offset + packed['byteLength']])
                    sky = primitive['attributes'].get('_SKY_VISIBILITY')
                    part += ((None if sky is None else accessor(sky).ravel(), dielectric_spec(material)),)
                    tangent_id = primitive["attributes"].get("TANGENT")
                    if tangent_id is not None and "normalTexture" in material:
                        t = accessor(tangent_id)
                        t = np.stack((t[:,0], -t[:,2], t[:,1], t[:,3]), axis=-1)
                        normal_tex = material["normalTexture"]
                        image_id = gltf["textures"][normal_tex["index"]]["source"]
                        packed_normal = gltf["bufferViews"][gltf["images"][image_id]["bufferView"]]
                        pos = packed_normal.get("byteOffset", 0)
                        part += ((t, binary[pos:pos + packed_normal["byteLength"]], float(normal_tex.get("scale", 1))),)
            yield part


class Surface:
    def __init__(self, path, threads=2, exclude=(), include_names=None):
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.threads = threads
        self.parts = []
        lo, hi = [], []
        for name, v, tri in glb_triangles(path, names=True):
            part = part_id(name)
            if part in exclude or (include_names is not None and name not in include_names):
                continue
            gid = self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(tri))
            assert gid == len(self.parts)
            self.parts.append(part)
            lo.append(v.min(axis=0)); hi.append(v.max(axis=0))
        self.parts = np.asarray(self.parts, 'u1')
        self.low, self.high = np.min(lo, axis=0), np.max(hi, axis=0)

    def cast(self, origins, directions):
        rays = np.ascontiguousarray(np.concatenate((origins, directions), axis=-1), dtype='f4')
        hit = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=self.threads)
        dist, ids = hit['t_hit'].numpy(), hit['geometry_ids'].numpy()
        valid = np.isfinite(dist)
        parts = np.zeros(dist.shape, 'u1')
        parts[valid] = self.parts[ids[valid]]
        return dist, parts, hit['primitive_normals'].numpy()
