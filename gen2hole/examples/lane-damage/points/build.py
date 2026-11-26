"""Build native road patches from PointPSSN XYZ observations and annotated photographic materials."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image
from scan import load, reconstruct
from material import texture

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[3] / "releases/field-chambers/gen2hole"
sys.path[:0] = [
    str(GEN),
    str(GEN / "recipes/series"),
    str(GEN / "recipes/common"),
    str(GEN / "examples/road-distress"),
]
from mesh_io import Mesh, grid_faces, smooth, write_json
from build_chamber import flatten_scene
from export_splats import make_one


def build(
    archive,
    materials,
    out,
    survey=HERE / "survey.json",
    *,
    refine=None,
    partitions=None,
    wall_projection=None,
):
    spec = json.loads(survey.read_text())
    rows = []
    series = out / "series"
    cache = series / "mesh_cache"
    refs = series / "references"
    refs.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source:
        for rec in spec["states"]:
            key = rec["id"]
            for name, digest in rec["sha256"].items():
                if hashlib.sha256(source.read(name)).hexdigest() != digest:
                    raise ValueError(name)
            xyz, labels = load(source, rec["split"], rec["index"])
            m = reconstruct(xyz, labels, rec["pitch_m"], rec.get("apron_m", 0.072))
            if refine is not None:
                refine(m, labels, rec)
            xy, z, mask = m["xy"], m["z"], m["mask"]
            extent = np.ptp(xy, axis=(0, 1))
            install = m.get("install", mask)
            dist = distance_transform_edt(~install) * rec["pitch_m"]
            tri = grid_faces(*z.shape)
            frac = mask.ravel()[tri].sum(1) >= 2
            img, donor, color_reference = texture(spec, rec, m, materials, wall_projection)
            ref = refs / (key + ".jpg")
            Image.fromarray(img).save(ref, quality=95)
            mesh = Mesh()
            mat = mesh.material(
                "Photographed pavement and exposed aggregate", img, 0.94, 0, 2.8
            )
            pieces = (
                [
                    ("Road adjoining the fracture", ~frac, True),
                    ("Fracture lip and exposed bed", frac, False),
                ]
                if partitions is None
                else partitions(m, tri)
            )
            for name, keep, road in pieces:
                mesh.add(
                    name, np.dstack([xy, z]), tri[keep], 0.5 + xy / extent, mat, road
                )
            field = dict(
                height=z,
                alpha=smooth((0.045 - dist) / 0.031),
                defect_mask=mask,
                water_mask=np.zeros_like(mask),
                width=extent[0],
                length=extent[1],
                installation_mask=install,
                registered_relief=True,
                observed_footprint=m["support"],
            )
            for name in ("basin_id", "separator_mask", "crest_mask", "trough_mask"):
                if name in m:
                    field[name] = m[name]
            dest = series / "assets/pothole" / key
            meta = dict(
                id=key,
                category="pothole",
                state="field_pit",
                family=spec.get("family", "wuchang_lidar_pits"),
                surface_class="pothole",
                label=rec["label"],
                source=spec["paper"],
                source_author=spec["authors"],
                source_image="references/" + ref.name,
                source_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                source_record=rec["sha256"],
                training_cross_validation=m["fit"],
                source_license="No explicit dataset licence in the linked repository.",
                contrast_reference_pothrgbd=color_reference,
                texture_source=[
                    spec["road_material"]["page"],
                    spec["materials"][donor]["page"],
                ],
                appearance="Photographic material analogue. The XYZ file contains no site colour.",
                vertical_scale_basis=spec["units"],
                source_origin_xyz=m["origin"].tolist(),
                source_local_basis_columns=m["basis"].tolist(),
                source_support_radius_m=m["limit"],
                depth_p95_m=float(np.percentile(-z[mask], 95)),
                surface_dimensions_m=extent.tolist(),
                seed=19000 + int(key[1:]),
            )
            mesh.save(dest, cache, meta, field)
            flatten_scene(dest / "asset.glb")
            evaldir = out / "evaluation" / key
            evaldir.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                evaldir / "survey.npz", xyz=xyz, defect=labels, reserved=m["hold"]
            )
            row = make_one(dest, cache)
            for role, name in [
                ("glb", "asset.glb"),
                ("ply", "asset.ply"),
                ("surface", "surface.npz"),
                ("metadata", "metadata.json"),
            ]:
                row[role] = str((dest / name).relative_to(series / "assets"))
            rows.append(row)
            print(key, flush=True)
    write_json(series / "assets/asset_manifest.json", rows)
    write_json(out / "run.json", dict(editions=["series"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "materials", "out"):
        p.add_argument(name, type=Path)
    build(**vars(p.parse_args()))
