"""Place several road openings on independently surveyed support planes in the same metric scene."""

from itertools import combinations, product
import json
from pathlib import Path
import re

import cv2
import numpy as np

from mesh_transfer import RoadMesh, fit_footprint, fit_road, footprint_size
from prop_lighting import view_bounds
from road_geometry import sha256


def asset_folder(root, key):
    if not isinstance(key, str) or not key.isalnum():
        raise ValueError("Use an alphanumeric catalogue asset ID")
    direct = root / key
    if (direct / "asset.glb").is_file():
        return direct
    matches = list(root.glob(f"*/series/assets/*/{key}/asset.glb"))
    if len(matches) != 1:
        raise ValueError(f"Expected one built asset named {key} under {root}")
    return matches[0].parent


class PavementLayout:
    standing = False

    def __init__(self, recipe, root):
        self.digest = sha256(recipe)
        data = json.loads(Path(recipe).read_text())
        self.key, self.label = data["id"], data["label"]
        if not isinstance(self.key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.key):
            raise ValueError("Use a simple layout ID for the output directory")
        self.members = []
        seen = set()
        self.meshes = {}
        for row in data["instances"]:
            key, name = row["asset"], row["id"]
            xy, angle = np.asarray(row["xy_m"], float), float(row["heading_deg"])
            if name in seen or not name or xy.shape != (2,) or not np.isfinite([*xy, angle]).all():
                raise ValueError("Invalid or repeated pavement instance")
            seen.add(name)
            path = asset_folder(root, key)
            if key not in self.meshes:
                mesh = RoadMesh(path, filter_pavement=True, measured_rim=True)
                if mesh.standing:
                    raise ValueError(
                        "Use the roadside layout for freestanding props and their ground shadows"
                    )
                self.meshes[key] = mesh
            self.members.append((row, path))
        if not 1 <= len(self.members) <= 65535:
            raise ValueError("Expected one or more pavement instances fitting uint16 labels")

    def fit(self, base, points, up, radius):
        origin, basis = np.array(base["center_world"]), np.array(base["basis_columns"])
        seats, hulls = [], []
        for row, path in self.members:
            theta = np.deg2rad(row["heading_deg"])
            seed = origin + basis[:, :2] @ row["xy_m"]
            heading = basis[:, :2] @ [np.cos(theta), np.sin(theta)]
            seat = fit_road(points, seed, heading, up_world=up, radius_m=radius)
            mesh = self.meshes[row["asset"]]
            size = footprint_size(path)
            if mesh.cover_outline is not None:
                # A rasterized top surface can stop one cell inside the GLB edge. Older
                # fields can omit an entire moved lid; extend the fit for that overhang.
                cell = mesh.extent / (np.array(mesh.cut.shape[::-1]) - 1)
                cover_span = 2 * abs(mesh.cover_outline).max(axis=0)
                if np.any(cover_span > size + 2 * cell):
                    size = np.maximum(size, cover_span)
            seat = fit_footprint(seat, points, size)
            y, x = np.nonzero(mesh.cut)
            uv = np.column_stack([x, y]) / (np.array(mesh.cut.shape[::-1]) - 1)
            xy = (uv - 0.5) * mesh.extent * [1, -1]
            if mesh.cover_outline is not None:
                xy = np.concatenate([xy, mesh.cover_outline])
            seat_basis = np.array(seat["basis_columns"])
            world = xy @ seat_basis[:, :2].T + seat["center_world"]
            hulls.append(cv2.convexHull(np.float32(((world - origin) @ basis)[:, :2])))
            seats.append(
                dict(
                    id=row["id"],
                    asset=row["asset"],
                    mesh_sha256=mesh.digest,
                    surface_sha256=sha256(path / "surface.npz"),
                    anchor=seat,
                )
            )
        if any(cv2.intersectConvexConvex(a, b)[0] > 1e-7 for a, b in combinations(hulls, 2)):
            raise ValueError("The surveyed openings have overlapping convex footprints")
        return dict(base, layout_sha256=self.digest, instances=seats)

    def bind(self, anchor):
        if anchor.get("layout_sha256") != self.digest or len(anchor["instances"]) != len(self.members):
            raise ValueError("The placement belongs to a different pavement layout")
        for seat, (row, path) in zip(anchor["instances"], self.members):
            mesh = self.meshes[row["asset"]]
            if not (
                seat["id"] == row["id"]
                and seat["asset"] == row["asset"]
                and seat["mesh_sha256"] == mesh.digest
                and seat["surface_sha256"] == sha256(path / "surface.npz")
            ):
                raise ValueError(f"Asset or geometry changed after fitting {row['id']}")

    def foreground_points(self, points, anchor, height_m):
        """Classify raw LiDAR returns against the nearest surveyed local road plane."""
        best = np.full(len(points), np.inf)
        above = np.zeros(len(points), bool)
        for seat in anchor["instances"]:
            local = seat["anchor"]
            xyz = (points - local["center_world"]) @ np.array(local["basis_columns"])
            distance = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
            closer = distance < best
            above[closer] = xyz[closer, 2] > height_m
            best = np.minimum(best, distance)
        return above

    def visible_bounds(self, anchor):
        points = []
        for seat in anchor["instances"]:
            local = seat["anchor"]
            mesh = self.meshes[seat["asset"]]
            bounds = mesh.bounds if mesh.measured_rim else view_bounds(mesh, local)
            corners = np.array(list(product(*bounds.T)))
            world = corners @ np.array(local["basis_columns"]).T + local["center_world"]
            points.extend((world - anchor["center_world"]) @ np.array(anchor["basis_columns"]))
        return np.array([np.min(points, axis=0), np.max(points, axis=0)])

    def sample(self, rgb, pose, rays, anchor, host, protected, **options):
        output = rgb.copy()
        distance = np.full(rgb.shape[:2], np.inf, "f4")
        part = np.zeros(rgb.shape[:2], "u1")
        instance = np.zeros(rgb.shape[:2], "u2")
        for index, seat in enumerate(anchor["instances"], 1):
            mesh = self.meshes[seat["asset"]]
            color, labels, metric = mesh.render(rgb, pose, rays, seat["anchor"], host, protected, **options)
            nearer = (labels > 0) & (metric < distance)
            output[nearer] = color[nearer]
            distance[nearer] = metric[nearer]
            part[nearer] = labels[nearer]
            instance[nearer] = index
        distance[instance == 0] = np.nan
        return output, part, distance, instance


def resolve_instances(samples):
    distances = np.stack([np.where(np.isfinite(s[2]), s[2], np.inf) for s in samples])
    nearest = np.argmin(distances, axis=0)[None]
    return np.take_along_axis(np.stack([s[3] for s in samples]), nearest, axis=0)[0]
