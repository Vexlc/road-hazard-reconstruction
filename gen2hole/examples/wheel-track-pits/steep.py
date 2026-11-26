"""Recover the steep trench using oriented 3-D neighbourhoods and trim unsupported completion."""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, Delaunay


def reconstruct_wall(m, training, labels):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(training))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=8))
    cloud.orient_normals_consistent_tangent_plane(18)
    normals = np.asarray(cloud.normals)
    if np.median(normals[~labels, 2]) < 0:
        normals *= -1
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=8, scale=1.08, n_threads=1
    )
    vertices = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    tree = cKDTree(training)
    distance, idx = tree.query(vertices)
    near = distance < 0.031
    in_capture = Delaunay(training[:, :2]).find_simplex(vertices[:, :2]) >= 0
    in_grid = (abs(vertices[:, :2]) <= np.ptp(m["xy"], axis=(0, 1)) / 2).all(axis=1)
    aligned = (np.asarray(mesh.vertex_normals) * normals[idx]).sum(1) > -0.15
    tri = tri[(near & in_capture & in_grid & aligned)[tri].all(axis=1)]
    # Canonical indices keep the native sampling independent of the mesher's vertex ordering.
    verts, remap = np.unique(np.round(vertices[tri.ravel()], 5), axis=0, return_inverse=True)
    tri = remap.reshape(-1, 3)
    a, b, c = verts[tri].transpose(1, 0, 2)
    tri = tri[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1.0e-11]
    pivot = tri.argmin(1)
    tri = np.take_along_axis(tri, (pivot[:, None] + np.arange(3)) % 3, axis=1)
    tri = np.unique(tri, axis=0)
    distance, idx = tree.query(verts[tri].mean(1), k=5)
    weight = 1 / np.maximum(distance, 0.003) ** 2
    is_pit = (weight * labels[idx]).sum(1) > 0.5 * weight.sum(1)
    m["observed_mesh"] = (verts, tri, is_pit)
    # An overhang can have several depths at one X,Y. Export the first vertical
    # return for installation and retain all inclined faces for optical/LiDAR rays.
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(verts.astype("f4")), o3d.core.Tensor(tri.astype("uint32")))
    xy = m["xy"]
    ray = np.zeros((*xy.shape[:2], 6), "f4")
    ray[:, :, :2] = xy
    ray[:, :, 2] = 1
    ray[:, :, 5] = -1
    hit = scene.cast_rays(o3d.core.Tensor(ray), nthreads=1)
    t = hit["t_hit"].numpy()
    visible = np.isfinite(t)
    mask = np.zeros_like(visible)
    mask[visible] = is_pit[hit["primitive_ids"].numpy()[visible]]
    m.update(
        z=np.where(visible, 1.0 - t, 0).astype("f4"),
        support=visible,
        mask=mask,
        confidence=mask.astype("f4"),
        capture_reconstruction=dict(
            method="Trimmed screened Poisson with oriented source normals",
            normal_neighbours=8,
            depth=8,
            max_observation_distance_m=0.031,
            orientation_min_dot=-0.15,
            first_hit_grid_pitch_m=0.004,
            triangles=len(tri),
        ),
    )


def texture_walls(m, image, vertices, faces, uv, extent):
    if "observed_mesh" not in m:
        return image, vertices, faces, uv
    import cv2
    from scipy.ndimage import distance_transform_edt

    corners = vertices[faces]
    a, b, c = corners.transpose(1, 0, 2)
    normals = abs(np.cross(b - a, c - a))
    steep = normals[:, 2] < 0.63 * np.linalg.norm(normals, axis=1)
    # Copy a clean exposed-aggregate swatch from the already colour-matched
    # atlas. Use the same texel density vertically as on the bed.
    h, w = image.shape[:2]
    interior = cv2.resize(m["mask"].astype("u1"), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    clearance = distance_transform_edt(interior)
    y, x = np.unravel_index(clearance.argmax(), clearance.shape)
    r = int(min(76, clearance[y, x] / 1.5))
    swatch = image[y - r : y + r, x - r : x + r]
    periodic = np.tile(
        np.vstack(
            [np.hstack([swatch, swatch[:, ::-1]]), np.hstack([swatch[::-1], swatch[::-1, ::-1]])]
        ),
        (h // (4 * r) + 1, w // (4 * r) + 1, 1),
    )[:h, :w]
    # Keep top-down UVs on the upper half. Align the second half with the dominant
    # tangent direction and height, which prevents pixels stretching down cliff faces.
    tex = uv[faces].copy()
    tex[:, :, 1] = 0.5 + tex[:, :, 1] * 0.5
    use_x = normals[:, 1] >= normals[:, 0]
    along = np.where(use_x[:, None], corners[:, :, 0], corners[:, :, 1])
    tex[steep, :, 0] = 0.5 + along[steep] / extent[0]
    tex[steep, :, 1] = 0.25 + corners[steep, :, 2] / (2 * extent[1])
    m["capture_reconstruction"]["steep_textured_faces"] = int(steep.sum())
    # Share vertices within each chart, retaining true UV seams where charts meet.
    flat = np.column_stack([corners.reshape(-1, 3), tex.reshape(-1, 2)])
    unique, index = np.unique(flat, axis=0, return_inverse=True)
    return np.vstack([image, periodic]), unique[:, :3], index.reshape(-1, 3), unique[:, 3:]
