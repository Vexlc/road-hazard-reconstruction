"""Verify road placements of the measured pits against the meshes, splats and sensor replay."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(run):
    bundle = run / "series"
    sensors = run / "sensors"
    measured = read(run / "evaluation/measurements.json")
    hashes = read(bundle / "checks.json")
    laser = read(sensors / "checks/scan_audit.json")
    cameras = read(sensors / "checks/camera_audit.json")
    ray_test = read(sensors / "checks/independent_geometry.json")
    geometry = read(sensors / "geometry/inventory.json")
    assets = {}
    scenes = {}
    anchors = {}
    for key, row in measured.items():
        folder = bundle / "assets/pothole" / key
        glb = digest(folder / "asset.glb")
        ply = digest(folder / "asset.ply")
        require(glb == hashes[key]["sha256"]["asset.glb"] == row["glb_sha256"], key + " GLB identity")
        require(ply == hashes[key]["sha256"]["asset.ply"], key + " PLY identity")
        grid = geometry[key]
        require(
            grid["mesh_sha256"] == glb and grid["grid_sha256"] == digest(sensors / grid["file"]),
            key + " simulated sensor surface identity",
        )
        assets[key] = dict(glb_sha256=glb, ply_sha256=ply, sites=[])
    for f in sorted((bundle / "scenes").glob("*/scene.json")):
        scene = read(f)
        site, case = scene["site"], scene["id"]
        (place,) = scene["placements"]
        key = place["asset_id"]
        scans = read(sensors / "cases" / (case + ".json"))
        (hit,) = scans["assets"]
        require(hit["asset_id"] == key and scans["scene_sha256"] == digest(f), case + " scan identity")
        require(scans["delta_sha256"] == digest(sensors / "cases" / (case + ".npz")), case + " scan delta")
        require(
            place["asset_ply_sha256"] == assets[key]["ply_sha256"] and place["size_multiplier"] == 1,
            case + " metric placement",
        )
        require(
            scene["checkpoint_reloaded"] and place["reload_max_pixel_error"] == 0, case + " native reload"
        )
        files = read(f.parent / "complete.json")["files"]
        for name, sha in files.items():
            require(digest(f.parent / name) == sha, case + " incomplete output: " + name)
        pose = [place[k] for k in ("center_world", "basis_columns", "slot", "source_reference_frame")]
        require(anchors.setdefault(site, pose) == pose, "Different road anchor at " + site)
        photos = read(sensors / "camera" / case / "views.json")
        visible = {view["camera"]: view["frame"] for view in photos if view["available"]}
        require("front_camera" in visible and hit["hits"] > 0, case + " missing observations")
        assets[key]["sites"].append(site)
        scenes[case] = dict(
            asset=key,
            native_checkpoint_reloaded=True,
            size_multiplier=place["size_multiplier"],
            reload_max_pixel_error=place["reload_max_pixel_error"],
            frames=scans["frames"],
            simulated_returns_by_lidar=hit["by_sensor"],
            camera_frames=visible,
        )
    require(all(a["sites"] for a in assets.values()), "Missing road placement")
    require(
        set(ray_test["per_asset"]) == set(assets) and not ray_test["nearest_part_disagreements"],
        "Independent mesh intersection check",
    )
    require(sum(s["frames"] for s in scenes.values()) == laser["counts"]["case_frames"], "Scan frame count")
    require(sum(len(s["camera_frames"]) for s in scenes.values()) == cameras["views"], "Camera count")
    return dict(
        assets=assets,
        scenes=scenes,
        common_site_anchors=anchors,
        scans=laser,
        cameras=cameras,
        independent_geometry=ray_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(check(args.run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
