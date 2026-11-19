"""Read native StreetCrafter PandaSet preprocessing with calibrated camera poses."""
import json
import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from street_gaussian.config import cfg
from street_gaussian.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm
from street_gaussian.utils.data_utils import get_val_frames
from street_gaussian.utils.graphics_utils import focal2fov
from street_gaussian.utils.novel_view_utils import pandaset_novel_view_cameras
from data_processor.pandaset_processor.pandaset_helpers import (
    load_camera_info, load_track, get_obj_info, PANDA_ID2CAMERA,
    PANDA_NONRIGID_DYNAMIC_CLASSES,
)


def readPandasetInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):
    poses, intrinsics = load_camera_info(path)
    _, instances = load_track(path)
    with open(os.path.join(path, 'timestamps.json')) as f:
        timestamps = json.load(f)
    start, end = cfg.data.selected_frames
    cams = cfg.data.cameras
    nframes = end - start + 1
    offset = min(timestamps[PANDA_ID2CAMERA[c]][start] for c in cams)
    train_frames, test_frames = get_val_frames(
        nframes, test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )
    object_info = {}
    for key, item in instances.items():
        ann = item['frame_annotations']
        frames = np.asarray(ann['frame_idx'])
        selected = (frames >= start) & (frames <= end)
        if selected.sum() < 2 or all(ann['stationary']):
            continue
        sizes = np.asarray(ann['box_size'])[selected].max(axis=0)
        # Keep objects that project into at least one selected camera.
        visible = False
        for frame, pose, size in zip(frames[selected], np.asarray(ann['obj_to_world'])[selected], np.asarray(ann['box_size'])[selected]):
            signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
            corners = signs * size / 2
            world = corners @ pose[:3, :3].T + pose[:3, 3]
            for cam in cams:
                w2c = np.linalg.inv(poses[frame, cam])
                p = world @ w2c[:3, :3].T + w2c[:3, 3]
                uv = p @ intrinsics[cam].T
                uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)
                visible |= bool(((p[:, 2] > .1) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)).any())
            if visible:
                break
        if not visible:
            continue
        oid = int(key)
        object_info[oid] = dict(
            id=len(object_info), object_id=oid, track_id=f'{oid:03d}',
            **{'class': item['class_name']}, class_label=0,
            deformable=item['class_name'] in PANDA_NONRIGID_DYNAMIC_CLASSES,
            length=float(sizes[0] * cfg.data.box_scale),
            width=float(sizes[1] * cfg.data.box_scale), height=float(sizes[2]),
            start_frame=int(frames[selected].min()), end_frame=int(frames[selected].max()),
        )
    tracklets = np.full((6, nframes, max(len(object_info), 1), 8), -1., dtype=np.float32)
    for oid, info in object_info.items():
        ann = instances[str(oid)]['frame_annotations']
        for cam in cams:
            for frame in range(start, end + 1):
                if not info['start_frame'] <= frame <= info['end_frame']:
                    continue
                sample = get_obj_info(ann, timestamps['front_camera'], timestamps[PANDA_ID2CAMERA[cam]][frame])
                if sample is None:
                    continue
                pose = sample['pose']
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                tracklets[cam, frame-start, info['id']] = np.r_[pose[:3, 3], quat, 1.]
    cam_infos = []
    for frame in range(start, end + 1):
        for cam in cams:
            name = f'{frame:03d}_{cam}'
            image_path = os.path.join(path, images, name + '.jpg')
            image = Image.open(image_path)
            width, height = image.size
            K = intrinsics[cam].copy()
            c2w = poses[frame, cam]
            w2c = np.linalg.inv(c2w)
            metadata = dict(frame=frame, frame_idx=frame-start, cam=cam,
                ego_pose=np.eye(4), extrinsic=c2w.copy(),
                timestamp=timestamps[PANDA_ID2CAMERA[cam]][frame]-offset,
                is_novel_view=False, is_val=frame-start in test_frames,
                guidance_rgb_path=os.path.join(path, 'lidar_forward/color_render', name+'.png'),
                guidance_mask_path=os.path.join(path, 'lidar_forward/color_render', name+'_mask.png'))
            guidance = {}
            for key, subdir in [('obj_bound', 'dynamic_masks/all'), ('sky_mask', 'sky_masks')]:
                p = os.path.join(path, subdir, name+'.png')
                if os.path.exists(p):
                    guidance[key] = Image.fromarray(np.asarray(Image.open(p).convert('L')) > 0)
            depth_path = os.path.join(path, 'lidar_forward/depth', name+'.npz')
            if cfg.mode == 'train' and os.path.exists(depth_path):
                with np.load(depth_path) as depth:
                    mask = depth['mask'].astype(bool)
                    value = np.zeros(mask.shape, np.float32)
                    value[mask] = depth['value']
                    guidance['lidar_depth'] = value
            cam_infos.append(CameraInfo(uid=len(cam_infos), R=w2c[:3, :3].T,
                T=w2c[:3, 3], FovY=focal2fov(K[1, 1], height),
                FovX=focal2fov(K[0, 0], width), K=K, image=image,
                image_path=image_path, image_name=name, width=width, height=height,
                metadata=metadata, guidance=guidance))
    train = [c for c in cam_infos if c.metadata['frame_idx'] in train_frames]
    test = [c for c in cam_infos if c.metadata['frame_idx'] in test_frames]
    novel = pandaset_novel_view_cameras(cam_infos, poses, object_info, tracklets)
    norm = getNerfppNorm(train)
    cfg.data.extent = float(cfg.data.extent or max(norm['radius'], 10.))
    metadata = dict(camera_tracklets=tracklets, obj_meta=object_info,
        num_images=len(cam_infos), num_cams=len(cams), num_frames=nframes,
        ego_frame_poses=poses[:, 0], scene_center=norm['center'], scene_radius=cfg.data.extent,
        camera_timestamps={c: [t-offset for t in timestamps[PANDA_ID2CAMERA[c]][start:end+1]] for c in cams})
    print(f'PandaSet: {len(train)} train / {len(test)} test cameras, {len(object_info)} moving objects')
    return SceneInfo(train_cameras=train, test_cameras=test, metadata=metadata, novel_view_cameras=novel)
