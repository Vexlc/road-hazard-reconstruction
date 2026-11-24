"""Render a calibrated fisheye by sampling overlapping perspective renders of the same scene."""
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from utils.graphics_utils import getProjectionMatrix


def calibration(camera_file):
    camera_id, model, width, height, *params = camera_file.read_text().split()
    if camera_id != '1' or model != 'OPENCV_FISHEYE' or len(params) != 8:
        raise ValueError('Expected a single calibrated OpenCV fisheye camera')
    return (int(width), int(height)), np.array(params, float)


def sensor_rays(x, y, params):
    fx, fy, cx, cy, *dist = params
    x, y = (x+.5-cx)/fx, (y+.5-cy)/fy
    radius = np.hypot(x, y)
    theta = radius.copy()
    for _ in range(10):
        projected = theta + sum(k * theta**(2*i+3) for i, k in enumerate(dist))
        slope = 1 + sum((2*i+3)*k*theta**(2*i+2) for i, k in enumerate(dist))
        theta -= (projected-radius)/slope
    scale = np.sin(theta)/np.maximum(radius, 1e-12)
    return np.stack((x*scale, y*scale, np.cos(theta)), axis=-1)


def tangent_patches(w, h, params):
    tw, th = 320, round(320*h/w)
    # Eight overlapping tangent images surround the central perspective. Fit
    # each inscribed rectangle to the source lens so every target pixel has RGB.
    fx, fy, cx, cy = params[:4]
    k = np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])
    grid_y, grid_x = np.mgrid[:th, :tw]
    def grid(rot, f, x, y):
        rays = np.stack(((x-(tw-1)/2)/f, (y-(th-1)/2)/f, np.ones_like(x)), axis=-1) @ rot.T
        radius = np.hypot(rays[...,0], rays[...,1])
        theta = np.arctan2(radius, rays[...,2])
        rd = theta + sum(v*theta**(2*i+3) for i, v in enumerate(params[4:]))
        return rays[..., :2] * (rd/np.maximum(radius,1e-12))[..., None] * [fx,fy] + k[:2,2]
    patches = []
    for ny in (.20,.5,.80):
        for nx in (.20,.5,.80):
            if nx == ny == .5:
                continue
            forward = sensor_rays(np.asarray(nx*w-.5), np.asarray(ny*h-.5), params)
            right = np.cross([0,1,0], forward)
            right /= np.linalg.norm(right)
            rot = np.column_stack((right, np.cross(forward,right), forward))
            low, high = tw*.15, tw*4.
            for _ in range(16):
                f = (low+high)/2
                # Include the outline. A nonlinear projection can reach its
                # largest radius along an edge between two of the corners.
                px = grid(rot, f, grid_x[::8,::8], grid_y[::8,::8])
                if np.any(px < 2) or np.any(px > [w-3,h-3]):
                    low = f
                else:
                    high = f
            f = high
            while True:
                px = grid(rot, f, grid_x, grid_y)
                if (px >= 0).all() and (px <= [w-1,h-1]).all():
                    break
                f *= 1.02
            patches.append((rot, px.astype('f4'), 2*np.arctan(tw/(2*f)), 2*np.arctan(th/(2*f))))
    return tw, th, patches


def tangent_views(train, source):
    (w, h), params = calibration(source/'sparse/0/cameras.txt')
    tw, th, patches = tangent_patches(w, h, params)
    views = []
    for cam in train:
        with Image.open(source/'images'/(cam.image_name+'.png')) as img:
            rgb = np.asarray(img.convert('RGB'))
        for n,(rot, pixels, fov_x, fov_y) in enumerate(patches):
            view = Image.fromarray(cv2.remap(rgb, pixels, None, cv2.INTER_LINEAR))
            views.append(cam._replace(image=view, image_name=f'{cam.image_name}_tile{n}',
                                      R=cam.R @ rot, T=rot.T @ cam.T, width=tw, height=th,
                                      FovX=fov_x, FovY=fov_y))
    return views


class FisheyeReplay:
    def __init__(self, camera_file=None, *, sensor=None, central=None):
        if sensor is None:
            sensor = calibration(camera_file)
            pin = camera_file.parents[3] / 'all/sparse/0/cameras.txt'
            _, _, pw, ph, pfx, pfy, *_ = pin.read_text().split()
            central = (int(pw), int(ph), float(pfx), float(pfy))
        (width, height), params = sensor
        y, x = np.mgrid[:height, :width]
        rays = sensor_rays(x, y, params)
        self.shape = (height, width, 3)
        tw, th, patches = tangent_patches(width, height, params)
        lenses = [(rot, tw, th, fov_x, fov_y) for rot, _, fov_x, fov_y in patches]
        pw, ph, pfx, pfy = central
        lenses.append((np.eye(3), pw, ph, 2*np.arctan(pw/(2*float(pfx))), 2*np.arctan(ph/(2*float(pfy)))))
        projections, extents = [], []
        for rot, w, h, fov_x, fov_y in lenses:
            points = rays @ rot
            xy = points[..., :2] / np.maximum(points[..., 2:], .001)
            extent = np.max(np.abs(xy) / np.tan(np.array([fov_x, fov_y])/2), axis=-1)
            extent[points[..., 2] <= 0] = np.inf
            projections.append(xy)
            extents.append(extent)
        # Extend the image plane around each tangent optical axis just enough
        # to include sensor corners. Keep its original focal length in pixels.
        cover = max(1.05, float(np.min(extents, axis=0).max()) + .12)
        if not np.isfinite(cover):
            raise ValueError('No perspective can see one or more fisheye rays')
        self.faces = []
        self.weight_sum = np.zeros((height, width), dtype=np.float32)
        for (rot,w,h,fov_x,fov_y), xy, extent in zip(lenses, projections, extents):
            fx, fy = np.array([w,h]) / (2*np.tan(np.array([fov_x,fov_y])/2))
            sw, sh = int(np.ceil(w*cover))+2, int(np.ceil(h*cover))+2
            fov_x, fov_y = 2*np.arctan([sw/(2*fx),sh/(2*fy)])
            weight = (1-np.minimum(extent/cover,1)**2)**2
            self.weight_sum += weight
            pixels = xy * [fx,fy] + [(sw-1)/2,(sh-1)/2]
            # remap only the rectangular region covered by this face
            yy, xx = np.where(weight > 0)
            roi = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rot
            proj = getProjectionMatrix(.01, 100., fov_x, fov_y).T.cuda()
            self.faces.append((torch.from_numpy(transform).cuda(), proj, sw, sh, fov_x, fov_y, roi,
                               weight[roi].astype('f4')[...,None], pixels[roi].astype('f4')))
        if self.weight_sum.min() <= 0:
            raise ValueError('Tangent views do not cover the calibrated sensor')

    def frame(self, cam, render_view):
        image = np.zeros(self.shape, dtype=np.float32)
        for rotation, proj, w, h, fov_x, fov_y, roi, weight, grid in self.faces:
            view = cam.world_view_transform @ rotation
            face = SimpleNamespace(FoVx=fov_x, FoVy=fov_y, image_width=w, image_height=h,
                                   world_view_transform=view, full_proj_transform=view @ proj,
                                   camera_center=cam.camera_center)
            rgb = render_view(face)
            if not torch.isfinite(rgb).all():
                raise ValueError(f'Invalid fisheye render at {cam.image_name}')
            rgb = rgb.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
            image[roi] += cv2.remap(rgb, grid, None, cv2.INTER_LINEAR) * weight
        return np.rint(image/self.weight_sum[...,None]).clip(0,255).astype('uint8')
