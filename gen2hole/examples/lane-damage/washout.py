"""Two surviving road ends and the exposed embankment of the River Muick washout."""
import numpy as np
from photo_mesh import RoadCamera, face_up, grid_faces, texture_uv, xyz


def resample(edge, count):
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(edge, axis=0), axis=1))]
    along = np.linspace(0., lengths[-1], count)
    return np.stack([np.interp(along, lengths, col) for col in edge.T], axis=-1)


def make(mesh, img, s):
    cam = RoadCamera(s)
    n = 101
    near, far = [resample(cam.ground(s[k]), n) for k in ('near_lip', 'far_lip')]
    scale = s['major_m'] / np.linalg.norm(far[n//2] - near[n//2])
    cam.eye *= scale
    near *= scale
    far *= scale
    mat = mesh.material('River Muick field photograph', img, .95, 0, .45)
    def add_strip(name, coords, towards=None):
        verts = coords.reshape(-1, 3)
        faces = grid_faces(*coords.shape[:2])
        if towards is None:
            faces = face_up(verts, faces)
        else:
            a,b,c = verts[faces].transpose(1,0,2)
            flip = (np.cross(b-a,c-a) @ towards) < 0
            faces[flip] = faces[flip,::-1]
        uv = texture_uv(cam.project(verts), s['size'])
        return verts, faces, uv

    def merge(name, strips):
        vv, ff, uu = [], [], []
        for vertices, faces, uv in strips:
            ff.append(faces+sum(map(len, vv))); vv.append(vertices); uu.append(uv)
        mesh.add(name, np.vstack(vv), np.vstack(ff), np.vstack(uu), mat)

    road = []
    bounds = [near,far]
    for lip, key in [(near,'near_approach'),(far,'far_approach')]:
        end = resample(cam.ground(s[key]), n)
        bounds.append(end)
        road.append(add_strip('pavement', xyz(np.linspace(lip,end,29), 0.)))
    merge('Detached pavement top', road)
    t,u = np.meshgrid(np.linspace(0,1,131),np.linspace(0,1,n),indexing='ij')
    xy = (1-t[...,None])*near+t[...,None]*far
    x,y = xy.transpose(2,0,1)
    # The open river side drops away. Monocular imagery constrains the lips and texture,
    # while the hidden bank profile and small relief below are explicit assumptions.
    depth = .38 + 1.25*u + .19*np.sin(np.pi*t)
    rough = .045*np.sin(3*x+1.2*y)*np.cos(2*y)+.023*np.cos(4*y-2*x)
    bed = xyz(xy,-depth+rough)
    merge('Fracture lip and exposed bed',[add_strip('bed',bed)])
    walls=[]
    direction=far[n//2]-near[n//2]
    for lip, lower, inward in [(near,bed[0],direction),(far,bed[-1],-direction)]:
        layers=np.linspace(xyz(lip,0.),lower,21)
        walls.append(add_strip('cut face',layers,np.r_[inward,0.]))
    merge('Fracture sidewall in surface coordinates', walls)
    span=2*np.max(abs(np.vstack(bounds)),axis=0)+.1
    return span,dict(centerline_gap_model_m=s['major_m'],near_lip_xyz_m=xyz(near,0.).tolist(),
                     far_lip_xyz_m=xyz(far,0.).tolist(),approach_ray_starts_xy_m=[p.tolist() for p in
                     [(near[n//2]+bounds[2][n//2])/2,(far[n//2]+bounds[3][n//2])/2]],
                     source_camera_eye_m=cam.eye.tolist(),
                     reconstruction_scope='Both surviving pavement ends and exposed bank only. Gap length, eroded profile and hidden faces inferred. River excluded.')
