"""Bake steep triangles in surface space to remove the stretch of a nadir projection."""
import numpy as np
from scipy.ndimage import gaussian_filter

from mesh_io import lookup, smooth, source


def side_weights(height, x_step):
    dy,dx=np.gradient(gaussian_filter(height,.78),x_step,x_step)
    return smooth((np.hypot(dx,dy)-.56)/1.25)*smooth((-height-.002)/.014)


def add_side(mesh, xyz, faces, uv, atlas, slope, key):
    tsize,pad=16,2
    side=int(np.ceil(np.sqrt(len(faces))))
    w=side*tsize
    tex=np.zeros((w,w,3),'u1')
    ij=np.arange(tsize)
    px,py=np.meshgrid(ij,ij)
    b=(px-pad)/(tsize-1-2*pad);c=(py-pad)/(tsize-1-2*pad)
    weights=np.stack([1-b-c,b,c],-1).astype('f4')
    face_uv=np.empty((len(faces),3,2),'f4')
    macro=source('fracture')
    low=gaussian_filter(atlas.astype('f4'),(10,10,0))
    for start in range(0,len(faces),256):
        subset=faces[start:start+256]
        q=xyz[subset]
        local=np.einsum('ijc,nck->nijk',weights,q)
        projected=np.einsum('ijc,nck->nijk',weights,uv[subset])
        blend=np.clip(np.einsum('ijc,nc->nij',weights,slope[subset]),0,1)[...,None]
        norm=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);weight=abs(norm[:,:2])**4
        weight/=np.maximum(weight.sum(1,keepdims=True),1e-18)
        x,y,z=np.moveaxis(local,-1,0)
        tx=lookup(macro,(x/.235+.3)*479,(-z/.235+.21)*479,mode='mirror')
        ty=lookup(macro,(y/.235+.57)*479,(-z/.235+.39)*479,mode='mirror')
        grit=weight[:,0,None,None,None]*ty+weight[:,1,None,None,None]*tx
        orig=lookup(atlas,projected[...,0],1-projected[...,1],scale=1.)
        base=lookup(low,projected[...,0],1-projected[...,1],scale=1.)
        if key=='spall':
            rock=(.58+grit.mean(-1,keepdims=True)/235.)*base*np.array([1.02,.995,.96])
        else:
            bias=.67 if key=='wet' else .87
            rock=grit*bias*np.clip((base.mean(-1,keepdims=True)+10)/97.,.53,1.22)
        color=np.uint8(np.clip(blend*rock+(1-blend)*orig,0,255))
        for j,block in enumerate(color):
            k=start+j; ox=(k%side)*tsize; oy=(k//side)*tsize
            tex[oy:oy+tsize,ox:ox+tsize]=block
            face_uv[k,:,0]=(ox+np.array([pad,tsize-1-pad,pad])+.5)/w
            face_uv[k,:,1]=1-(oy+np.array([pad,pad,tsize-1-pad])+.5)/w
    mat=mesh.material('Photographic exposed fracture face',tex,.83 if key=='wet' else .92,0.,1.05)
    vertices=xyz[faces].reshape(-1,3)
    mesh.add('Fracture sidewall in surface coordinates',vertices,np.arange(len(vertices)).reshape(-1,3),face_uv,mat)
