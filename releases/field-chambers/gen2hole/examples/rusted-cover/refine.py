"""Refine the sampled steel only at aperture edges and preserve its traced connectivity."""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def sample(image, uv):
    return map_coordinates(image,uv[:,::-1].T,order=1,mode="nearest")


def subdivide(uv, faces, near):
    edges = np.stack((faces,np.roll(faces,-1,axis=1)),axis=2)
    lo,hi = edges.min(2),edges.max(2)
    base = len(uv)
    codes = lo.astype("i8")*base+hi
    marked = np.unique(codes[near])
    which = np.searchsorted(marked,codes).clip(0,len(marked)-1)
    split = marked[which]==codes
    count = split.sum(1)
    mid = which+base
    output = [faces[count==0]]
    for n in (1,2):
        sel=count==n
        start=(split[sel] if n==1 else np.roll(~split[sel],1,axis=1)).argmax(1)
        order=(start[:,None]+np.arange(3))%3
        a,b,c=np.take_along_axis(faces[sel],order,axis=1).T
        d,e,_=np.take_along_axis(mid[sel],order,axis=1).T
        if n==1:
            output.extend([np.c_[a,d,c],np.c_[d,b,c]])
        else:
            output.extend([np.c_[a,d,c],np.c_[d,e,c],np.c_[d,b,e]])
    a,b,c=faces[count==3].T
    d,e,f=mid[count==3].T
    output.extend([np.c_[a,d,f],np.c_[d,b,e],np.c_[f,e,c],np.c_[d,e,f]])
    return np.vstack([uv,(uv[marked//base]+uv[marked%base])*.5]),np.vstack(output)


def boundary_mesh(uv,faces,signed,seeds,steps):
    for stage in range(steps):
        level=sample(signed,uv)
        # Refine complete triangles. A neighbour splits each shared edge too.
        band = np.abs(level[faces]).min(1)<6./(stage+1)
        band |= (level[faces].min(1)*level[faces].max(1))<=0
        uv,faces=subdivide(uv,faces,band)
    level=sample(signed,uv)
    level=np.where(abs(level)<.002,np.where(level>=0,.002,-.002),level)
    # Some dark flecks have only a diagonal contact in the thresholded photo.
    # Keep void components with a documented seed; the others retain steel.
    i,j=faces.ravel(),np.roll(faces,-1,axis=1).ravel()
    for negative in (True,False):
        active=(level<0) if negative else (level>0)
        linked=active[i]&active[j]
        graph=coo_matrix((np.ones(linked.sum(),"u1"),(i[linked],j[linked])),shape=(len(uv),len(uv)))
        _,groups=connected_components(graph,directed=False,return_labels=True)
        if negative:
            ids=np.flatnonzero(active)
            nearest=cKDTree(uv[active]).query(seeds)[1]
            keep=groups[ids[nearest]]
            if len(set(keep))!=len(seeds):
                raise ValueError("Distinct documented apertures merged")
        else:
            keep=[groups[0]]
        flip=active&~np.isin(groups,keep)
        level[flip]*=-1
    return uv,faces,level
