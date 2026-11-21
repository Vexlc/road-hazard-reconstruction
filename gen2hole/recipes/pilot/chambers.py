"""Fit a circular or four-sided chamber and hinged/displaced covers to image landmarks."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter

from mesh_io import grid_faces, lookup, luminance, road_atlas, smooth, source


def contour(theta, shape):
    u, v = np.cos(theta), np.sin(theta)
    if shape == 'rect':
        a = np.maximum(abs(u),abs(v))
        u,v = u/a,v/a
    return u,v


def sheet_uv(theta, levels):
    a,b = np.meshgrid(theta/(2*np.pi), 1-np.asarray(levels))
    return np.stack([a,b],-1)


def sheet(mesh, name, xyz, uv, material, inward=True, road=False):
    mesh.add(name, xyz, grid_faces(*xyz.shape[:2],flip=inward), uv, material, road)


def image_at_outline(img, outer, inner, u, v, t):
    params = np.array(outer)[None,:]*(1-t[:,None])+np.array(inner)[None,:]*t[:,None]
    x = params[:,0,None] + u[None,:]*params[:,2,None]
    y = params[:,1,None] - v[None,:]*params[:,3,None]
    return lookup(img,x,y)


def chamber_wall(mesh, key, cfg, layout, sx, sy, ix, iy, theta):
    donor = cfg.get('wall_source',key)
    spec, img = layout[donor],source(donor)
    shape = cfg['shape']
    tx = np.linspace(0,2*np.pi,1537)
    tu,tv = contour(tx,shape)
    # Near wall is foreshortened in the survey image. Continue the observed wall
    # around its hidden half without sampling the road rim or the water at the bottom.
    tv = np.abs(tv)
    levels = np.linspace(0,1,641)
    rgb = image_at_outline(img,spec['inner'],spec['wall_end'],tu,tv,levels)
    row = np.median(luminance(rgb),axis=1)
    target = np.percentile(row[18:160],75)
    gain = np.clip((max(target,.25)/(gaussian_filter(row,14)+.008))**.66,.94,3.30)
    wall_img = np.uint8(np.clip(rgb*gain[:,None,None],0,255))
    if key=='brick':
        # Repeat the observed courses at masonry scale down the inferred shaft.
        # Shift azimuth at each join and blend across a narrow band of mortar.
        src_row=wall_img.copy()
        q=levels*2.85
        phase=q%1
        band=np.floor(q).astype(int)
        xx=np.broadcast_to(np.linspace(0,1,len(tx))[None,:],wall_img.shape[:2])
        def sweep(k,p):
            uu=(xx+.177*k[:,None])%1.
            vv=np.broadcast_to((.025+.90*p)[:,None],uu.shape)
            return lookup(src_row,uu,vv,scale=1.)
        first=sweep(band,phase);second=sweep(band-1,.90+.10*np.minimum(phase/.075,1.))
        blend=smooth(phase/.075);blend[band==0]=1.
        wall_img=np.uint8(first*blend[:,None,None]+second*(1-blend[:,None,None]))
    wall = mesh.material('Exposed brick and mortar' if key=='brick' else 'Concrete shaft lining', wall_img,.94,0.,2.5)
    n = 145 if key=='brick' else 83
    zt=np.linspace(0,1,n)
    uv=sheet_uv(theta,zt)
    u,v=contour(theta,shape)
    detail = luminance(wall_img)
    cell = lookup(detail-gaussian_filter(detail,5),uv[:,:,0],1-uv[:,:,1],scale=1.)
    radius = 1+.039*smooth(zt/.13)[:,None] + np.clip(cell,-.26,.26)*(.034 if key=='brick' else .010)
    radius[0] = 1
    radius[-1] = 1.039
    z = np.broadcast_to(-.060-zt[:,None]*(cfg['depth']-.060),radius.shape)
    xyz=np.stack([u[None,:]*ix*radius,v[None,:]*iy*radius,z],-1)
    sheet(mesh,'Inspection shaft',xyz,uv,wall)
    # Register the bottom to the visible dark core; use a single disk/quad fan.
    fw=256
    xx,yy=np.meshgrid(np.linspace(-1,1,fw),np.linspace(1,-1,fw))
    bx,by,brx,bry=spec['bottom_photo']
    floor_tex=lookup(img,bx+xx*brx,by-yy*bry)
    bottom=mesh.material('Sediment at shaft base',np.uint8(np.clip(floor_tex*1.35+8,0,255)),.96,0.,2.)
    px=np.r_[0,ix*1.039*u]; py=np.r_[0,iy*1.039*v]
    face=np.c_[np.zeros(len(theta)-1,int),np.arange(1,len(theta)),np.arange(2,len(theta)+1)]
    mesh.add('Shaft bottom',np.c_[px,py,np.full(len(px),-cfg['depth'])],face,np.c_[.5+.5*px/(ix*1.039),.5+.5*py/(iy*1.039)],bottom)


def circular_layer(mesh, name, radius_x, radius_y, levels, relief, mat, mapper, square=False, downward=False):
    theta=np.linspace(0,2*np.pi,385)
    r=np.linspace(0,1,levels)
    u,v=contour(theta,'rect' if square else 'round')
    x,y=r[:,None]*u, r[:,None]*v
    uv=np.stack([.5+.5*x,.5+.5*y],-1)
    xyz=mapper(radius_x*x,radius_y*y,relief(x,y))
    sheet(mesh,name,xyz,uv,mat,inward=downward)
    return x[-1], y[-1]


def hinge(mesh, name, start, end, radius, mat):
    start,end=np.array(start),np.array(end)
    axis=end-start; axis=axis/np.linalg.norm(axis)
    t=np.cross(axis,[0.,0.,1.]); t/=np.linalg.norm(t)
    b=np.cross(axis,t)
    angles=np.linspace(0,2*np.pi,33)
    radial=radius*(t*np.cos(angles)[:,None]+b*np.sin(angles)[:,None])
    xyz=np.stack([start+radial,end+radial])
    # Row direction is +axis, angle increases in the t,b plane.
    sheet(mesh,name,xyz,sheet_uv(angles,[0,1]),mat,inward=True)
    for j,centre in enumerate([start,end]):
        vtx=np.concatenate([centre[None],centre+radial])
        face=np.c_[np.zeros(32,int),np.arange(1,33),np.arange(2,34)]
        if j==0:face=face[:,::-1]
        mesh.add(name+' end '+str(j),vtx,face,np.vstack([[.5,.5],.5+.49*np.c_[np.cos(angles),np.sin(angles)]]),mat)


def lid_model(mesh,key,cfg,img,szx,szy):
    lid=cfg['lid']
    kind=lid['type']; rect=kind=='end_hinge'
    c=cfg['outer']; inner=cfg['inner']
    rx,ry=(lid['width']/2,lid['length']/2) if rect else (lid['diameter']/2,)*2
    if rect:
        tl,tr,br,bl=np.asarray(lid['quad'])
        measured=(np.linalg.norm(bl-tl)+np.linalg.norm(br-tr))*.5/(2*inner[3])
        angle=math.acos(np.clip(measured,.035,.99))
        def top_uv(x,y):
            q=(x+1)*.5; p=(1-y)*.5
            return (1-p)[...,None]*((1-q)[...,None]*tl+q[...,None]*tr)+p[...,None]*((1-q)[...,None]*bl+q[...,None]*br)
    else:
        cx,cy,ex,ey=lid['ellipse']
        hinge_x=(cx-ex-c[0])*szx/c[2]
        hinge_y=(c[1]-cy)*szy/c[3]
        cos_est=(ex/c[2])/(ey/c[3])
        angle=np.arccos(np.clip(cos_est,.035,.99)) if kind=='side_hinge' else 0.
        def top_uv(x,y):
            return np.stack([cx+ex*x+lid.get('bow_x',0)*y*y,cy-ey*y],-1)
    thick=lid['thickness']; dz=.012
    def mapper(x,y,z):
        if rect:
            t=ry-y
            return np.stack([x,ry-t*np.cos(angle)+z*np.sin(angle),dz+t*np.sin(angle)+z*np.cos(angle)],-1)
        if kind=='displaced':
            dx=(lid['ellipse'][0]-c[0])*szx/c[2]
            dy=(c[1]-lid['ellipse'][1])*szy/c[3]
            return np.stack([dx+x,dy+y,thick+.010+z],-1)
        t=rx+x
        return np.stack([hinge_x+t*np.cos(angle)-z*np.sin(angle),hinge_y+y,dz+t*np.sin(angle)+z*np.cos(angle)],-1)
    xx, yy = np.meshgrid(np.linspace(-1,1,1024),np.linspace(1,-1,1024))
    pix=top_uv(xx*.985,yy*.985)
    face_img=lookup(img,pix[:,:,0],pix[:,:,1])
    top=mesh.material('Worn metal tread',face_img,.70 if key!='offset' else .74,.51,2.4)
    ln=luminance(face_img)
    metal_relief=gaussian_filter(ln,1.2)-gaussian_filter(ln,8.)
    def upper(x,y):
        out=np.clip(lookup(metal_relief,.5+.5*x,.5-.5*y,scale=1.)*.014,-.0023,.0033)
        if key=='raised':
            for h in [-.65,.73]:
                out-=.004*smooth(1-((x-.105)/.066)**4-((y-h)/.083)**4)
        elif key=='offset':
            out-=.005*smooth(1-(x/.12)**4-((y+.73)/.043)**4)
        return out
    eu,ev=circular_layer(mesh,'Lid face',rx,ry,106,upper,top,mapper,rect)
    back=source('back')
    if rect:
        backside=lookup(source('squareback'),240.5+xx*211.7,236.0-yy*211.0)
    else:
        backside=lookup(back,237+xx*222,239-yy*225)
    backmat=mesh.material('Sand-cast underside',backside,.85,.36,2.5)
    def underside(x,y):
        if rect:
            ribs=np.maximum.reduce([smooth((.063-abs(x-.554))/.026),smooth((.063-abs(x+.541))/.026),smooth((.055-abs(y-.016))/.025)])
            ribs=np.maximum(ribs,smooth((np.maximum(abs(x),abs(y))-.91)/.065))
        else:
            r=np.hypot(x,y); a=np.arctan2(y,x)
            arms=r*abs(np.sin(3*(a-np.pi/2)))/3
            ribs=np.maximum.reduce([smooth((.056-arms)/.026),smooth((.22-r)/.06),smooth((r-.89)/.07)])
        return -thick-(.022 if rect else .010)*ribs
    circular_layer(mesh,'Lid underside with casting ribs',rx,ry,81,underside,backmat,mapper,rect,True)
    tx,ty=np.meshgrid(np.linspace(-1,1,512),np.linspace(-1,1,96))
    rust=lookup(back,238+tx*123,20+ty*3.2+4*tx*tx)
    sidemat=mesh.material('Edge and hinge pin iron',rust,.79,.49,2.)
    edge_top=upper(eu,ev)
    z=np.stack([edge_top,edge_top-.0015,np.full_like(eu,-thick*.80),underside(eu,ev)])
    shrink=np.array([1.,1.0008,1.,1.])[:,None]
    xyz=mapper(rx*eu[None,:]*shrink,ry*ev[None,:]*shrink,z)
    # Outward side faces, from top to bottom.
    sheet(mesh,'Lid rolled edge',xyz,sheet_uv(np.linspace(0,2*np.pi,len(eu)),[0,.08,.72,1]),sidemat,inward=False)
    if rect:
        for i,xcenter in enumerate([-rx*.72,rx*.72]):
            hinge(mesh,'Hinge knuckle '+str(i),[xcenter-.036,ry,.014],[xcenter+.036,ry,.014],.022,sidemat)
    elif kind=='side_hinge':
        for i,(y0,y1) in enumerate([(-.095,-.049),(-.045,.045),(.049,.095)]):
            hinge(mesh,'Hinge knuckle '+str(i),[hinge_x,y0+hinge_y,.012],[hinge_x,y1+hinge_y,.012],.018,sidemat)
        # Two short stationary lugs bridge the photographed pivot and the fixed iron seat.
        for i,ym in enumerate([hinge_y-.070,hinge_y+.070]):
            x0=-szx*.983;x1=hinge_x
            a=np.array([[x0,ym-.016,-.009],[x1,ym-.016,-.009],[x1,ym+.016,-.009],[x0,ym+.016,-.009]])
            b=a.copy();b[:,2]+=.025
            verts=np.concatenate([a,b])
            face=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
            mesh.add('Hinge frame lug '+str(i),verts,face,np.tile([[.2,.2],[.8,.2],[.8,.8],[.2,.8]],(2,1)),sidemat)
    return float(np.rad2deg(angle))


def build(mesh, key, cfg, layout, grain_atlas):
    img=source(key)
    shape=cfg['shape']
    sx,sy=cfg['width']/2,cfg['length']/2
    cx,cy,ex,ey=cfg['outer']
    ix,iy=sx*cfg['inner'][2]/ex,sy*cfg['inner'][3]/ey
    collar=cfg.get('collar',1.)
    extent=max(sx,sy)*max(1.87,collar+.58)
    if key=='offset':
        lid_x=(cfg['lid']['ellipse'][0]-cx)*sx/ex
        lid_y=(cy-cfg['lid']['ellipse'][1])*sy/ey
        lid_r=cfg['lid']['diameter']/2
        extent=max(extent,lid_x+lid_r+.24)
    a=np.linspace(-extent,extent,1520,dtype='f4')
    x,y=np.meshgrid(a,-a)
    px,py=cx+x*ex/sx,cy-y*ey/sy
    erase=None
    if 'lid' in cfg:
        lid=cfg['lid']
        if 'ellipse' in lid:
            lx,ly,rx,ry=lid['ellipse']; v=(ly-py)/ry
            q=np.hypot((px-lx-lid.get('bow_x',0)*v*v)/rx,v)
            erase=smooth((1.075-q)/.055)
        else:
            erase=smooth(np.minimum.reduce([px-35,446-px,py-37,232-py])/6.)
    roadimg=road_atlas(img,px,py,grain_atlas,erase)
    pavement=mesh.material('Road and joint mortar',roadimg,.89,0.,3.0)
    theta=np.linspace(0,2*np.pi,385)
    u,v=contour(theta,shape)
    squ,sqv=contour(theta,'rect')
    lum=luminance(roadimg); bump=gaussian_filter(lum,1.2)-gaussian_filter(lum,5.2)
    def road_band(name,outx,outy,inx,iny,rows,road):
        t=np.linspace(0,1,rows)[:,None]
        x=outx*(1-t)+inx*t; y=outy*(1-t)+iny*t
        uv=np.stack([.5+x/(2*extent),.5+y/(2*extent)],-1)
        z=lookup(bump,uv[:,:,0],1-uv[:,:,1],scale=1.)*(.0078 if collar>1 else .004)
        z[0]=0;z[-1]=0
        sheet(mesh,name,np.stack([x,y,z],-1),uv,pavement,road=road)
    road_band('Road around the aperture',extent*squ,extent*sqv,sx*collar*u,sy*collar*v,43,True)
    if collar>1:
        road_band('Chipped concrete collar',sx*collar*u,sy*collar*v,sx*u,sy*v,38,False)
    tx=np.linspace(0,2*np.pi,1537)
    tu,tv=contour(tx,shape)
    if key=='raised':tu=abs(tu)
    if key=='hatch':tv=-abs(tv)
    rt=np.linspace(0,1,146)
    frame_rgb=image_at_outline(img,cfg['outer'],cfg['inner'],tu,tv,rt)
    # The vertical throat has little area in a nadir photograph. Map its side
    # in independent horizontal and depth coordinates using the supplementary iron.
    rux,ruy=np.meshgrid(np.linspace(0,1,len(tx)),rt)
    iron=source('back')
    mx=(rux*8.)%2;mx=1-abs(mx-1)
    deeper=lookup(iron,342+mx*57,124+(ruy-.5)*76)
    deeper=np.uint8(np.clip(deeper.astype('f4')*.70,0,255))
    blend=smooth((rt-.56)/.30)[:,None,None]
    frame_rgb=np.uint8(frame_rgb*(1-blend)+deeper*blend)
    rim=mesh.material('Cast iron frame and seat',frame_rgb,.73,.59,1.8)
    blend=np.array([0,.025,.10,.22,.43,.49,.62,.83,.92,1,1])
    z=np.array([0,.002,.005,.006,.005,.002,-.010,-.012,-.013,-.020,-.060])
    rxx=(1-blend)*sx+blend*ix; ryy=(1-blend)*sy+blend*iy
    rz=np.broadcast_to(z[:,None],(len(z),len(theta)))
    xyz=np.stack([rxx[:,None]*u,ryy[:,None]*v,rz],-1)
    sheet(mesh,'Cast ring and cover seat',xyz,sheet_uv(theta,np.array([0,.03,.12,.24,.42,.50,.60,.78,.89,.96,1])),rim)
    chamber_wall(mesh,key,cfg,layout,sx,sy,ix,iy,theta)
    angle=None
    if 'lid' in cfg:
        angle=lid_model(mesh,key,cfg,img,sx,sy)
    a=np.linspace(-extent,extent,321,dtype='f4')
    x,y=np.meshgrid(a,-a)
    def radius(a,b):
        return np.hypot(a,b) if shape=='round' else np.maximum(abs(a),abs(b))
    signed=radius(x/sx,y/sy)
    alpha=smooth(((collar+.35)-signed)*min(sx,sy)/.092)
    if key=='offset':
        alpha=np.maximum(alpha,smooth((lid_r+.125-np.hypot(x-lid_x,y-lid_y))/.092))
    aperture=radius(x/ix,y/iy)<1.
    h=np.where(aperture,-cfg['depth'],0)
    field={'height':h,'alpha':alpha,'width':extent*2,'length':extent*2,'defect_mask':(signed<collar),'opening_mask':aperture}
    result={'width_m':ix*2,'length_m':iy*2,'frame_width_m':sx*2,'frame_length_m':sy*2,'depth_design_m':cfg['depth'],
            'exponent':2. if shape=='round' else 16.,'state':cfg.get('lid',{}).get('type','open'),
            'geometry_source':'image-registered rim and aperture, photometric wall unwrapping and fitted solid lid',
            'road_projection':[cx,cy,ex/sx,ey/sy,img.shape[0]],'hidden_wall_reference':cfg.get('wall_source',key)}
    if angle is not None:
        result['lid_angle_degrees']=angle
        result['lid_thickness_m']=cfg['lid']['thickness']
        if key=='raised':result['hinge_fit']='pivot XY recovered from leftmost lid landmark and frame scale; two fixed lugs bridge the inner seat'
        result['unseen_underside_reference']='squareback' if key=='hatch' else 'back'
    return field,result
