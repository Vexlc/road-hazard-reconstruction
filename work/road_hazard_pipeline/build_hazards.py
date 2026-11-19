"""Parametric road-hazard meshes using CC0 scanned surface materials.

Geometry includes actual excavated road surfaces, iron frames, shaft walls and
two-sided displaced covers. Missing structures are authored, not claimed as scans.
"""
from pathlib import Path
import argparse
import json,numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,gaussian_filter
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'work/road_hazard_sources';OUT=ROOT/'outputs/road_hazards'
OUT.mkdir(exist_ok=True)
TEX=OUT/'materials';TEX.mkdir(exist_ok=True)

MATERIALS={}
def prepare_material(name,source,tint,rough=.9,metal=0,scale=1.,gray=.0):
    d=SRC/source;dest=TEX/name;dest.mkdir(exist_ok=True)
    def find(key):return next(d.glob('*_'+key+'.png'))
    c=np.array(Image.open(find('Color')).convert('RGB'),dtype='f4')/255
    crop=None
    if name=='iron':
        crop=(int(c.shape[1]*.64),int(c.shape[0]*.45),int(c.shape[1]*.77),int(c.shape[0]*.60))
        c=np.array(Image.fromarray(np.uint8(c*255)).crop(crop).resize((2048,2048)),dtype='f4')/255
    if gray:c=(1-gray)*c+gray*(c@np.array([.2126,.7152,.0722]))[...,None]
    c=np.uint8(np.clip(c*np.array(tint),0,1)*255);Image.fromarray(c).save(dest/'color.png')
    for key,file in [('NormalGL','normal.png'),('Roughness','roughness.png')]:
        a=np.array(Image.open(find(key)));a=a.astype('f4')/(65535 if a.dtype.itemsize==2 else 255)
        if crop is not None:a=np.array(Image.fromarray(np.uint8(np.clip(a,0,1)*255)).crop(crop).resize((2048,2048)),dtype='f4')/255
        if key=='Roughness' and name=='iron':a=np.clip(a,.82,.98)
        if key=='Roughness' and name=='cover':a=np.clip(a,.55,.98)
        Image.fromarray(np.uint8(np.clip(a,0,1)*255)).save(dest/file)
    for p in dest.glob('*_2k.png'):p.unlink()
    MATERIALS[name]={'name':name,'color':str(dest/'color.png'),'normal':str(dest/'normal.png'),'roughness':str(dest/'roughness.png'),'roughness_factor':rough,'metallic':metal,'texture_scale_m':scale,'source':f'https://ambientcg.com/view?id={source}','license':'CC0-1.0'}

def sample_source(source,key,uv):
    a=np.asarray(Image.open(next((SRC/source).glob('*_'+key+'.png'))));a=a.astype('f4')/(65535 if a.dtype.itemsize==2 else 255)
    return map_coordinates(a,[(1-uv[...,1])*(a.shape[0]-1),uv[...,0]*(a.shape[1]-1)],order=1,mode='wrap')

class Mesh:
    def __init__(self):self.v=[];self.f=[];self.uv=[];self.mi=[];self.names=[];self.parts=[]
    def add(self,vertices,faces,uv,material,part=''):
        if material not in self.names:self.names.append(material)
        offset=sum(len(v) for v in self.v);self.v.append(np.asarray(vertices,dtype='f4'));self.f.append(np.asarray(faces,dtype='i4')+offset);self.uv.append(np.asarray(uv,dtype='f4'));self.mi.append(np.full(len(faces),self.names.index(material),dtype='i4'));self.parts.append(part or material)
    def arrays(self):return {'vertices':np.concatenate(self.v),'faces':np.concatenate(self.f),'uv':np.concatenate(self.uv),'face_material':np.concatenate(self.mi)}
    def save(self,path,meta):
        path.mkdir(exist_ok=True);np.savez_compressed(path/'mesh.npz',**self.arrays())
        (path/'materials.json').write_text(json.dumps([MATERIALS[n] for n in self.names],indent=2))
        meta.update(units='metres',axes='Z-up',construction='scanned PBR materials with authored hazard/shaft/cover geometry',gradient_reconstruction=False,parts=self.parts)
        (path/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        print('BUILT',path.name,sum(len(f) for f in self.f),flush=True)

def grid(n):
    ids=np.arange(n*n).reshape(n,n)
    return np.concatenate([np.stack([ids[:-1,:-1],ids[1:,:-1],ids[:-1,1:]],-1).reshape(-1,3),np.stack([ids[1:,:-1],ids[1:,1:],ids[:-1,1:]],-1).reshape(-1,3)])

def road(mesh,kind,seed=0,pit=None,hole=False,damage=False):
    n=513;extent=1.9;yy,xx=np.meshgrid(np.linspace(extent/2,-extent/2,n),np.linspace(-extent/2,extent/2,n),indexing='ij')
    rng=np.random.default_rng(seed);tex_scale=2.2 if kind=='asphalt' else 1.8
    uv=np.stack([xx/tex_scale+.5,yy/tex_scale+.5],-1)
    source='Asphalt031' if kind=='asphalt' else 'Concrete034'
    texture_h=sample_source(source,'Displacement',uv)
    height=(texture_h-np.median(texture_h))*(.006 if kind=='asphalt' else .003)
    theta=np.arctan2(yy,xx)
    radius=np.hypot(xx,yy)
    if hole:
        # Breakage in the road around the steel frame, rather than a stone paving border.
        ring=np.clip((.58-radius)/.16,0,1)*np.clip((radius-.39)/.05,0,1)
        height-=ring*(.006+(.008 if damage else .002)*(1+np.sin(theta*11+.5)))
    if pit is not None:
        rx,ry,depth=pit['rx'],pit['ry'],pit['depth']
        rr=np.sqrt((xx/rx)**2+(yy/ry)**2)/(1+.075*np.sin(3*theta+.8)+.045*np.sin(7*theta-1.2)+.025*np.cos(13*theta+seed)+.009*np.sin(37*theta)+.006*np.cos(71*theta))
        edge=np.clip((1-rr)/.065,0,1)
        lower=np.clip((.94-rr)/.28,0,1)
        cut=-depth*(.48*np.sqrt(edge)+.52*(lower*lower*(3-2*lower)))
        noise=gaussian_filter(rng.normal(size=(n,n)),.7)
        cut+=noise*.006*np.clip(edge,0,1)
        height+=cut
        if pit['variant']=='elongated':
            height-=np.clip((1.04-rr)/.2,0,1)*.015*np.sin(xx*24+yy*11)**2
        # Radial cracks terminate around the pit, instead of being painted black strokes.
        for angle in [.35,2.1,3.5,5.4]:
            along=xx*np.cos(angle)+yy*np.sin(angle);across=-xx*np.sin(angle)+yy*np.cos(angle)
            crooked=across-.012*np.sin(along*21+seed)
            crack=np.exp(-(crooked/.003)**2)*((rr>1)&(rr<1.5)&(along>0))
            height-=crack*.008
    vertices=np.stack([xx,yy,height],-1).reshape(-1,3);faces=grid(n);uv=uv.reshape(-1,2)
    if hole:faces=faces[np.mean(radius.reshape(-1)[faces],axis=1)>.402]
    if pit is not None:
        face_rr=np.mean(rr.reshape(-1)[faces],1)
        for select,material,part in [(face_rr>=1,kind,'unbroken road'),((face_rr<1)&(face_rr>.89),'fracture_'+kind,'fractured upper road layer'),(face_rr<=.89,'aggregate','exposed aggregate pit floor')]:
            mesh.add(vertices,faces[select],uv,material,part)
    elif hole:
        rad=np.mean(radius.reshape(-1)[faces],1)
        mesh.add(vertices,faces[rad>.51],uv,kind,'road around opening')
        mesh.add(vertices,faces[rad<=.51],uv,'fracture_'+kind,'damaged collar bedding')
    else:mesh.add(vertices,faces,uv,kind)
    return extent,height

def lathe(mesh,profile,material,segments=256,phase=0,part='',inward=False):
    t=np.linspace(0,2*np.pi,segments,endpoint=False);p=np.asarray(profile)
    x=p[:,0,None]*np.cos(t)[None];y=p[:,0,None]*np.sin(t)[None];z=np.broadcast_to(p[:,1,None],x.shape)
    if material=='iron':
        wear=.0007*np.sin(t*33+.7)+.0004*np.cos(t*61)
        x=x+wear*np.cos(t);y=y+wear*np.sin(t);z=z+wear*.4
    v=np.stack([x,y,z],-1).reshape(-1,3)
    uv=np.stack([np.broadcast_to(t[None]/(2*np.pi)*3.0,x.shape),np.broadcast_to(-p[:,1,None]/.8,x.shape)],-1).reshape(-1,2)
    if material=='iron':
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        uv=np.stack([np.broadcast_to(t[None]/(2*np.pi)*12,x.shape),np.broadcast_to(arc[:,None]/.15,x.shape)],-1).reshape(-1,2)
    fs=[]
    for j in range(len(p)-1):
        a=j*segments+np.arange(segments);b=j*segments+(np.arange(segments)+1)%segments;c=a+segments;d=b+segments
        fs.extend([np.stack([a,b,c],1),np.stack([b,d,c],1)])
    f=np.concatenate(fs)
    if inward:f=f[:,[0,2,1]]
    mesh.add(v,f,uv,material,part)

def cylinder(mesh,radius,depth,z,material,segments=192,part=''):
    t=np.linspace(0,2*np.pi,segments,endpoint=False);top=np.c_[radius*np.cos(t),radius*np.sin(t),np.full(segments,z)];bottom=top.copy();bottom[:,2]-=depth
    a=np.arange(segments);b=(a+1)%segments
    f=np.concatenate([np.stack([a,a+segments,b],1),np.stack([b,a+segments,b+segments],1)])
    uv=np.r_[np.c_[t/(2*np.pi)*12,np.zeros(segments)],np.c_[t/(2*np.pi)*12,np.full(segments,depth/.15)]]
    mesh.add(np.r_[top,bottom],f,uv,material,part+' side')
    for ring,height,reverse in [(top,z,False),(bottom,z-depth,True)]:
        v=np.r_[ring,[[0,0,height]]];f=np.stack([np.full(segments,segments),a,b],1)
        if reverse:f=f[:,[0,2,1]]
        mesh.add(v,f,v[:,:2]/(radius*2)+.5,material,part+' cap')

def box(mesh,center,size,material,transform=None,part=''):
    v=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*size+center
    f=np.array([[0,4,2],[2,4,6],[1,3,5],[3,7,5],[0,1,4],[1,5,4],[2,6,3],[3,6,7],[0,2,1],[1,2,3],[4,5,6],[5,7,6]])[:,[0,2,1]]
    if transform is not None:v=v@transform[:3,:3].T+transform[:3,3]
    mesh.add(v,f,v[:,:2]*4,material,part)

def frame_and_shaft(mesh,damage=False,depth=1.05):
    profile=[(.326,-.067),(.326,-.010),(.338,.002),(.389,.002),(.405,-.015),(.405,-.080),(.372,-.100),(.351,-.067),(.326,-.067)]
    lathe(mesh,profile,'iron',part='cast iron frame with inner cover seat',inward=True)
    profile=[]
    for z in np.linspace(-.07,-depth,45):
        joint=.004*np.exp(-((z+.36)/.009)**2)+.004*np.exp(-((z+.72)/.009)**2)
        profile.append((.327+joint,z))
    if damage:
        lathe(mesh,[(.342,z) for _,z in profile],'shaft',part='masonry mortar backing')
        for row,z in enumerate(np.arange(-.17,-depth+.05,-.19)):
            for column in range(14):
                t=2*np.pi*(column+.5*(row%2))/14
                transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('z',t).as_matrix()
                box(mesh,np.array([.356,0,z]),np.array([.060,.145,.178]),'brick',transform=transform,part='staggered masonry shaft brick')
    else:lathe(mesh,profile,'shaft',part='full-depth concrete shaft wall')
    cylinder(mesh,.327,.025,-depth,'damp',part='recessed dark shaft bottom')
    # Interior rungs add real depth cues; their light is occluded by the shaft.
    for z in [-.28,-.57,-.86]:
        box(mesh,np.array([0,.30,z]),np.array([.24,.016,.018]),'iron',part='interior access rung')

def cover(mesh,angle=65,offset=0,lean=False):
    n=385;yy,xx=np.meshgrid(np.linspace(.35,-.35,n),np.linspace(-.35,.35,n),indexing='ij');r=np.hypot(xx,yy)
    uv=np.stack([xx/.78+.5,yy/.78+.5],-1)
    disp=sample_source('ManholeCover009','Displacement',uv)
    h=(disp-np.median(disp[r<.33]))*.028+.012
    v=np.stack([xx,yy,h],-1).reshape(-1,3);f=grid(n);f=f[np.max(r.reshape(-1)[f],axis=1)<.342]
    # Rotate a physically thick lid about the back edge of the well frame.
    mat=np.eye(4)
    if offset:
        mat[:3,3]=[offset,.0,.065]
    else:
        rot=Rotation.from_euler('x',-angle,degrees=True).as_matrix();pivot=np.array([0,.343,.012]);mat[:3,:3]=rot;mat[:3,3]=pivot-rot@pivot
    if lean:mat[:3,3]+=[.37,.08,.0]
    def transformed(v):return v@mat[:3,:3].T+mat[:3,3]
    mesh.add(transformed(v),f,uv.reshape(-1,2),'cover','photogrammetric cast-iron lid top')
    temporary=Mesh();cylinder(temporary,.342,.048,-.002,'iron',part='lid thickness and back')
    # These ribs are visible when the cover is lifted; no mirrored top texture.
    for x in [-.20,-.10,0,.10,.20]:
        length=2*np.sqrt(.305**2-x*x);box(temporary,np.array([x,0,-.058]),np.array([.014,length,.020]),'iron',part='cast underside reinforcement rib')
    offset_index=0
    for vertices,faces,tex,mi,part in zip(temporary.v,temporary.f,temporary.uv,temporary.mi,temporary.parts):
        material=temporary.names[int(mi[0])];mesh.add(transformed(vertices),faces-offset_index,tex,material,part)
        offset_index+=len(vertices)

def chips(mesh,kind,pit,seed):
    rng=np.random.default_rng(seed)
    for i in range(68):
        angle=rng.uniform(0,2*np.pi);rr=rng.uniform(.73,1.12);x=np.cos(angle)*pit['rx']*rr;y=np.sin(angle)*pit['ry']*rr
        size=rng.uniform(.008,.030)
        r=rr/(1+.075*np.sin(3*angle+.8)+.045*np.sin(7*angle-1.2)+.025*np.cos(13*angle+seed))
        e=np.clip((1-r)/.065,0,1);t=np.clip((.94-r)/.28,0,1)
        floor=-pit['depth']*(.48*np.sqrt(e)+.52*t*t*(3-2*t))
        v=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
        v*=rng.uniform(.55,1.4,(6,3));v*=np.array([size,size*.8,size*.38]);v=v@Rotation.from_euler('xyz',rng.uniform(0,2*np.pi,3)).as_matrix().T
        v[:,2]-=v[:,2].min();v+=np.array([x,y,floor-.002])
        f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
        mesh.add(v,f,v[:,:2]*8,'aggregate','fractured loose road aggregate')

def footprint(path,height,extent):
    n=height.shape[0];yy,xx=np.meshgrid(np.linspace(.5,-.5,n),np.linspace(-.5,.5,n),indexing='ij')
    distance=np.minimum(.5-abs(xx),.5-abs(yy))*extent
    alpha=np.clip(distance/.075,0,1)
    np.savez_compressed(path/'surface.npz',height=height.astype('f4'),alpha=alpha.astype('f4'),width=extent,length=extent)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--catalog',help='Optional JSON list with unique asset IDs and physical geometry parameters');args=parser.parse_args()
    # Gray scanned concrete for the slab; exposed coarse aggregate is a separate material.
    prepare_material('asphalt','Asphalt031',[.60,.60,.60],scale=2.2)
    prepare_material('concrete','Concrete034',[1.0,1.0,1.0],scale=1.8)
    prepare_material('fracture_asphalt','Asphalt031',[.48,.48,.48],scale=1.1)
    prepare_material('fracture_concrete','Concrete034',[1.12,1.12,1.12],scale=.9)
    prepare_material('aggregate','Concrete049',[.95,.99,1.02],gray=.9,scale=.65)
    prepare_material('shaft','Concrete034',[.68,.66,.63],scale=.8)
    prepare_material('brick','Concrete034',[.88,.44,.27],scale=.45)
    prepare_material('damp','Concrete049',[.22,.24,.24],gray=.9,scale=.6)
    prepare_material('iron','ManholeCover009',[.48,.49,.50],rough=.92,metal=.42,gray=.90,scale=.7)
    prepare_material('cover','ManholeCover009',[.54,.55,.56],rough=.90,metal=.80,gray=.82,scale=.78)
    configs=[]
    for kind in ['asphalt','concrete']:
        for i,(variant,rx,ry,depth) in enumerate([('shallow',.36,.25,.045),('deep',.55,.37,.13),('elongated',.66,.23,.09)]):
            configs.append({'id':f'{kind}_pothole_{variant}','kind':kind,'type':'pothole','variant':variant,'rx':rx,'ry':ry,'depth':depth,'seed':36+i})
        for variant,depth in [('intact',1.05),('broken',1.35)]:configs.append({'id':f'{kind}_missing_{variant}','kind':kind,'type':'missing','variant':variant,'depth':depth,'seed':48})
        for variant,angle,offset in [('raised30',30,0),('raised65',65,0),('offset',0,.28),('leaning75',75,0)]:configs.append({'id':f'{kind}_cover_{variant}','kind':kind,'type':'cover','variant':variant,'angle':angle,'offset':offset,'depth':1.05,'seed':52})
    if args.catalog:configs=json.loads(Path(args.catalog).read_text())
    (OUT/'catalog.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    for cfg in configs:
        mesh=Mesh();kind=cfg['kind'];d=OUT/cfg['id']
        if cfg['type']=='pothole':
            extent,height=road(mesh,kind,cfg['seed'],pit=cfg);chips(mesh,kind,cfg,cfg['seed'])
        else:
            extent,height=road(mesh,kind,cfg['seed'],hole=True,damage=cfg['variant']=='broken');frame_and_shaft(mesh,cfg['variant']=='broken',cfg['depth'])
            # The footprint's negative core is used only for road removal/backing.
            n=height.shape[0];yy,xx=np.meshgrid(np.linspace(.95,-.95,n),np.linspace(-.95,.95,n),indexing='ij');height=np.where(np.hypot(xx,yy)<.325,-cfg['depth'],height)
            if cfg['type']=='cover':cover(mesh,cfg['angle'],cfg['offset'],cfg['variant']=='leaning75')
        mesh.save(d,cfg);footprint(d,height,extent)

if __name__=='__main__':main()
