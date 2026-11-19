import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/pandaset_sdk').resolve()))
from pandaset import DataSet,geometry
scene=sys.argv[1] if len(sys.argv)>1 else '056'
s=DataSet('work/data/raw')[scene];s.load_camera();s.load_lidar()
root=Path('work/data/processed')/scene
errors=[]
for frame in [0,25,50,79]:
 cam=s.camera['front_camera'];c2w=geometry._heading_position_to_mat(cam.poses[frame]['heading'],cam.poses[frame]['position'])
 stored=np.loadtxt(root/'extrinsics'/f'{frame:03d}_0.txt');assert np.allclose(c2w,stored)
 points=s.lidar[frame].iloc[::20,:3].to_numpy()
 projected,_,ids=geometry.projection(points,cam[frame],cam.poses[frame],cam.intrinsics,filter_outliers=True)
 w2c=np.linalg.inv(stored);p=points[ids]@w2c[:3,:3].T+w2c[:3,3]
 K=np.loadtxt(root/'intrinsics/0.txt');uv=p[:,:2]/p[:,2:]*K[:2]+K[2:4]
 errors.append(float(np.max(np.abs(uv-projected))))
print('calibration matches raw SDK; max reprojection difference:',max(errors),'pixels')
assert max(errors)<1e-6
