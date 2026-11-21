"""Trace the fixed ego-body boundary in the rear images using temporal edge persistence."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from road_geometry import SITES,save_json


def main(raw,out):
    profiles={}
    for site in SITES:
        edges=[]
        for frame in range(0,80,3):
            photo=cv2.imread(str(raw/site/'camera/back_camera'/f'{frame:02d}.jpg'),cv2.IMREAD_GRAYSCALE)
            image=photo[776:948].astype('f4')
            diff=np.zeros_like(image)
            diff[2:-2]=abs(image[4:]-image[:-4])
            edges.append(diff)
        signal=np.median(np.stack(edges),axis=0)
        score=gaussian_filter1d(signal,5,axis=1)
        x=np.arange(photo.shape[1])
        prior=826+.000078*(x-940)**2
        y=np.arange(len(score))[:,None]+776
        score[abs(y-prior)>19]=0
        edge=776+score.argmax(0)
        smooth=gaussian_filter1d(median_filter(edge.astype('f4'),size=21),7)
        guard=7
        column=np.r_[np.arange(0,len(x),12),len(x)-1]
        profiles[site]=dict(camera='back_camera',size=[photo.shape[1],photo.shape[0]],x=column.tolist(),
            first_protected_y=np.floor(smooth[column]-guard).astype(int).tolist(),guard_px=guard,
            boundary_y_range=[int(smooth.min()),int(smooth.max())],sampled_frames=list(range(0,80,3)))
    save_json(out,profiles)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.raw,a.out)
