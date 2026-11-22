"""Identify intact patches surrounded by the optical crack mask in source coordinates."""

import numpy as np
from scipy import ndimage

MIN_AREA_M2 = 0.0025


def enclosed_regions(mask, pitch):
    regions, count = ndimage.label(~mask)
    areas = np.bincount(regions.ravel(), minlength=count + 1) * pitch**2
    outside = np.unique(np.r_[regions[0], regions[-1], regions[:, 0], regions[:, -1], 0])
    ids = np.flatnonzero((areas >= MIN_AREA_M2) & ~np.isin(np.arange(len(areas)), outside))
    ids = ids[np.argsort(-areas[ids], kind='stable')]
    lookup = np.zeros(len(areas), 'i4')
    lookup[ids] = np.arange(len(ids)) + 1
    return lookup[regions], areas[ids]
