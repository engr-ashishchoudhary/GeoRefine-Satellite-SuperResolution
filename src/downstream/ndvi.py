"""
NDVI (Normalized Difference Vegetation Index) calculation for GeoRefine.

NDVI = (NIR - Red) / (NIR + Red)

Pure array function - knows nothing about scenes, file paths, or which
raster (original LR or SR output) it is being called on. src/downstream/pipeline.py
decides what to compute NDVI from; this module only computes the formula
itself, correctly and consistently, from real pixel values.
"""
from __future__ import annotations

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI from red and nir band arrays of the same shape.

    Returns an array of the same shape, dtype float64. Pixels where
    (nir + red) == 0 are set to NaN - NDVI is mathematically undefined
    there (0/0), not zero. This is a real edge case for masked/nodata
    areas and must not be silently reported as "no vegetation" (NDVI 0),
    which would be a fabricated value.
    """
    if red.shape != nir.shape:
        raise ValueError(f"Shape mismatch between red and nir: {red.shape} vs {nir.shape}")

    red64 = red.astype(np.float64)
    nir64 = nir.astype(np.float64)
    denom = nir64 + red64

    ndvi = np.full(red64.shape, np.nan, dtype=np.float64)
    valid = denom != 0
    ndvi[valid] = (nir64[valid] - red64[valid]) / denom[valid]
    return ndvi