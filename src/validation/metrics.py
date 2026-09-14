"""
Core validation metrics for GeoRefine: PSNR, SSIM, RMSE, SAM.

These are pure array-comparison functions - they know nothing about scenes,
scan paths, or the real-HR vs. synthetic-HR distinction. src/validation/pipeline.py
is responsible for deciding *what* to compare (see that module's docstring);
this module only computes *how similar* two already-aligned, same-shape
(bands, H, W) arrays are.

All four functions accept an optional boolean `mask` (True = valid pixel,
False = excluded) so nodata pixels at scene edges or from resampling do not
skew results. If no mask is given, all pixels are treated as valid.

PSNR / RMSE / SSIM are computed per-band and averaged (unweighted mean
across bands) - documented in src/validation/README.md. SSIM specifically
uses skimage.metrics.structural_similarity per band rather than its
multichannel mode, since Sentinel-2 red/nir bands are not a natural
3-channel image the way RGB channels are, and multichannel SSIM's
cross-channel weighting assumptions do not apply here.

SAM (Spectral Angle Mapper) is inherently a per-pixel, cross-band
calculation - it is computed once, across all bands together, not
band-by-band. With only 2 bands (red, nir) it has limited discriminative
power compared to its typical use on hyperspectral data (see this
project's known limitations); it is included because the project spec
requires it, not because 2 bands make it a strong indicator on its own.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity


def _apply_mask(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray]):
    """Return flattened (a, b) pixel values restricted to valid pixels, for a single band."""
    if mask is None:
        return a.ravel(), b.ravel()
    return a[mask], b[mask]


def compute_rmse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Root-mean-square error, averaged across bands.

    a, b: (bands, H, W) arrays of the same shape.
    mask: optional (H, W) boolean array, True = include this pixel.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    per_band = []
    for band_idx in range(a.shape[0]):
        av, bv = _apply_mask(a[band_idx], b[band_idx], mask)
        if av.size == 0:
            continue
        per_band.append(float(np.sqrt(np.mean((av.astype(np.float64) - bv.astype(np.float64)) ** 2))))
    if not per_band:
        raise ValueError("No valid (unmasked) pixels to compute RMSE over")
    return float(np.mean(per_band))


def compute_psnr(
    a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None, mask: Optional[np.ndarray] = None
) -> float:
    """Peak signal-to-noise ratio in dB, averaged across bands.

    data_range: the dynamic range of the reference image (b), e.g. b.max() - b.min().
        If None, derived per-band from b's own min/max (over valid pixels).
    Returns math.inf if a and b are identical over all valid pixels for a band
    (RMSE == 0) - this is mathematically correct (infinite PSNR for an exact
    match), not an error, and is documented as such here since it can look
    surprising in a metrics report.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    per_band = []
    for band_idx in range(a.shape[0]):
        av, bv = _apply_mask(a[band_idx], b[band_idx], mask)
        if av.size == 0:
            continue
        av64, bv64 = av.astype(np.float64), bv.astype(np.float64)
        band_range = data_range if data_range is not None else (bv64.max() - bv64.min())
        if band_range <= 0:
            # Degenerate case: reference band is constant. PSNR is undefined
            # in the usual sense; skip rather than divide by zero or fabricate a number.
            continue
        mse = np.mean((av64 - bv64) ** 2)
        if mse == 0:
            per_band.append(math.inf)
        else:
            per_band.append(float(20 * math.log10(band_range) - 10 * math.log10(mse)))
    if not per_band:
        raise ValueError("No valid bands to compute PSNR over (all masked out or all constant)")
    finite = [v for v in per_band if math.isfinite(v)]
    if not finite:
        return math.inf
    # If some bands are finite and others inf, average using only finite
    # bands' actual values but still report inf if ALL bands were inf.
    if len(finite) < len(per_band):
        return float(np.mean(finite))
    return float(np.mean(per_band))


def compute_ssim(
    a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None, mask: Optional[np.ndarray] = None
) -> float:
    """Structural similarity index, averaged across bands.

    Computed per-band with skimage.metrics.structural_similarity (windowed,
    not global) - see this module's docstring for why per-band rather than
    multichannel. If a mask is given, SSIM is computed on the full band
    (windowed SSIM requires a regular 2D grid) and then only the masked
    pixels of the resulting per-pixel SSIM map are averaged.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    per_band = []
    for band_idx in range(a.shape[0]):
        av, bv = a[band_idx].astype(np.float64), b[band_idx].astype(np.float64)
        band_range = data_range if data_range is not None else (bv.max() - bv.min())
        if band_range <= 0:
            continue
        score, ssim_map = structural_similarity(av, bv, data_range=band_range, full=True)
        if mask is not None:
            if not mask.any():
                continue
            per_band.append(float(ssim_map[mask].mean()))
        else:
            per_band.append(float(score))
    if not per_band:
        raise ValueError("No valid bands to compute SSIM over (all masked out or all constant)")
    return float(np.mean(per_band))


def compute_sam(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Spectral Angle Mapper, in degrees, averaged over valid pixels.

    a, b: (bands, H, W) arrays of the same shape, at least 2 bands.
    For each pixel, treats the values across bands as a spectral vector and
    computes the angle between the two vectors (a's and b's) at that pixel.
    0 degrees = identical spectral shape; larger angles = more spectral
    distortion. Pixels where either vector has zero magnitude are excluded
    (angle is undefined there), not treated as zero error.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] < 2:
        raise ValueError("SAM requires at least 2 bands")

    bands, height, width = a.shape
    a_flat = a.reshape(bands, -1).astype(np.float64)
    b_flat = b.reshape(bands, -1).astype(np.float64)

    dot = np.sum(a_flat * b_flat, axis=0)
    norm_a = np.linalg.norm(a_flat, axis=0)
    norm_b = np.linalg.norm(b_flat, axis=0)
    denom = norm_a * norm_b

    valid = denom > 0
    if mask is not None:
        valid = valid & mask.ravel()
    if not valid.any():
        raise ValueError("No valid pixels to compute SAM over (all zero-magnitude or masked out)")

    cos_theta = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    return float(np.degrees(angles_rad).mean())