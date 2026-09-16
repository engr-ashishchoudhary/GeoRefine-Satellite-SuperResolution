"""
Phase 11/13/14: raster-to-PNG rendering for the dashboard.

Deliberately separate from app/data_adapter.py, which never loads pixel
data by design (see its docstring). This module owns pixel-level
rendering; data_adapter.py continues to own file-presence/metadata
reporting only.

Band order assumption (documented, not invented here): every module in
src/ builds requested_bands as {"red": ..., "nir": ...} in that literal
order when loading a band stack (see src/preprocessing/bands.py and every
caller of it). Since Python dicts preserve insertion order, this means
band index 0 is always red and index 1 is always nir in every raster this
project's pipeline writes. This module relies on that same convention -
it is not a new assumption specific to rendering.

Three distinct rendering paths, each with a visually distinct palette so
no two panels can be confused for each other:
  - render_false_color_png(): 2-band input/SR rasters (Phase 11). R=NIR,
    G/B=Red - a display convention, not a scientific claim.
  - render_ndvi_png(): single-band NDVI (Phase 13). Fixed brown-to-green
    colormap over NDVI's defined -1..1 range - not percentile-stretched,
    since that would make NDVI values incomparable across scenes.
  - render_uncertainty_png(): the uncertainty raster (Phase 14). Unlike
    NDVI, uncertainty has no fixed physical range - it is percentile-
    stretched (reusing percentile_stretch(), same as Phase 11) then mapped
    through a navy-to-red heat colormap. Bands are averaged into a single
    display heatmap for simplicity; the per-band breakdown is not lost -
    it lives in data/processed/uncertainty_report.json (Phase 7), surfaced
    directly in the dashboard panel rather than as a second image.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from PIL import Image


def percentile_stretch(band: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray:
    """Stretch a single band to uint8 [0, 255] using percentile clipping for
    display contrast. NaN/non-finite pixels are excluded from the
    percentile calculation and rendered as 0 (black), not interpolated or
    guessed at.
    """
    band = band.astype(np.float64)
    valid = np.isfinite(band)
    if not valid.any():
        return np.zeros(band.shape, dtype=np.uint8)

    low, high = np.percentile(band[valid], [low_percentile, high_percentile])
    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)

    stretched = np.clip((band - low) / (high - low), 0.0, 1.0)
    stretched = np.where(valid, stretched, 0.0)
    return (stretched * 255).astype(np.uint8)


def render_false_color_png(
    band_stack: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> bytes:
    """Render a (bands, H, W) array (band 0 = red, band 1 = nir) as a
    false-color PNG (R=NIR, G=Red, B=Red). Requires at least 2 bands.
    """
    if band_stack.ndim != 3 or band_stack.shape[0] < 2:
        raise ValueError(f"Expected a (bands>=2, H, W) array, got shape {band_stack.shape}")

    red = band_stack[0]
    nir = band_stack[1]

    r = percentile_stretch(nir, low_percentile, high_percentile)
    g = percentile_stretch(red, low_percentile, high_percentile)
    b = percentile_stretch(red, low_percentile, high_percentile)

    rgb = np.stack([r, g, b], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_raster_file_as_png(
    path: Union[str, Path], low_percentile: float = 2.0, high_percentile: float = 98.0
) -> bytes:
    """Read a raster file's full pixel array and render it as a false-color PNG."""
    with rasterio.open(path) as src:
        array = src.read()
    return render_false_color_png(array, low_percentile, high_percentile)


# NDVI colormap: brown (bare soil/water/low NDVI) -> yellow -> green
# (healthy vegetation), a standard vegetation-index display convention.
# Fixed value stops across the physically meaningful NDVI range, not
# data-dependent - two different scenes' NDVI values are visually
# comparable using this same mapping.
_NDVI_COLOR_STOPS = [
    (-1.0, (140, 90, 40)),
    (0.0, (210, 180, 140)),
    (0.2, (255, 255, 190)),
    (0.5, (130, 200, 80)),
    (1.0, (10, 90, 10)),
]


def render_ndvi_png(array: np.ndarray, ndvi_min: float = -1.0, ndvi_max: float = 1.0) -> bytes:
    """Render a single-band (H, W) NDVI array as an RGBA PNG using a fixed
    brown-to-green colormap. NaN pixels (undefined NDVI, see
    src.downstream.ndvi.compute_ndvi) are rendered fully transparent
    (alpha=0), not a solid color - "no data" must look visually distinct
    from "low NDVI," not just be another shade of brown.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected a (H, W) array, got shape {array.shape}")

    valid = np.isfinite(array)
    clipped = np.clip(array, ndvi_min, ndvi_max)

    # NaN pixels would otherwise propagate NaN through np.interp() into the
    # R/G/B channels, causing an invalid-value warning (and undefined
    # behavior) when cast to uint8. Substitute a safe in-range value (0.0)
    # for invalid pixels before interpolating - the actual color doesn't
    # matter since alpha=0 makes these pixels fully transparent regardless.
    safe_values = np.where(valid, clipped, 0.0)

    stop_values = [s[0] for s in _NDVI_COLOR_STOPS]
    r = np.interp(safe_values, stop_values, [s[1][0] for s in _NDVI_COLOR_STOPS])
    g = np.interp(safe_values, stop_values, [s[1][1] for s in _NDVI_COLOR_STOPS])
    b = np.interp(safe_values, stop_values, [s[1][2] for s in _NDVI_COLOR_STOPS])
    alpha = np.where(valid, 255, 0).astype(np.float64)

    rgba = np.stack([r, g, b, alpha], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_ndvi_raster_file_as_png(
    path: Union[str, Path], ndvi_min: float = -1.0, ndvi_max: float = 1.0
) -> bytes:
    """Read a single-band NDVI raster file and render it as a colormapped PNG."""
    with rasterio.open(path) as src:
        array = src.read(1)
    return render_ndvi_png(array, ndvi_min, ndvi_max)


# Uncertainty heatmap colormap: navy (low uncertainty) -> blue -> yellow ->
# orange -> red (high uncertainty). Deliberately distinct from both the
# false-color and NDVI palettes above, so a viewer can never mistake one
# panel's colors for another's.
_UNCERTAINTY_COLOR_STOPS = [
    (0.0, (10, 10, 40)),
    (0.25, (40, 80, 160)),
    (0.5, (250, 210, 40)),
    (0.75, (240, 120, 20)),
    (1.0, (200, 20, 20)),
]


def _heat_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map a [0, 1]-normalized array to RGB via the navy-to-red heat colormap."""
    stop_values = [s[0] for s in _UNCERTAINTY_COLOR_STOPS]
    r = np.interp(normalized, stop_values, [s[1][0] for s in _UNCERTAINTY_COLOR_STOPS])
    g = np.interp(normalized, stop_values, [s[1][1] for s in _UNCERTAINTY_COLOR_STOPS])
    b = np.interp(normalized, stop_values, [s[1][2] for s in _UNCERTAINTY_COLOR_STOPS])
    return np.stack([r, g, b], axis=-1)


def render_uncertainty_png(
    band_stack: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> bytes:
    """Render a (bands, H, W) uncertainty array as a heatmap PNG.

    Bands are averaged into a single 2D array before rendering - see this
    module's docstring for why. The averaged array is percentile-stretched
    (reusing percentile_stretch(), same technique as the false-color
    render) since uncertainty has no fixed physical range the way NDVI
    does, then mapped through the navy-to-red heat colormap.
    """
    if band_stack.ndim != 3:
        raise ValueError(f"Expected a (bands, H, W) array, got shape {band_stack.shape}")

    band_averaged = band_stack.mean(axis=0)
    stretched = percentile_stretch(band_averaged, low_percentile, high_percentile)
    normalized = stretched.astype(np.float64) / 255.0

    rgb = _heat_colormap(normalized).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_uncertainty_raster_file_as_png(
    path: Union[str, Path], low_percentile: float = 2.0, high_percentile: float = 98.0
) -> bytes:
    """Read the uncertainty raster file and render it as a heatmap PNG."""
    with rasterio.open(path) as src:
        array = src.read()
    return render_uncertainty_png(array, low_percentile, high_percentile)