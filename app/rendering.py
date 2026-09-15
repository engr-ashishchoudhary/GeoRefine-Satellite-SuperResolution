"""
Phase 11: raster-to-PNG rendering for the dashboard's before/after view.

Deliberately separate from app/data_adapter.py, which never loads pixel
data by design (see its docstring). This module owns pixel-level
rendering; data_adapter.py continues to own file-presence/metadata
reporting only.

Band order assumption (documented, not invented here): every module in
src/ builds requested_bands as {"red": ..., "nir": ...} in that literal
order when loading a band stack (see src/preprocessing/bands.py and every
caller of it). Since Python dicts preserve insertion order, this means
band index 0 is always red and index 1 is always nir in every raster this
project's pipeline writes (demo_data/input/scene.tif,
demo_data/sr/scene_sr.tif). This module relies on that same convention -
it is not a new assumption specific to rendering.

Rendering choice: Sentinel-2 imagery here is 2-band (red, nir) - there is
no green band, so no true-color RGB is possible. This module renders a
false-color composite (R channel = NIR, G/B channels = Red) purely for
visual comparison. This is a documented visualization convention, not a
scientific claim about the data - see app/README.md.
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