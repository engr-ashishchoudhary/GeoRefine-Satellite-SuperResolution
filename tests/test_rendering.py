"""
Tests for app.rendering.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from app.rendering import percentile_stretch, render_false_color_png, render_raster_file_as_png


def test_percentile_stretch_output_range():
    rng = np.random.default_rng(0)
    band = rng.random((16, 16)) * 1000
    stretched = percentile_stretch(band)
    assert stretched.dtype == np.uint8
    assert stretched.min() >= 0
    assert stretched.max() <= 255


def test_percentile_stretch_constant_band_is_zero():
    band = np.full((8, 8), 42.0)
    stretched = percentile_stretch(band)
    assert np.all(stretched == 0)


def test_percentile_stretch_excludes_nan_from_calculation():
    band = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan]])
    stretched = percentile_stretch(band, low_percentile=0, high_percentile=100)
    assert stretched[0, 2] == 0  # NaN pixel rendered as black
    assert stretched[1, 2] == 0


def test_render_false_color_png_produces_valid_png():
    rng = np.random.default_rng(0)
    band_stack = rng.random((2, 32, 32)) * 255
    png_bytes = render_false_color_png(band_stack)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.format == "PNG"
    assert image.size == (32, 32)
    assert image.mode == "RGB"


def test_render_false_color_png_rejects_single_band():
    band_stack = np.zeros((1, 8, 8))
    with pytest.raises(ValueError):
        render_false_color_png(band_stack)


def test_render_raster_file_as_png_reads_real_file(tmp_path):
    path = tmp_path / "scene.tif"
    transform = from_origin(0.0, 16.0, 1.0, 1.0)
    data = (np.random.rand(2, 16, 16) * 200).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=16, width=16,
        count=2, dtype="uint16", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)

    png_bytes = render_raster_file_as_png(path)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.size == (16, 16)