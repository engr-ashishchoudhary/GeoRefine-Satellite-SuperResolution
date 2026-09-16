"""
Tests for app.rendering (Phases 11, 13, 14).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from app.rendering import (
    percentile_stretch,
    render_false_color_png,
    render_ndvi_png,
    render_ndvi_raster_file_as_png,
    render_raster_file_as_png,
    render_uncertainty_png,
    render_uncertainty_raster_file_as_png,
)


# ---------------------------------------------------------------------------
# Phase 11: false-color rendering
# ---------------------------------------------------------------------------

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
    assert stretched[0, 2] == 0
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


# ---------------------------------------------------------------------------
# Phase 13: NDVI colormap rendering
# ---------------------------------------------------------------------------

def test_render_ndvi_png_produces_valid_rgba_png():
    rng = np.random.default_rng(0)
    array = rng.uniform(-1.0, 1.0, size=(20, 24))
    png_bytes = render_ndvi_png(array)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.format == "PNG"
    assert image.size == (24, 20)
    assert image.mode == "RGBA"


def test_render_ndvi_png_nan_pixels_are_transparent():
    array = np.array([[0.5, np.nan], [-0.5, 0.0]])
    png_bytes = render_ndvi_png(array)
    image = Image.open(io.BytesIO(png_bytes))
    pixels = np.array(image)
    assert pixels[0, 1, 3] == 0
    assert pixels[0, 0, 3] == 255


def test_render_ndvi_png_low_value_is_brownish():
    array = np.full((4, 4), -1.0)
    png_bytes = render_ndvi_png(array)
    image = Image.open(io.BytesIO(png_bytes))
    pixels = np.array(image)
    r, g, b, a = pixels[0, 0]
    assert r > g


def test_render_ndvi_png_high_value_is_greenish():
    array = np.full((4, 4), 1.0)
    png_bytes = render_ndvi_png(array)
    image = Image.open(io.BytesIO(png_bytes))
    pixels = np.array(image)
    r, g, b, a = pixels[0, 0]
    assert g > r


def test_render_ndvi_png_rejects_wrong_ndim():
    array = np.zeros((2, 4, 4))
    with pytest.raises(ValueError):
        render_ndvi_png(array)


def test_render_ndvi_raster_file_as_png_reads_real_file(tmp_path):
    path = tmp_path / "ndvi.tif"
    transform = from_origin(0.0, 16.0, 1.0, 1.0)
    rng = np.random.default_rng(0)
    data = rng.uniform(-1.0, 1.0, size=(1, 16, 16)).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=16, width=16,
        count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data)

    png_bytes = render_ndvi_raster_file_as_png(path)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.size == (16, 16)
    assert image.mode == "RGBA"


# ---------------------------------------------------------------------------
# Phase 14: uncertainty heatmap rendering
# ---------------------------------------------------------------------------

def test_render_uncertainty_png_produces_valid_rgb_png():
    rng = np.random.default_rng(0)
    band_stack = rng.random((2, 20, 24)) * 5.0
    png_bytes = render_uncertainty_png(band_stack)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.format == "PNG"
    assert image.size == (24, 20)
    assert image.mode == "RGB"


def test_render_uncertainty_png_low_value_is_darker_than_high_value():
    low = np.zeros((2, 4, 4))
    high = np.full((2, 4, 4), 100.0)

    combined = np.concatenate([low, high], axis=2)  # side-by-side low | high, same percentile stretch
    png_bytes = render_uncertainty_png(combined)
    image = Image.open(io.BytesIO(png_bytes))
    pixels = np.array(image)

    low_pixel_sum = int(pixels[0, 0].sum())
    high_pixel_sum = int(pixels[0, -1].sum())
    assert high_pixel_sum > low_pixel_sum  # high uncertainty -> brighter/hotter color


def test_render_uncertainty_png_accepts_single_band():
    band_stack = np.random.rand(1, 8, 8)
    png_bytes = render_uncertainty_png(band_stack)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.size == (8, 8)


def test_render_uncertainty_png_rejects_wrong_ndim():
    array = np.zeros((8, 8))
    with pytest.raises(ValueError):
        render_uncertainty_png(array)


def test_render_uncertainty_raster_file_as_png_reads_real_file(tmp_path):
    path = tmp_path / "uncertainty.tif"
    transform = from_origin(0.0, 16.0, 1.0, 1.0)
    data = (np.random.rand(2, 16, 16) * 3.0).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=16, width=16,
        count=2, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)

    png_bytes = render_uncertainty_raster_file_as_png(path)
    image = Image.open(io.BytesIO(png_bytes))
    assert image.size == (16, 16)
    assert image.mode == "RGB"