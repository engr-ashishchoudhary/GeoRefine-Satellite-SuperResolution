"""
Tests for src.downstream.ndvi and src.downstream.pipeline.

Like test_inference.py, test_validation.py, and test_uncertainty.py, these
use a small, randomly-initialized RRDBNet rather than the real pretrained
checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset.registry import build_index
from src.downstream.ndvi import compute_ndvi
from src.downstream.pipeline import run_ndvi_analysis
from src.model.rrdbnet import RRDBNet


@pytest.fixture
def tiny_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=2, num_grow_ch=4, scale=4)
    model.eval()
    return model


def _write_fake_geotiff(path: Path, width=32, height=32, count=1, crs="EPSG:4326", res=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0.0, height * res, res, res)
    data = (np.random.rand(count, height, width) * 200 + 10).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="uint16", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# ndvi.py - pure logic
# ---------------------------------------------------------------------------

def test_compute_ndvi_known_values():
    red = np.array([[10.0, 20.0]])
    nir = np.array([[30.0, 20.0]])
    ndvi = compute_ndvi(red, nir)
    # (30-10)/(30+10) = 0.5 ; (20-20)/(20+20) = 0.0
    np.testing.assert_allclose(ndvi, [[0.5, 0.0]])


def test_compute_ndvi_zero_denominator_is_nan():
    red = np.array([[0.0]])
    nir = np.array([[0.0]])
    ndvi = compute_ndvi(red, nir)
    assert np.isnan(ndvi[0, 0])


def test_compute_ndvi_range_is_bounded():
    rng = np.random.default_rng(0)
    red = rng.random((10, 10)) * 255
    nir = rng.random((10, 10)) * 255
    ndvi = compute_ndvi(red, nir)
    valid = ~np.isnan(ndvi)
    assert np.all(ndvi[valid] >= -1.0)
    assert np.all(ndvi[valid] <= 1.0)


def test_compute_ndvi_rejects_shape_mismatch():
    red = np.zeros((4, 4))
    nir = np.zeros((2, 2))
    with pytest.raises(ValueError):
        compute_ndvi(red, nir)


# ---------------------------------------------------------------------------
# pipeline.py - full Phase 1 -> Phase 8 integration
# ---------------------------------------------------------------------------

def test_run_ndvi_analysis_produces_valid_output(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=16, height=16)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=16, height=16)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(demo_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
        "demo_data_contract": {
            "sr": "sr/scene_sr.tif",
            "original_ndvi": "ndvi/original_ndvi.tif",
            "sr_ndvi": "ndvi/sr_ndvi.tif",
        },
        "downstream": {"output_report_filename": "ndvi_report.json"},
    }

    report = run_ndvi_analysis(dataset_index_path, config, tiny_model)

    assert report["scene_id"] == "sceneA"
    assert report["original_ndvi"]["shape"] == [16, 16]
    assert report["sr_ndvi"]["shape"] == [64, 64]  # 16 * scale(4)

    original_path = demo_dir / "ndvi" / "original_ndvi.tif"
    sr_path = demo_dir / "ndvi" / "sr_ndvi.tif"
    assert original_path.exists()
    assert sr_path.exists()

    with rasterio.open(original_path) as src:
        assert src.count == 1
        assert src.width == 16
        assert src.height == 16
        data = src.read(1)
        valid = ~np.isnan(data)
        assert np.all(data[valid] >= -1.0)
        assert np.all(data[valid] <= 1.0)

    with rasterio.open(sr_path) as src:
        assert src.width == 64
        assert src.height == 64

    report_path = processed_dir / "ndvi_report.json"
    assert report_path.exists()
    assert "not directly observed satellite pixels" in report["note"]


def test_run_ndvi_analysis_raises_when_no_scenes(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(tmp_path / "demo")},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
        "demo_data_contract": {
            "sr": "sr/scene_sr.tif",
            "original_ndvi": "ndvi/original_ndvi.tif",
            "sr_ndvi": "ndvi/sr_ndvi.tif",
        },
        "downstream": {},
    }

    with pytest.raises(ValueError):
        run_ndvi_analysis(dataset_index_path, config, tiny_model)
