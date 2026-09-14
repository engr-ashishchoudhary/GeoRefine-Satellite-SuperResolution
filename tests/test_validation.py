"""
Tests for src.validation.metrics and src.validation.pipeline.

Like test_inference.py, these use a small, randomly-initialized RRDBNet
rather than the real pretrained checkpoint, to keep tests fast and
independent of the checkpoint having been downloaded. The metrics tests
check known, hand-verifiable values (identical arrays, orthogonal spectral
vectors); the pipeline tests run a full Phase 1 -> Phase 2 -> Phase 6
integration against synthetic GeoTIFFs, covering both the real-HR path and
the synthetic-degradation-proxy path.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset.registry import build_index
from src.model.rrdbnet import RRDBNet
from src.preprocessing.pipeline import build_preprocessing_index
from src.validation.metrics import compute_psnr, compute_rmse, compute_sam, compute_ssim
from src.validation.pipeline import run_validation


@pytest.fixture
def tiny_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=2, num_grow_ch=4, scale=4)
    model.eval()
    return model


def _write_fake_geotiff(path: Path, width=32, height=32, count=1, crs="EPSG:4326", res=1.0, origin=(0.0, None)):
    path.parent.mkdir(parents=True, exist_ok=True)
    top = origin[1] if origin[1] is not None else height * res
    transform = from_origin(origin[0], top, res, res)
    data = (np.random.rand(count, height, width) * 200 + 10).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="uint16", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


def _base_config(raw_dir, processed_dir, demo_dir):
    return {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(demo_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "dataset": {
            "synthetic_hr_fallback": {"enabled": True, "blur_sigma": 1.0, "downsample_factor": 4},
        },
        "preprocessing": {
            "output_index_filename": "preprocessing_index.json",
            "synthetic_output_subdir": "synthetic_lr",
            "min_overlap_fraction": 0.5,
            "resolution_ratio_tolerance": 0.15,
        },
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
        "demo_data_contract": {
            "input": "input/scene.tif",
            "sr": "sr/scene_sr.tif",
            "metrics": "metrics/metrics.json",
        },
        "validation": {"output_report_filename": "validation_report.json", "nodata_exclude": True},
    }


# ---------------------------------------------------------------------------
# metrics.py - pure, hand-verifiable values
# ---------------------------------------------------------------------------

def test_compute_rmse_identical_arrays_is_zero():
    rng = np.random.default_rng(0)
    a = rng.random((2, 8, 8))
    assert compute_rmse(a, a) == pytest.approx(0.0)


def test_compute_psnr_identical_arrays_is_infinite():
    rng = np.random.default_rng(0)
    a = rng.random((2, 8, 8)) * 100
    assert math.isinf(compute_psnr(a, a))


def test_compute_ssim_identical_arrays_is_one():
    rng = np.random.default_rng(0)
    a = rng.random((2, 16, 16)) * 100
    assert compute_ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_compute_sam_identical_arrays_is_zero():
    rng = np.random.default_rng(0)
    a = rng.random((2, 8, 8)) + 0.1  # avoid exact-zero vectors
    assert compute_sam(a, a) == pytest.approx(0.0, abs=1e-6)


def test_compute_sam_orthogonal_vectors_is_ninety_degrees():
    a = np.array([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])  # band0=1, band1=0 everywhere
    b = np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]])  # band0=0, band1=1 everywhere
    assert compute_sam(a, b) == pytest.approx(90.0, abs=1e-6)


def test_metrics_raise_on_shape_mismatch():
    a = np.zeros((2, 8, 8))
    b = np.zeros((2, 4, 4))
    with pytest.raises(ValueError):
        compute_rmse(a, b)
    with pytest.raises(ValueError):
        compute_psnr(a, b)
    with pytest.raises(ValueError):
        compute_ssim(a, b)
    with pytest.raises(ValueError):
        compute_sam(a, b)


def test_compute_rmse_respects_mask():
    a = np.zeros((1, 4, 4))
    b = np.zeros((1, 4, 4))
    b[0, 0, 0] = 1000.0  # one wildly different pixel
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False  # exclude that pixel
    assert compute_rmse(a, b, mask=mask) == pytest.approx(0.0)
    assert compute_rmse(a, b) > 0.0  # without the mask, the outlier pulls RMSE up


def test_compute_sam_requires_at_least_two_bands():
    a = np.zeros((1, 4, 4))
    with pytest.raises(ValueError):
        compute_sam(a, a)


# ---------------------------------------------------------------------------
# pipeline.py - full Phase 1 -> Phase 2 -> Phase 6 integration
# ---------------------------------------------------------------------------

def test_run_validation_real_hr_path(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "B04.tif", width=16, height=16, res=4.0, origin=(0.0, 64.0))
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "B08.tif", width=16, height=16, res=4.0, origin=(0.0, 64.0))
    _write_fake_geotiff(
        raw_dir / "sceneB" / "hr" / "ref.tif", width=64, height=64, res=1.0, origin=(0.0, 64.0), count=2
    )

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = _base_config(raw_dir, processed_dir, demo_dir)
    config["inference"]["scene_id"] = "sceneB"

    preprocessing_index_path = processed_dir / "preprocessing_index.json"
    build_preprocessing_index(dataset_index_path, config, preprocessing_index_path)

    report = run_validation(dataset_index_path, preprocessing_index_path, config, tiny_model)

    assert report["comparison_type"] == "real_hr_reference"
    assert report["scene_id"] == "sceneB"
    for key in ("psnr", "ssim", "rmse", "sam"):
        assert key in report["metrics"]
        value = report["metrics"][key]
        assert value is None or isinstance(value, float)

    metrics_path = demo_dir / "metrics" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics_json = json.load(f)
    for key in ("psnr", "ssim", "rmse", "sam", "scene_id", "comparison_type", "note"):
        assert key in metrics_json
    assert metrics_json["comparison_type"] == "real_hr_reference"

    report_path = processed_dir / "validation_report.json"
    assert report_path.exists()

    sr_path = demo_dir / "sr" / "scene_sr.tif"
    assert sr_path.exists()


def test_run_validation_synthetic_path(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=32, height=32)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=32, height=32)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = _base_config(raw_dir, processed_dir, demo_dir)
    config["inference"]["scene_id"] = "sceneA"

    preprocessing_index_path = processed_dir / "preprocessing_index.json"
    build_preprocessing_index(dataset_index_path, config, preprocessing_index_path)

    report = run_validation(dataset_index_path, preprocessing_index_path, config, tiny_model)

    assert report["comparison_type"] == "synthetic_degradation_proxy"
    assert report["scene_id"] == "sceneA"
    for key in ("psnr", "ssim", "rmse", "sam"):
        assert key in report["metrics"]

    metrics_path = demo_dir / "metrics" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics_json = json.load(f)
    assert metrics_json["comparison_type"] == "synthetic_degradation_proxy"
    assert "NOT a measurement of real-world HR recovery" in metrics_json["note"]

    report_path = processed_dir / "validation_report.json"
    assert report_path.exists()


def test_run_validation_raises_when_no_hr_and_fallback_disabled(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif")
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif")

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = _base_config(raw_dir, processed_dir, demo_dir)
    config["dataset"]["synthetic_hr_fallback"]["enabled"] = False

    preprocessing_index_path = processed_dir / "preprocessing_index.json"
    build_preprocessing_index(dataset_index_path, config, preprocessing_index_path)

    with pytest.raises(ValueError):
        run_validation(dataset_index_path, preprocessing_index_path, config, tiny_model)