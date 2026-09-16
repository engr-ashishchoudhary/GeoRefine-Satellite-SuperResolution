"""
Tests for app.data_adapter and the FastAPI app's endpoints (Phases 10-14).
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from app.data_adapter import (
    describe_metrics,
    get_demo_status,
    is_placeholder_metrics,
    load_manifest,
    load_metrics,
    load_ndvi_report,
    load_raster_summary,
    load_uncertainty_report,
)


def _write_fake_geotiff(path: Path, width=8, height=8, count=1, dtype="uint16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0.0, height, 1.0, 1.0)
    if dtype == "float32":
        data = np.random.uniform(-1.0, 1.0, size=(count, height, width)).astype("float32")
    else:
        data = (np.random.rand(count, height, width) * 100).astype(dtype)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype=dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def _base_config(demo_dir, processed_dir=None):
    return {
        "paths": {
            "demo_data": str(demo_dir),
            "data_processed": str(processed_dir) if processed_dir else str(demo_dir.parent / "processed"),
        },
        "demo_data_contract": {
            "input": "input/scene.tif",
            "sr": "sr/scene_sr.tif",
            "metrics": "metrics/metrics.json",
            "original_ndvi": "ndvi/original_ndvi.tif",
            "sr_ndvi": "ndvi/sr_ndvi.tif",
            "uncertainty": "uncertainty/scene_uncertainty.tif",
        },
        "pipeline": {"manifest_filename": "manifest.json"},
        "visualization": {
            "stretch_low_percentile": 2.0,
            "stretch_high_percentile": 98.0,
            "ndvi_min": -1.0,
            "ndvi_max": 1.0,
        },
        "downstream": {"output_report_filename": "ndvi_report.json"},
        "uncertainty": {"output_report_filename": "uncertainty_report.json"},
    }


# ---------------------------------------------------------------------------
# data_adapter.py - pure logic
# ---------------------------------------------------------------------------

def test_load_manifest_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    assert load_manifest(config) is None


def test_get_demo_status_reports_missing_files(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    status = get_demo_status(config)
    assert status["ready"] is False


def test_load_metrics_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    assert load_metrics(config) is None


def test_is_placeholder_metrics_true_for_phase0_style():
    placeholder = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "sam": 0.0, "note": "Placeholder values from Phase 0."}
    assert is_placeholder_metrics(placeholder) is True


def test_is_placeholder_metrics_false_for_real_result():
    real = {
        "psnr": 5.0, "ssim": 0.9, "rmse": 2.0, "sam": 1.0,
        "scene_id": "sceneA", "comparison_type": "real_hr_reference", "note": "...",
    }
    assert is_placeholder_metrics(real) is False


def test_describe_metrics_unavailable_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    result = describe_metrics(config)
    assert result == {"available": False, "is_placeholder": False, "metrics": None}


def test_load_raster_summary_returns_metadata(tmp_path):
    demo_dir = tmp_path / "demo_data"
    _write_fake_geotiff(demo_dir / "sr" / "scene_sr.tif", width=16, height=16, count=2)
    config = _base_config(demo_dir)
    summary = load_raster_summary(config, "sr")
    assert summary["width"] == 16


def test_load_ndvi_report_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data", tmp_path / "processed")
    assert load_ndvi_report(config) is None


def test_load_ndvi_report_returns_contents_when_present(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    report_path = processed_dir / "ndvi_report.json"
    with open(report_path, "w") as f:
        json.dump({"scene_id": "sceneA", "original_ndvi": {"stats": {"mean": 0.3}}}, f)
    config = _base_config(tmp_path / "demo_data", processed_dir)
    report = load_ndvi_report(config)
    assert report["scene_id"] == "sceneA"


def test_load_uncertainty_report_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data", tmp_path / "processed")
    assert load_uncertainty_report(config) is None


def test_load_uncertainty_report_returns_contents_when_present(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    report_path = processed_dir / "uncertainty_report.json"
    with open(report_path, "w") as f:
        json.dump({"scene_id": "sceneA", "num_augmentations": 4, "per_band_stats": []}, f)
    config = _base_config(tmp_path / "demo_data", processed_dir)
    report = load_uncertainty_report(config)
    assert report["scene_id"] == "sceneA"
    assert report["num_augmentations"] == 4


# ---------------------------------------------------------------------------
# app.py - FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_config(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo_data"
    processed_dir = tmp_path / "processed"
    config = _base_config(demo_dir, processed_dir)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    monkeypatch.setenv("GEOREFINE_CONFIG", str(config_path))

    import app.app as app_module
    importlib.reload(app_module)

    from fastapi.testclient import TestClient
    return TestClient(app_module.app), demo_dir, processed_dir


def test_api_status_reports_not_ready_when_empty(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_api_metrics_reports_unavailable_when_missing(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/metrics")
    assert response.json()["available"] is False


def test_index_page_served(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/")
    assert response.status_code == 200
    assert "GeoRefine" in response.text


def test_api_render_returns_404_when_missing(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/render/input")
    assert response.status_code == 404


def test_api_render_returns_png_when_present(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    _write_fake_geotiff(demo_dir / "input" / "scene.tif", width=16, height=16, count=2)

    response = client.get("/api/render/input")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_api_raster_info_returns_dimensions_when_present(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    _write_fake_geotiff(demo_dir / "input" / "scene.tif", width=8, height=8, count=2)
    _write_fake_geotiff(demo_dir / "sr" / "scene_sr.tif", width=32, height=32, count=2)

    response = client.get("/api/raster-info")
    data = response.json()
    assert data["input"]["width"] == 8
    assert data["sr"]["width"] == 32


def test_api_render_ndvi_returns_404_when_missing(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/render-ndvi/original_ndvi")
    assert response.status_code == 404


def test_api_render_ndvi_returns_png_when_present(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    _write_fake_geotiff(demo_dir / "ndvi" / "original_ndvi.tif", width=16, height=16, count=1, dtype="float32")

    response = client.get("/api/render-ndvi/original_ndvi")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_api_ndvi_info_returns_null_when_missing(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/ndvi-info")
    data = response.json()
    assert data["original_ndvi"] is None
    assert data["report"] is None


def test_api_render_uncertainty_returns_404_when_missing(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/render-uncertainty")
    assert response.status_code == 404


def test_api_render_uncertainty_returns_png_when_present(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    _write_fake_geotiff(demo_dir / "uncertainty" / "scene_uncertainty.tif", width=16, height=16, count=2, dtype="float32")

    response = client.get("/api/render-uncertainty")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_uncertainty_info_returns_null_when_missing(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    response = client.get("/api/uncertainty-info")
    data = response.json()
    assert data["uncertainty"] is None
    assert data["report"] is None


def test_api_uncertainty_info_returns_data_when_present(client_with_config):
    client, demo_dir, processed_dir = client_with_config
    _write_fake_geotiff(demo_dir / "uncertainty" / "scene_uncertainty.tif", width=16, height=16, count=2, dtype="float32")
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "uncertainty_report.json", "w") as f:
        json.dump({"scene_id": "sceneA", "num_augmentations": 4, "per_band_stats": []}, f)

    response = client.get("/api/uncertainty-info")
    data = response.json()
    assert data["uncertainty"]["width"] == 16
    assert data["report"]["scene_id"] == "sceneA"