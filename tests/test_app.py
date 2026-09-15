"""
Tests for app.data_adapter and the FastAPI app's foundation endpoints.
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

from app.data_adapter import get_demo_status, load_manifest, load_metrics, load_raster_summary


def _write_fake_geotiff(path: Path, width=8, height=8, count=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0.0, height, 1.0, 1.0)
    data = (np.random.rand(count, height, width) * 100).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="uint16", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def _base_config(demo_dir):
    return {
        "paths": {"demo_data": str(demo_dir)},
        "demo_data_contract": {
            "input": "input/scene.tif",
            "sr": "sr/scene_sr.tif",
            "metrics": "metrics/metrics.json",
        },
        "pipeline": {"manifest_filename": "manifest.json"},
    }


# ---------------------------------------------------------------------------
# data_adapter.py - pure logic
# ---------------------------------------------------------------------------

def test_load_manifest_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    assert load_manifest(config) is None


def test_load_manifest_returns_contents_when_present(tmp_path):
    demo_dir = tmp_path / "demo_data"
    demo_dir.mkdir()
    manifest_path = demo_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"scene": "input/scene.tif"}, f)
    config = _base_config(demo_dir)
    assert load_manifest(config) == {"scene": "input/scene.tif"}


def test_get_demo_status_reports_missing_files(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    status = get_demo_status(config)
    assert status["manifest_available"] is False
    assert status["ready"] is False
    assert all(not f["exists"] for f in status["files"].values())


def test_get_demo_status_reports_ready_when_all_present(tmp_path):
    demo_dir = tmp_path / "demo_data"
    _write_fake_geotiff(demo_dir / "input" / "scene.tif")
    _write_fake_geotiff(demo_dir / "sr" / "scene_sr.tif")
    metrics_path = demo_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    with open(metrics_path, "w") as f:
        json.dump({"psnr": 1.0, "ssim": 1.0, "rmse": 1.0, "sam": 1.0}, f)

    config = _base_config(demo_dir)
    status = get_demo_status(config)
    assert status["ready"] is True
    assert all(f["exists"] for f in status["files"].values())


def test_load_metrics_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    assert load_metrics(config) is None


def test_load_metrics_returns_contents_when_present(tmp_path):
    demo_dir = tmp_path / "demo_data"
    metrics_path = demo_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    with open(metrics_path, "w") as f:
        json.dump({"psnr": 5.0, "ssim": 0.9, "rmse": 2.0, "sam": 1.0}, f)
    config = _base_config(demo_dir)
    metrics = load_metrics(config)
    assert metrics["psnr"] == 5.0


def test_load_raster_summary_returns_metadata(tmp_path):
    demo_dir = tmp_path / "demo_data"
    _write_fake_geotiff(demo_dir / "sr" / "scene_sr.tif", width=16, height=16, count=2)
    config = _base_config(demo_dir)
    summary = load_raster_summary(config, "sr")
    assert summary["width"] == 16
    assert summary["height"] == 16
    assert summary["count"] == 2


def test_load_raster_summary_returns_none_when_missing(tmp_path):
    config = _base_config(tmp_path / "demo_data")
    assert load_raster_summary(config, "sr") is None


# ---------------------------------------------------------------------------
# app.py - FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_config(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo_data"
    config = _base_config(demo_dir)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    monkeypatch.setenv("GEOREFINE_CONFIG", str(config_path))

    import app.app as app_module
    importlib.reload(app_module)

    from fastapi.testclient import TestClient
    return TestClient(app_module.app), demo_dir


def test_api_status_reports_not_ready_when_empty(client_with_config):
    client, demo_dir = client_with_config
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False


def test_api_metrics_reports_unavailable_when_missing(client_with_config):
    client, demo_dir = client_with_config
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False


def test_api_metrics_returns_values_when_present(client_with_config):
    client, demo_dir = client_with_config
    metrics_path = demo_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    with open(metrics_path, "w") as f:
        json.dump({"psnr": 5.0, "ssim": 0.9, "rmse": 2.0, "sam": 1.0}, f)

    response = client.get("/api/metrics")
    data = response.json()
    assert data["available"] is True
    assert data["metrics"]["psnr"] == 5.0


def test_index_page_served(client_with_config):
    client, demo_dir = client_with_config
    response = client.get("/")
    assert response.status_code == 200
    assert "GeoRefine" in response.text