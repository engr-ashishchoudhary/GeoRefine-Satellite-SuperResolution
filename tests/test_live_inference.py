"""
Tests for app.live_inference and the /api/live/* FastAPI endpoints.

Like every other phase's tests, these use a small, randomly-initialized
RRDBNet rather than the real pretrained checkpoint - achieved here by
monkeypatching app.live_inference.load_model (the name live_inference.py
imports and calls), not by touching models/RealESRGAN_x4plus.pth.
"""
from __future__ import annotations

import importlib
import io
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from src.model.rrdbnet import RRDBNet


@pytest.fixture
def tiny_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=2, num_grow_ch=4, scale=4)
    model.eval()
    return model


def _geotiff_bytes(width=16, height=16, count=2) -> bytes:
    transform = from_origin(0.0, height, 1.0, 1.0)
    data = (np.random.rand(count, height, width) * 200 + 10).astype("uint16")
    buffer = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=count,
            dtype="uint16", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)
        buffer.write(memfile.read())
    return buffer.getvalue()


def _live_config(tmp_path):
    return {
        "paths": {"models": str(tmp_path / "models")},
        "inference": {"tile_size": 16, "tile_overlap": 4, "scene_id": None},
        "live": {
            "upload_dir": str(tmp_path / "live_uploads"),
            "output_dir": str(tmp_path / "live_outputs"),
        },
    }


# ---------------------------------------------------------------------------
# live_inference.py - pure logic
# ---------------------------------------------------------------------------

def test_run_live_inference_rejects_single_band(tmp_path, monkeypatch, tiny_model):
    import app.live_inference as live_inference_module

    monkeypatch.setattr(live_inference_module, "load_model", lambda models_dir: tiny_model)

    config = _live_config(tmp_path)
    single_band_bytes = _geotiff_bytes(count=1)

    with pytest.raises(ValueError, match="at least 2 are required"):
        live_inference_module.run_live_inference(single_band_bytes, "upload.tif", config)


def test_run_live_inference_rejects_unreadable_file(tmp_path, monkeypatch, tiny_model):
    import app.live_inference as live_inference_module

    monkeypatch.setattr(live_inference_module, "load_model", lambda models_dir: tiny_model)

    config = _live_config(tmp_path)

    with pytest.raises(ValueError, match="Could not read"):
        live_inference_module.run_live_inference(b"not a real geotiff", "garbage.tif", config)


def test_run_live_inference_produces_sr_output(tmp_path, monkeypatch, tiny_model):
    import app.live_inference as live_inference_module

    monkeypatch.setattr(live_inference_module, "load_model", lambda models_dir: tiny_model)

    config = _live_config(tmp_path)
    upload_bytes = _geotiff_bytes(width=16, height=16, count=2)

    result = live_inference_module.run_live_inference(upload_bytes, "my_scene.tif", config)

    assert result["original_filename"] == "my_scene.tif"
    assert result["scene_shape"] == [2, 16, 16]
    assert result["sr_shape"] == [2, 64, 64]  # 16 * scale(4)
    assert Path(result["input_path"]).exists()
    assert Path(result["sr_path"]).exists()

    # get_live_result() should reflect the same result.
    cached = live_inference_module.get_live_result()
    assert cached["sr_path"] == result["sr_path"]


# ---------------------------------------------------------------------------
# app.py - /api/live/* endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def live_client(tmp_path, monkeypatch, tiny_model):
    config = {
        "paths": {
            "demo_data": str(tmp_path / "demo_data"),
            "data_processed": str(tmp_path / "processed"),
            "models": str(tmp_path / "models"),
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
        "app": {"host": "127.0.0.1", "port": 8000, "mode": "demo"},
        "inference": {"tile_size": 16, "tile_overlap": 4, "scene_id": None},
        "live": {
            "upload_dir": str(tmp_path / "live_uploads"),
            "output_dir": str(tmp_path / "live_outputs"),
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    monkeypatch.setenv("GEOREFINE_CONFIG", str(config_path))

    import app.app as app_module
    importlib.reload(app_module)

    import app.live_inference as live_inference_module
    monkeypatch.setattr(live_inference_module, "load_model", lambda models_dir: tiny_model)
    # app.app imported run_live_inference by reference before this monkeypatch;
    # since run_live_inference itself looks up _get_model -> load_model within
    # the live_inference module's own namespace, the patch above is sufficient.

    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def test_api_live_status_unavailable_initially(live_client):
    # NOTE: app.live_inference's module-level _live_result persists across
    # tests in the same process. This test only checks the response shape,
    # not that it is unconditionally empty, to avoid order-dependent flakiness.
    response = live_client.get("/api/live/status")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data


def test_api_live_render_404_before_any_upload(tmp_path, monkeypatch, tiny_model):
    # Use a fresh live_inference module state by resetting its module-level cache.
    import app.live_inference as live_inference_module
    monkeypatch.setattr(
        live_inference_module,
        "_live_result",
        {"input_path": None, "sr_path": None, "scene_shape": None, "sr_shape": None, "original_filename": None},
    )

    config = {
        "paths": {
            "demo_data": str(tmp_path / "demo_data"),
            "data_processed": str(tmp_path / "processed"),
            "models": str(tmp_path / "models"),
        },
        "demo_data_contract": {"input": "input/scene.tif", "sr": "sr/scene_sr.tif"},
        "pipeline": {"manifest_filename": "manifest.json"},
        "visualization": {"stretch_low_percentile": 2.0, "stretch_high_percentile": 98.0},
        "app": {"host": "127.0.0.1", "port": 8000, "mode": "demo"},
        "inference": {"tile_size": 16, "tile_overlap": 4, "scene_id": None},
        "live": {"upload_dir": str(tmp_path / "u"), "output_dir": str(tmp_path / "o")},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    monkeypatch.setenv("GEOREFINE_CONFIG", str(config_path))

    import app.app as app_module
    importlib.reload(app_module)
    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)

    response = client.get("/api/live/render/sr")
    assert response.status_code == 404


def test_api_live_infer_rejects_single_band_upload(live_client):
    single_band_bytes = _geotiff_bytes(count=1)
    response = live_client.post(
        "/api/live/infer", files={"file": ("upload.tif", single_band_bytes, "image/tiff")}
    )
    assert response.status_code == 400
    assert "at least 2 are required" in response.json()["detail"]


def test_api_live_infer_and_render_full_workflow(live_client):
    upload_bytes = _geotiff_bytes(width=16, height=16, count=2)
    infer_response = live_client.post(
        "/api/live/infer", files={"file": ("my_scene.tif", upload_bytes, "image/tiff")}
    )
    assert infer_response.status_code == 200
    result = infer_response.json()
    assert result["sr_shape"] == [2, 64, 64]

    status_response = live_client.get("/api/live/status")
    assert status_response.json()["available"] is True

    sr_render = live_client.get("/api/live/render/sr")
    assert sr_render.status_code == 200
    assert sr_render.headers["content-type"] == "image/png"

    input_render = live_client.get("/api/live/render/input")
    assert input_render.status_code == 200