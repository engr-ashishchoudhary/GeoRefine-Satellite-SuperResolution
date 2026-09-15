"""
Tests for src.pipeline.manifest and src.pipeline.orchestrator.

Like the other Phase 5+ test files, these use a small, randomly-initialized
RRDBNet rather than the real pretrained checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset.registry import build_index
from src.model.rrdbnet import RRDBNet
from src.pipeline.manifest import build_manifest, validate_demo_outputs, write_manifest
from src.pipeline.orchestrator import run_full_demo_pipeline
from src.preprocessing.pipeline import build_preprocessing_index


@pytest.fixture
def tiny_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=2, num_grow_ch=4, scale=4)
    model.eval()
    return model


def _write_fake_geotiff(path: Path, width=16, height=16, count=1, crs="EPSG:4326", res=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0.0, height * res, res, res)
    data = (np.random.rand(count, height, width) * 200 + 10).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="uint16", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


def _full_config(raw_dir, processed_dir, demo_dir):
    return {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(demo_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "dataset": {
            "index_filename": "dataset_index.json",
            "synthetic_hr_fallback": {"enabled": True, "blur_sigma": 1.0, "downsample_factor": 4},
        },
        "preprocessing": {
            "output_index_filename": "preprocessing_index.json",
            "synthetic_output_subdir": "synthetic_lr",
            "min_overlap_fraction": 0.5,
            "resolution_ratio_tolerance": 0.15,
        },
        "inference": {"tile_size": 16, "tile_overlap": 4, "scene_id": None},
        "demo_data_contract": {
            "input": "input/scene.tif",
            "sr": "sr/scene_sr.tif",
            "uncertainty": "uncertainty/scene_uncertainty.tif",
            "original_ndvi": "ndvi/original_ndvi.tif",
            "sr_ndvi": "ndvi/sr_ndvi.tif",
            "metrics": "metrics/metrics.json",
        },
        "validation": {"output_report_filename": "validation_report.json", "nodata_exclude": True},
        "uncertainty": {
            "augmentations": ["identity", "hflip", "vflip", "hvflip"],
            "output_report_filename": "uncertainty_report.json",
        },
        "downstream": {"output_report_filename": "ndvi_report.json"},
        "pipeline": {"manifest_filename": "manifest.json"},
    }


# ---------------------------------------------------------------------------
# manifest.py - pure logic
# ---------------------------------------------------------------------------

def test_build_manifest_matches_contract():
    config = {"demo_data_contract": {"scene": "input/scene.tif", "sr": "sr/scene_sr.tif"}}
    manifest = build_manifest(config)
    assert manifest == {"scene": "input/scene.tif", "sr": "sr/scene_sr.tif"}


def test_validate_demo_outputs_flags_missing_files(tmp_path):
    demo_dir = tmp_path / "demo_data"
    demo_dir.mkdir()
    config = {
        "paths": {"demo_data": str(demo_dir)},
        "demo_data_contract": {"scene": "input/scene.tif", "metrics": "metrics/metrics.json"},
    }
    issues = validate_demo_outputs(config)
    assert len(issues) == 2
    assert any("scene" in issue for issue in issues)
    assert any("metrics" in issue for issue in issues)


def test_validate_demo_outputs_flags_incomplete_metrics_json(tmp_path):
    demo_dir = tmp_path / "demo_data"
    metrics_path = demo_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    with open(metrics_path, "w") as f:
        json.dump({"psnr": 1.0, "ssim": 1.0}, f)  # missing rmse, sam

    config = {"paths": {"demo_data": str(demo_dir)}, "demo_data_contract": {"metrics": "metrics/metrics.json"}}
    issues = validate_demo_outputs(config)
    assert len(issues) == 1
    assert "missing required key" in issues[0]


def test_validate_demo_outputs_passes_for_valid_files(tmp_path):
    demo_dir = tmp_path / "demo_data"
    raster_path = demo_dir / "sr" / "scene_sr.tif"
    _write_fake_geotiff(raster_path, width=8, height=8)
    metrics_path = demo_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"psnr": 1.0, "ssim": 1.0, "rmse": 1.0, "sam": 1.0}, f)

    config = {
        "paths": {"demo_data": str(demo_dir)},
        "demo_data_contract": {"sr": "sr/scene_sr.tif", "metrics": "metrics/metrics.json"},
    }
    issues = validate_demo_outputs(config)
    assert issues == []


def test_write_manifest_writes_valid_json(tmp_path):
    config = {"demo_data_contract": {"scene": "input/scene.tif"}}
    output_path = tmp_path / "manifest.json"
    manifest = write_manifest(config, output_path)
    assert output_path.exists()
    with open(output_path) as f:
        saved = json.load(f)
    assert saved == manifest == {"scene": "input/scene.tif"}


# ---------------------------------------------------------------------------
# orchestrator.py - full Phase 1 -> Phase 9 integration
# ---------------------------------------------------------------------------

def test_run_full_demo_pipeline_end_to_end(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=16, height=16)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=16, height=16)

    config = _full_config(raw_dir, processed_dir, demo_dir)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    preprocessing_index_path = processed_dir / "preprocessing_index.json"
    build_preprocessing_index(dataset_index_path, config, preprocessing_index_path)

    report = run_full_demo_pipeline(dataset_index_path, preprocessing_index_path, config, tiny_model)

    assert report["scene_id"] == "sceneA"
    assert report["issues"] == []

    manifest_path = demo_dir / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest == config["demo_data_contract"]

    # Every file the manifest points to must actually exist.
    for relative_path in manifest.values():
        assert (demo_dir / relative_path).exists()

    # Re-validating independently should also find no issues.
    from src.pipeline.manifest import validate_demo_outputs

    assert validate_demo_outputs(config) == []


def test_run_full_demo_pipeline_raises_on_missing_preprocessing_data(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=16, height=16)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=16, height=16)

    config = _full_config(raw_dir, processed_dir, demo_dir)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    # Deliberately do NOT build preprocessing_index.json.
    preprocessing_index_path = processed_dir / "preprocessing_index.json"

    with pytest.raises(FileNotFoundError):
        run_full_demo_pipeline(dataset_index_path, preprocessing_index_path, config, tiny_model)