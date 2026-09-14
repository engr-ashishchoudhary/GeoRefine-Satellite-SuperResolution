"""
Tests for src.uncertainty.tta and src.uncertainty.pipeline.

Like test_inference.py and test_validation.py, these use a small,
randomly-initialized RRDBNet rather than the real pretrained checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset.registry import build_index
from src.model.rrdbnet import RRDBNet
from src.uncertainty.pipeline import run_uncertainty_estimation
from src.uncertainty.tta import (
    AUGMENTATION_NAMES,
    apply_augmentation,
    invert_augmentation,
    validate_augmentation_names,
)


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
# tta.py - pure logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", AUGMENTATION_NAMES)
def test_apply_then_invert_is_identity(name):
    rng = np.random.default_rng(0)
    array = rng.random((2, 8, 10))
    augmented = apply_augmentation(array, name)
    restored = invert_augmentation(augmented, name)
    np.testing.assert_array_equal(restored, array)


def test_apply_augmentation_identity_is_unchanged():
    array = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_array_equal(apply_augmentation(array, "identity"), array)


def test_apply_augmentation_hflip_reverses_width_axis():
    array = np.arange(24).reshape(2, 3, 4)
    flipped = apply_augmentation(array, "hflip")
    np.testing.assert_array_equal(flipped, array[:, :, ::-1])


def test_apply_augmentation_rejects_unknown_name():
    array = np.zeros((2, 4, 4))
    with pytest.raises(ValueError):
        apply_augmentation(array, "rotate90")


def test_validate_augmentation_names_rejects_unknown():
    with pytest.raises(ValueError):
        validate_augmentation_names(["identity", "rotate90"])


def test_validate_augmentation_names_accepts_known():
    validate_augmentation_names(["identity", "hflip"])  # should not raise


# ---------------------------------------------------------------------------
# pipeline.py - full Phase 1 -> Phase 7 integration
# ---------------------------------------------------------------------------

def test_run_uncertainty_estimation_produces_valid_output(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=24, height=24)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=24, height=24)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(demo_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
        "demo_data_contract": {"uncertainty": "uncertainty/scene_uncertainty.tif"},
        "uncertainty": {
            "augmentations": ["identity", "hflip", "vflip", "hvflip"],
            "output_report_filename": "uncertainty_report.json",
        },
    }

    report = run_uncertainty_estimation(dataset_index_path, config, tiny_model)

    assert report["scene_id"] == "sceneA"
    assert report["num_augmentations"] == 4
    assert len(report["per_band_stats"]) == 2  # red, nir

    for band_stat in report["per_band_stats"]:
        assert band_stat["mean_uncertainty"] >= 0.0
        assert band_stat["max_uncertainty"] >= band_stat["mean_uncertainty"]

    uncertainty_path = demo_dir / "uncertainty" / "scene_uncertainty.tif"
    assert uncertainty_path.exists()
    with rasterio.open(uncertainty_path) as src:
        assert src.width == 24 * 4
        assert src.height == 24 * 4
        assert src.count == 2
        assert src.crs is not None
        data = src.read()
        assert np.all(data >= 0.0)  # standard deviation is never negative

    report_path = processed_dir / "uncertainty_report.json"
    assert report_path.exists()


def test_run_uncertainty_estimation_raises_when_no_scenes(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(tmp_path / "demo")},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
        "demo_data_contract": {"uncertainty": "uncertainty/scene_uncertainty.tif"},
        "uncertainty": {"augmentations": ["identity", "hflip"]},
    }

    with pytest.raises(ValueError):
        run_uncertainty_estimation(dataset_index_path, config, tiny_model)


def test_run_uncertainty_estimation_rejects_too_few_augmentations(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=16, height=16)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=16, height=16)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(tmp_path / "demo")},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
        "demo_data_contract": {"uncertainty": "uncertainty/scene_uncertainty.tif"},
        "uncertainty": {"augmentations": ["identity"]},
    }

    with pytest.raises(ValueError):
        run_uncertainty_estimation(dataset_index_path, config, tiny_model)
