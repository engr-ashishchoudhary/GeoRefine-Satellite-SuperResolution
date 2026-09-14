"""
Tests for src.inference.tiling and src.inference.scene_inference.
Like test_model.py, these use a small, randomly-initialized RRDBNet rather
than the real pretrained checkpoint, to keep tests fast and independent of
the checkpoint having been downloaded. The full-scene inference test
verifies the tiled overlap-blend pipeline produces correctly-shaped,
georeferenced output - not image quality (that is Phase 6's job).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from src.dataset.registry import build_index
from src.inference.scene_inference import run_demo_output, run_scene_inference
from src.inference.tiling import blend_weight_mask, compute_tile_grid
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
# tiling.py - pure logic
# ---------------------------------------------------------------------------
def test_compute_tile_grid_single_tile_for_small_scene():
    tiles = compute_tile_grid(height=64, width=64, tile_size=512, overlap=32)
    assert tiles == [(0, 0, 64, 64)]
def test_compute_tile_grid_covers_entire_scene():
    height, width = 100, 130
    tiles = compute_tile_grid(height, width, tile_size=64, overlap=16)
    covered = np.zeros((height, width), dtype=bool)
    for row, col, tile_h, tile_w in tiles:
        covered[row : row + tile_h, col : col + tile_w] = True
    assert covered.all()  # every pixel is covered by at least one tile
def test_blend_weight_mask_shape_and_range():
    weight = blend_weight_mask(tile_h=64, tile_w=64, overlap=16)
    assert weight.shape == (64, 64)
    assert weight.max() == pytest.approx(1.0)
    assert weight.min() > 0.0  # floor prevents exact zero
def test_blend_weight_mask_no_overlap_is_uniform():
    weight = blend_weight_mask(tile_h=32, tile_w=32, overlap=0)
    np.testing.assert_array_equal(weight, np.ones((32, 32)))
# ---------------------------------------------------------------------------
# scene_inference.py - full Phase 1 -> 5 integration
# ---------------------------------------------------------------------------
def test_run_scene_inference_produces_georeferenced_output(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=48, height=48)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=48, height=48)
    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)
    import json
    with open(dataset_index_path) as f:
        dataset_index = json.load(f)
    scene = dataset_index["scenes"][0]
    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(tmp_path / "demo")},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 32, "tile_overlap": 8, "scene_id": None},
    }
    output_path = processed_dir / "sceneA_sr.tif"
    run_scene_inference(scene, config, tiny_model, output_path)
    assert output_path.exists()
    with rasterio.open(output_path) as src:
        assert src.width == 48 * 4
        assert src.height == 48 * 4
        assert src.count == 2
        assert src.crs is not None
def test_run_demo_output_writes_contract_files(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    demo_dir = tmp_path / "demo_data"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=32, height=32)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=32, height=32)
    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)
    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(demo_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 512, "tile_overlap": 32, "scene_id": None},
        "demo_data_contract": {"input": "input/scene.tif", "sr": "sr/scene_sr.tif"},
    }
    result = run_demo_output(dataset_index_path, config, tiny_model)
    assert Path(result["input"]).exists()
    assert Path(result["sr"]).exists()
    assert result["scene_id"] == "sceneA"
    with rasterio.open(result["sr"]) as src:
        assert src.width == 32 * 4
        assert src.height == 32 * 4
def test_run_demo_output_raises_when_no_scenes(tmp_path, tiny_model):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)
    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir), "demo_data": str(tmp_path / "demo")},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "inference": {"tile_size": 512, "tile_overlap": 32, "scene_id": None},
        "demo_data_contract": {"input": "input/scene.tif", "sr": "sr/scene_sr.tif"},
    }
    with pytest.raises(ValueError):
        run_demo_output(dataset_index_path, config, tiny_model)
