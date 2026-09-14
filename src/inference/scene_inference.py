"""
Phase 5 orchestration: full-scene super-resolution inference.
Reuses src.preprocessing.bands.load_band_stack() (Phase 2) to load a
scene's real LR imagery - the same band-matching logic already used for
alignment and patch generation, so scenes are interpreted identically
across every phase.
Tiling (src.inference.tiling) exists because a full Sentinel-2 scene can be
far larger than what a single forward pass through the pretrained model
should attempt on a normal laptop. Each tile is processed independently via
src.model.super_resolution.apply_super_resolution() (per-band pseudo-RGB,
see that module's docstring for the documented approximation), then
combined into the full-resolution output canvas using overlap-add blending
so tile boundaries do not produce visible seams.
This module produces two things:
  1. run_scene_inference() - the general-purpose function: given any scene
     dict (from dataset_index.json) and a loaded model, produces a
     georeferenced SR GeoTIFF at an arbitrary output path. This is the
     function later phases (or an eventual dashboard "live mode") would
     call for any scene, not just the demo one.
  2. run_demo_output() - a thin wrapper that selects one scene (per
     config.yaml's inference.scene_id, or the first available scene) and
     writes it to the project's stable demo_data/ output contract
     (demo_data/input/scene.tif, demo_data/sr/scene_sr.tif), established
     in Phase 0 and unchanged since.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import rasterio
from rasterio import Affine
from src.dataset.scene import read_raster_metadata
from src.model.super_resolution import apply_super_resolution
from src.preprocessing.bands import load_band_stack
from .tiling import blend_weight_mask, compute_tile_grid
def _load_scene_lr_stack(scene: Dict[str, Any], config: Dict[str, Any]):
    """Returns (band_stack_array, crs, transform_6, nodata) for a scene's real LR imagery."""
    lr_metas = [read_raster_metadata(f["path"]) for f in scene["lr_files"]]
    sentinel2_cfg = config.get("sentinel2", {})
    requested_bands = {"red": sentinel2_cfg.get("red_band", "B4"), "nir": sentinel2_cfg.get("nir_band", "B8")}
    band_stack = load_band_stack(lr_metas, requested_bands)
    if not band_stack.ok:
        raise ValueError(f"Could not load LR band stack for scene '{scene['scene_id']}': {band_stack.issues}")
    lr_nodata = lr_metas[0].nodata if lr_metas else None
    return band_stack.data, band_stack.crs, band_stack.transform, lr_nodata
def run_scene_inference(
    scene: Dict[str, Any],
    config: Dict[str, Any],
    model,
    output_path: Path,
) -> Path:
    """Run tiled super-resolution inference on one full scene, writing a
    georeferenced SR GeoTIFF to output_path. Returns output_path.
    """
    lr_array, lr_crs, lr_transform_6, lr_nodata = _load_scene_lr_stack(scene, config)
    num_bands, height, width = lr_array.shape
    inference_cfg = config.get("inference", {})
    tile_size = inference_cfg.get("tile_size", 512)
    overlap = inference_cfg.get("tile_overlap", 32)
    scale = model.scale
    out_height, out_width = height * scale, width * scale
    accumulated = np.zeros((num_bands, out_height, out_width), dtype=np.float64)
    weight_sum = np.zeros((out_height, out_width), dtype=np.float64)
    tiles = compute_tile_grid(height, width, tile_size, overlap)
    for row, col, tile_h, tile_w in tiles:
        lr_tile = lr_array[:, row : row + tile_h, col : col + tile_w]
        sr_tile = apply_super_resolution(model, lr_tile)  # (bands, tile_h*scale, tile_w*scale)
        weight = blend_weight_mask(tile_h * scale, tile_w * scale, overlap * scale)
        out_row, out_col = row * scale, col * scale
        out_tile_h, out_tile_w = sr_tile.shape[1], sr_tile.shape[2]
        accumulated[:, out_row : out_row + out_tile_h, out_col : out_col + out_tile_w] += sr_tile * weight
        weight_sum[out_row : out_row + out_tile_h, out_col : out_col + out_tile_w] += weight
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)  # avoid divide-by-zero on any uncovered pixel
    sr_full = (accumulated / weight_sum).astype(np.float32)
    lr_transform = Affine(*lr_transform_6)
    sr_transform = lr_transform @ Affine.scale(1.0 / scale, 1.0 / scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=out_height, width=out_width, count=num_bands,
        dtype="float32", nodata=lr_nodata,
        crs=lr_crs, transform=sr_transform,
    ) as dst:
        dst.write(sr_full)
    return output_path
def _copy_input_scene(scene: Dict[str, Any], config: Dict[str, Any], output_path: Path) -> Path:
    """Write the scene's real LR band stack, unmodified, as demo_data/input/scene.tif."""
    lr_array, lr_crs, lr_transform_6, lr_nodata = _load_scene_lr_stack(scene, config)
    num_bands, height, width = lr_array.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=num_bands,
        dtype=str(lr_array.dtype), nodata=lr_nodata,
        crs=lr_crs, transform=Affine(*lr_transform_6),
    ) as dst:
        dst.write(lr_array)
    return output_path
def run_demo_output(
    dataset_index_path: Path,
    config: Dict[str, Any],
    model,
) -> Dict[str, str]:
    """Select one scene and write it to the demo_data/ output contract
    (demo_data/input/scene.tif, demo_data/sr/scene_sr.tif).
    Scene selection: config['inference']['scene_id'] if set, otherwise the
    first scene in dataset_index.json that has at least one LR file.
    Raises ValueError if no usable scene is found - this is surfaced to the
    CLI rather than silently producing empty/placeholder demo_data files.
    """
    with open(dataset_index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)
    scenes = [s for s in dataset_index.get("scenes", []) if s.get("lr_files")]
    if not scenes:
        raise ValueError(
            f"No scenes with LR files found in {dataset_index_path}. "
            "Add raw imagery and re-run build_dataset_index.py first."
        )
    requested_scene_id: Optional[str] = config.get("inference", {}).get("scene_id")
    if requested_scene_id:
        matches = [s for s in scenes if s["scene_id"] == requested_scene_id]
        if not matches:
            raise ValueError(f"Requested scene_id '{requested_scene_id}' not found among scenes with LR files.")
        scene = matches[0]
    else:
        scene = scenes[0]
    demo_dir = Path(config["paths"]["demo_data"])
    contract = config["demo_data_contract"]
    input_path = demo_dir / contract["input"]
    sr_path = demo_dir / contract["sr"]
    _copy_input_scene(scene, config, input_path)
    run_scene_inference(scene, config, model, sr_path)
    return {"scene_id": scene["scene_id"], "input": str(input_path), "sr": str(sr_path)}
