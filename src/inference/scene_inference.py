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

This module produces:
  1. run_tiled_inference() - the shared, array-based core: given an
     in-memory (bands, H, W) array, its CRS/transform, and a loaded model,
     returns the tiled/blended SR array and its transform. Introduced in
     Phase 6 so validation can run the same tiling/blending logic directly
     on rasters that are not "scenes" in the dataset_index.json sense
     (e.g. src/preprocessing's synthetic_lr.tif), without duplicating it.
  2. run_scene_inference() - the general-purpose, scene-based function:
     given any scene dict (from dataset_index.json) and a loaded model,
     produces a georeferenced SR GeoTIFF at an arbitrary output path. Now a
     thin wrapper around run_tiled_inference() plus GeoTIFF writing.
  3. select_scene() - given a non-empty list of scene dicts, picks one per
     config.yaml's inference.scene_id, or the first if unset. Extracted
     from run_demo_output() so Phase 6 can reuse the same selection logic
     for the same demo scene, without duplicating it.
  4. run_demo_output() - a thin wrapper that selects one scene and writes
     it to the project's stable demo_data/ output contract
     (demo_data/input/scene.tif, demo_data/sr/scene_sr.tif), established
     in Phase 0 and unchanged since.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def run_tiled_inference(
    array: np.ndarray,
    crs: Optional[str],
    transform_6: Tuple[float, float, float, float, float, float],
    model,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]:
    """Run tiled super-resolution on an in-memory (bands, H, W) array.

    Returns (sr_array, sr_transform_6). The array is NOT written to disk -
    callers decide the output path, dtype, and nodata handling. This is the
    shared core used both by run_scene_inference() (a scene's real LR
    imagery) and by Phase 6 validation (synthetic_lr.tif rasters), so the
    tiling/blending logic lives in exactly one place.
    """
    num_bands, height, width = array.shape
    inference_cfg = config.get("inference", {})
    tile_size = inference_cfg.get("tile_size", 512)
    overlap = inference_cfg.get("tile_overlap", 32)
    scale = model.scale
    out_height, out_width = height * scale, width * scale
    accumulated = np.zeros((num_bands, out_height, out_width), dtype=np.float64)
    weight_sum = np.zeros((out_height, out_width), dtype=np.float64)
    tiles = compute_tile_grid(height, width, tile_size, overlap)
    for row, col, tile_h, tile_w in tiles:
        lr_tile = array[:, row : row + tile_h, col : col + tile_w]
        sr_tile = apply_super_resolution(model, lr_tile)  # (bands, tile_h*scale, tile_w*scale)
        weight = blend_weight_mask(tile_h * scale, tile_w * scale, overlap * scale)
        out_row, out_col = row * scale, col * scale
        out_tile_h, out_tile_w = sr_tile.shape[1], sr_tile.shape[2]
        accumulated[:, out_row : out_row + out_tile_h, out_col : out_col + out_tile_w] += sr_tile * weight
        weight_sum[out_row : out_row + out_tile_h, out_col : out_col + out_tile_w] += weight
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)  # avoid divide-by-zero on any uncovered pixel
    sr_array = (accumulated / weight_sum).astype(np.float32)
    src_transform = Affine(*transform_6)
    sr_transform = src_transform @ Affine.scale(1.0 / scale, 1.0 / scale)
    return sr_array, tuple(sr_transform)[:6]


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
    sr_array, sr_transform_6 = run_tiled_inference(lr_array, lr_crs, lr_transform_6, model, config)
    num_bands, out_height, out_width = sr_array.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=out_height, width=out_width, count=num_bands,
        dtype="float32", nodata=lr_nodata,
        crs=lr_crs, transform=Affine(*sr_transform_6),
    ) as dst:
        dst.write(sr_array)
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


def select_scene(scenes: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Given a non-empty list of scene dicts (each with at least one LR
    file), pick one according to config['inference']['scene_id'], or the
    first if unset. Raises ValueError if a requested scene_id is not found.

    Assumes `scenes` has already been filtered to entries with lr_files -
    callers are responsible for that filtering and for handling the
    "no scenes at all" case with their own contextual error message.
    """
    requested_scene_id: Optional[str] = config.get("inference", {}).get("scene_id")
    if requested_scene_id:
        matches = [s for s in scenes if s["scene_id"] == requested_scene_id]
        if not matches:
            raise ValueError(f"Requested scene_id '{requested_scene_id}' not found among scenes with LR files.")
        return matches[0]
    return scenes[0]


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
    scene = select_scene(scenes, config)

    demo_dir = Path(config["paths"]["demo_data"])
    contract = config["demo_data_contract"]
    input_path = demo_dir / contract["input"]
    sr_path = demo_dir / contract["sr"]
    _copy_input_scene(scene, config, input_path)
    run_scene_inference(scene, config, model, sr_path)
    return {"scene_id": scene["scene_id"], "input": str(input_path), "sr": str(sr_path)}