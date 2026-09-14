"""
Phase 6 orchestration: compute PSNR/SSIM/RMSE/SAM for the demo scene and
write both the stable demo_data/metrics/metrics.json contract (Phase 0)
and a fuller data/processed/validation_report.json audit trail.

Two mutually-exclusive comparison paths, matching Phase 2's scene typing
(preprocessing_index.json's is_synthetic_hr / alignment fields):

1. REAL-HR scenes (an HR file passed Phase 2's check_alignment):
   SR is computed from the scene's real LR imagery (refreshing
   demo_data/sr/scene_sr.tif via run_scene_inference, so the SR raster
   being validated is never stale). The real HR reference is then
   resampled onto the SR raster's *exact* grid - Phase 2 never does this
   pixel-level step, only a metadata/bounds-level check, so Phase 6 owns
   it (see resample_to_reference()). This is the only path that measures
   reconstruction against genuine, independently-sourced ground truth.

2. SYNTHETIC-HR (proxy) scenes (no real HR; synthetic_hr_fallback was
   used): SR is computed directly from synthetic_lr.tif (Phase 2's
   deliberately-degraded raster) via run_tiled_inference(), and compared
   against proxy_hr (the scene's real LR imagery - the same array
   synthetic_hr.py degraded from). This measures how well the model
   reconstructs a *known, deliberately-applied degradation* - it is
   explicitly NOT a measurement of real-world HR recovery, and is labeled
   as such in every output this module produces (comparison_type =
   "synthetic_degradation_proxy"), per the project's scientific-honesty
   requirements (see README.md and data/README.md).

If a scene has neither an aligned real HR entry nor a synthetic entry
(e.g. fallback disabled and no real HR available), run_validation() raises
ValueError with a clear message rather than fabricating a result.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import Resampling, reproject

from src.inference.scene_inference import run_scene_inference, run_tiled_inference, select_scene
from src.preprocessing.bands import load_band_stack
from src.dataset.scene import read_raster_metadata

from .metrics import compute_psnr, compute_rmse, compute_sam, compute_ssim


def resample_to_reference(
    src_path: str,
    ref_crs: Optional[str],
    ref_transform_6: Tuple[float, float, float, float, float, float],
    ref_width: int,
    ref_height: int,
) -> np.ndarray:
    """Reproject/resample a raster onto an exact target grid (CRS, transform, size).

    Distinct from src.preprocessing.alignment.reproject_hr_to_lr_grid, which
    targets the LR grid at the LR raster's own resolution. This targets the
    SR output's exact grid (already upsampled to LR resolution * scale) -
    Phase 2 never produces this pixel-aligned comparison raster, since its
    alignment check is metadata/bounds-level only. Phase 6 owns this step
    because it is validation-specific.

    Destination pixels outside the source raster's coverage are filled with
    NaN so they can be excluded from metric calculations rather than
    silently treated as 0, which would bias every metric.
    """
    with rasterio.open(src_path) as src:
        dest = np.full((src.count, ref_height, ref_width), np.nan, dtype=np.float64)
        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dest[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=Affine(*ref_transform_6),
                dst_crs=ref_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return dest


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """(H, W) boolean mask: True where every given (bands, H, W) array is finite in all bands."""
    mask = None
    for arr in arrays:
        m = np.all(np.isfinite(arr), axis=0)
        mask = m if mask is None else (mask & m)
    return mask


def _nodata_mask(*arrays: np.ndarray, nodata: Optional[float]) -> Optional[np.ndarray]:
    """(H, W) boolean mask: True where every array's pixel != nodata in all bands. None if nodata is None."""
    if nodata is None:
        return None
    mask = None
    for arr in arrays:
        m = np.all(arr != nodata, axis=0)
        mask = m if mask is None else (mask & m)
    return mask


def _combine_masks(*masks: Optional[np.ndarray]) -> Optional[np.ndarray]:
    result = None
    for m in masks:
        if m is None:
            continue
        result = m if result is None else (result & m)
    return result


def _center_crop_to_common_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop two (bands, H, W) arrays to their common (min height, min width),
    centered. Used for the synthetic path, where block-mean downsampling in
    synthetic_hr.py can crop a remainder that SR's exact integer upsampling
    does not perfectly re-create.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Band count mismatch: {a.shape[0]} vs {b.shape[0]}")
    h = min(a.shape[1], b.shape[1])
    w = min(a.shape[2], b.shape[2])

    def _crop(arr: np.ndarray) -> np.ndarray:
        top = (arr.shape[1] - h) // 2
        left = (arr.shape[2] - w) // 2
        return arr[:, top : top + h, left : left + w]

    return _crop(a), _crop(b)


def _compute_all_metrics(sr: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray]) -> Dict[str, float]:
    return {
        "psnr": compute_psnr(sr, ref, mask=mask),
        "ssim": compute_ssim(sr, ref, mask=mask),
        "rmse": compute_rmse(sr, ref, mask=mask),
        "sam": compute_sam(sr, ref, mask=mask) if sr.shape[0] >= 2 else None,
    }


def _find_preprocessing_entry(preprocessing_index: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    for entry in preprocessing_index.get("scenes", []):
        if entry.get("scene_id") == scene_id:
            return entry
    raise ValueError(
        f"Scene '{scene_id}' not found in preprocessing_index.json. "
        "Run `python scripts/run_preprocessing.py` first (Phase 2)."
    )


def _validate_real_hr_scene(
    scene: Dict[str, Any],
    pp_entry: Dict[str, Any],
    config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    aligned_entries = [a for a in pp_entry["alignment"] if a["is_aligned"]]
    if not aligned_entries:
        raise ValueError(
            f"Scene '{scene['scene_id']}' has HR file(s) but none passed Phase 2's alignment "
            "check - cannot validate against real HR. See preprocessing_index.json's 'alignment' issues."
        )
    hr_file_path = aligned_entries[0]["hr_file"]

    demo_dir = Path(config["paths"]["demo_data"])
    contract = config["demo_data_contract"]
    sr_path = demo_dir / contract["sr"]
    run_scene_inference(scene, config, model, sr_path)

    with rasterio.open(sr_path) as src:
        sr_array = src.read().astype(np.float64)
        sr_crs = src.crs.to_string() if src.crs else None
        sr_transform_6 = tuple(src.transform)[:6]
        sr_nodata = src.nodata

    hr_array = resample_to_reference(hr_file_path, sr_crs, sr_transform_6, sr_array.shape[2], sr_array.shape[1])

    if hr_array.shape[0] != sr_array.shape[0]:
        raise ValueError(
            f"Scene '{scene['scene_id']}': real HR reference has {hr_array.shape[0]} band(s) but SR "
            f"output has {sr_array.shape[0]} - this validation path assumes matching band count/order "
            "between the HR reference and the SR output (documented assumption; not yet verified "
            "against a real Sentinel-2 real-HR pair)."
        )

    nodata_cfg = config.get("validation", {}).get("nodata_exclude", True)
    mask = _finite_mask(sr_array, hr_array)
    if nodata_cfg:
        mask = _combine_masks(mask, _nodata_mask(sr_array, hr_array, nodata=sr_nodata))

    metrics = _compute_all_metrics(sr_array, hr_array, mask)

    return {
        "comparison_type": "real_hr_reference",
        "hr_file": hr_file_path,
        "sr_file": str(sr_path),
        "shape": list(sr_array.shape),
        "valid_pixel_fraction": float(mask.mean()) if mask is not None else 1.0,
        "metrics": metrics,
        "note": (
            "SR output compared against a real, independently-sourced HR reference "
            "resampled onto the SR output's exact grid. This is a genuine reconstruction "
            "accuracy measurement."
        ),
    }


def _validate_synthetic_hr_scene(
    scene: Dict[str, Any],
    pp_entry: Dict[str, Any],
    config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    synth = pp_entry["synthetic"]
    synthetic_lr_path = synth["synthetic_lr_path"]

    with rasterio.open(synthetic_lr_path) as src:
        synthetic_lr_array = src.read().astype(np.float64)
        synthetic_lr_crs = src.crs.to_string() if src.crs else None
        synthetic_lr_transform_6 = tuple(src.transform)[:6]

    sr_array, _sr_transform_6 = run_tiled_inference(
        synthetic_lr_array, synthetic_lr_crs, synthetic_lr_transform_6, model, config
    )

    sentinel2_cfg = config.get("sentinel2", {})
    requested_bands = {"red": sentinel2_cfg.get("red_band", "B4"), "nir": sentinel2_cfg.get("nir_band", "B8")}
    lr_metas = [read_raster_metadata(f["path"]) for f in scene["lr_files"]]
    band_stack = load_band_stack(lr_metas, requested_bands)
    if not band_stack.ok:
        raise ValueError(
            f"Could not reload proxy HR (real LR) band stack for scene '{scene['scene_id']}': {band_stack.issues}"
        )
    proxy_hr_array = band_stack.data.astype(np.float64)
    lr_nodata = lr_metas[0].nodata if lr_metas else None

    sr_cropped, proxy_cropped = _center_crop_to_common_shape(sr_array.astype(np.float64), proxy_hr_array)

    nodata_cfg = config.get("validation", {}).get("nodata_exclude", True)
    mask = _finite_mask(sr_cropped, proxy_cropped)
    if nodata_cfg:
        mask = _combine_masks(mask, _nodata_mask(sr_cropped, proxy_cropped, nodata=lr_nodata))

    metrics = _compute_all_metrics(sr_cropped, proxy_cropped, mask)

    return {
        "comparison_type": "synthetic_degradation_proxy",
        "synthetic_lr_file": synthetic_lr_path,
        "proxy_hr_files": synth["proxy_hr_paths"],
        "shape": list(sr_cropped.shape),
        "valid_pixel_fraction": float(mask.mean()) if mask is not None else 1.0,
        "metrics": metrics,
        "note": (
            "SR output computed from a synthetically-degraded raster (known Gaussian blur + "
            "downsample applied to real imagery) and compared against that real imagery. This "
            "measures reconstruction of a known, deliberately-applied degradation - it is NOT a "
            "measurement of real-world HR recovery, since no independent HR ground truth exists "
            "for this scene."
        ),
    }


def run_validation(
    dataset_index_path: Path,
    preprocessing_index_path: Path,
    config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Validate the demo scene (same selection rule as run_demo_output) and
    write both demo_data/metrics/metrics.json and
    data/processed/validation_report.json.

    Returns the full report dict that was written to validation_report.json.
    """
    with open(dataset_index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)
    with open(preprocessing_index_path, "r", encoding="utf-8") as f:
        preprocessing_index = json.load(f)

    scenes = [s for s in dataset_index.get("scenes", []) if s.get("lr_files")]
    if not scenes:
        raise ValueError(
            f"No scenes with LR files found in {dataset_index_path}. "
            "Add raw imagery and re-run build_dataset_index.py first."
        )
    scene = select_scene(scenes, config)
    pp_entry = _find_preprocessing_entry(preprocessing_index, scene["scene_id"])

    has_aligned_real_hr = any(a["is_aligned"] for a in pp_entry.get("alignment", []))
    if has_aligned_real_hr:
        result = _validate_real_hr_scene(scene, pp_entry, config, model)
    elif pp_entry.get("is_synthetic_hr"):
        result = _validate_synthetic_hr_scene(scene, pp_entry, config, model)
    else:
        raise ValueError(
            f"Scene '{scene['scene_id']}' has neither an aligned real HR reference nor a synthetic "
            "HR fallback entry in preprocessing_index.json - nothing to validate against. Check "
            "config.yaml's dataset.synthetic_hr_fallback.enabled and re-run Phase 2."
        )

    report = {
        "scene_id": scene["scene_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    processed_dir = Path(config["paths"]["data_processed"])
    report_filename = config.get("validation", {}).get("output_report_filename", "validation_report.json")
    report_path = processed_dir / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    demo_dir = Path(config["paths"]["demo_data"])
    contract = config["demo_data_contract"]
    metrics_path = demo_dir / contract["metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "psnr": result["metrics"]["psnr"],
        "ssim": result["metrics"]["ssim"],
        "rmse": result["metrics"]["rmse"],
        "sam": result["metrics"]["sam"],
        "scene_id": scene["scene_id"],
        "comparison_type": result["comparison_type"],
        "note": result["note"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    report["report_path"] = str(report_path)
    report["metrics_path"] = str(metrics_path)
    return report