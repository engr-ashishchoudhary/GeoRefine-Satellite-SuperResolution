"""
Phase 8 orchestration: compute NDVI for the demo scene's original (real LR)
imagery and its SR output, from actual raster calculations.

Two separate NDVI rasters are produced, at their respective native
resolutions:
  - original_ndvi.tif: from the scene's real LR imagery, at LR resolution.
  - sr_ndvi.tif: from the SR output (refreshed via run_scene_inference, so
    it is never stale), at SR resolution (LR resolution * model.scale).

A direct pixel-wise NDVI difference map is deliberately NOT produced here -
see src/downstream/README.md's "Known limitations" for why (the two
rasters are at different resolutions; reconciling that is a dashboard
visualization concern for Phase 13, not this phase's job). Both rasters
are independently real, computed NDVI - suitable for visual/statistical
comparison as-is.

Requires the scene's LR files to have a known (non-"unknown") band order
from src.preprocessing.bands.load_band_stack - i.e. matched single-band
files, not an unverified pre-merged composite. This is the same
restriction bands.py already documents; Phase 8 does not add a new one.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio import Affine

from src.dataset.scene import read_raster_metadata
from src.inference.scene_inference import run_scene_inference, select_scene
from src.preprocessing.bands import load_band_stack

from .ndvi import compute_ndvi


def _band_indices(band_order: list) -> Dict[str, int]:
    """Map logical band names ('red', 'nir') to their index in a stacked array.

    Raises ValueError if band_order is not exactly the resolved logical
    names (e.g. is ["unknown", "unknown"] for an unverified composite
    file), since NDVI requires knowing which array index is which band.
    """
    if "red" not in band_order or "nir" not in band_order:
        raise ValueError(
            f"Cannot compute NDVI: band order {band_order} does not contain resolved 'red' and "
            "'nir' bands. This happens for a pre-merged multi-band composite LR file, whose band "
            "order load_band_stack() cannot verify (see src/preprocessing/bands.py). NDVI requires "
            "single-band LR files matched by filename."
        )
    return {"red": band_order.index("red"), "nir": band_order.index("nir")}


def _write_ndvi_raster(
    ndvi: np.ndarray, crs, transform_6, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=ndvi.shape[0], width=ndvi.shape[1], count=1,
        dtype="float32", nodata=np.nan,
        crs=crs, transform=Affine(*transform_6),
    ) as dst:
        dst.write(ndvi.astype(np.float32), 1)
    return output_path


def _ndvi_stats(ndvi: np.ndarray) -> Dict[str, float]:
    valid = ~np.isnan(ndvi)
    if not valid.any():
        return {"mean": None, "min": None, "max": None, "valid_pixel_fraction": 0.0}
    return {
        "mean": float(ndvi[valid].mean()),
        "min": float(ndvi[valid].min()),
        "max": float(ndvi[valid].max()),
        "valid_pixel_fraction": float(valid.mean()),
    }


def run_ndvi_analysis(
    dataset_index_path: Path,
    config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Compute original and SR NDVI for the demo scene, writing both
    GeoTIFFs to the demo_data/ contract and a report to
    data/processed/ndvi_report.json.

    Returns the report dict that was written.
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

    sentinel2_cfg = config.get("sentinel2", {})
    requested_bands = {"red": sentinel2_cfg.get("red_band", "B4"), "nir": sentinel2_cfg.get("nir_band", "B8")}
    lr_metas = [read_raster_metadata(f["path"]) for f in scene["lr_files"]]
    band_stack = load_band_stack(lr_metas, requested_bands)
    if not band_stack.ok:
        raise ValueError(f"Could not load LR band stack for scene '{scene['scene_id']}': {band_stack.issues}")

    indices = _band_indices(band_stack.band_order)
    original_red = band_stack.data[indices["red"]]
    original_nir = band_stack.data[indices["nir"]]
    original_ndvi = compute_ndvi(original_red, original_nir)

    demo_dir = Path(config["paths"]["demo_data"])
    contract = config["demo_data_contract"]

    original_ndvi_path = demo_dir / contract["original_ndvi"]
    _write_ndvi_raster(original_ndvi, band_stack.crs, band_stack.transform, original_ndvi_path)

    sr_path = demo_dir / contract["sr"]
    run_scene_inference(scene, config, model, sr_path)

    with rasterio.open(sr_path) as src:
        sr_array = src.read().astype(np.float64)
        sr_crs = src.crs.to_string() if src.crs else None
        sr_transform_6 = tuple(src.transform)[:6]

    sr_red = sr_array[indices["red"]]
    sr_nir = sr_array[indices["nir"]]
    sr_ndvi = compute_ndvi(sr_red, sr_nir)

    sr_ndvi_path = demo_dir / contract["sr_ndvi"]
    _write_ndvi_raster(sr_ndvi, sr_crs, sr_transform_6, sr_ndvi_path)

    report = {
        "scene_id": scene["scene_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_ndvi": {
            "file": str(original_ndvi_path),
            "shape": list(original_ndvi.shape),
            "stats": _ndvi_stats(original_ndvi),
        },
        "sr_ndvi": {
            "file": str(sr_ndvi_path),
            "shape": list(sr_ndvi.shape),
            "stats": _ndvi_stats(sr_ndvi),
        },
        "note": (
            "Both NDVI rasters are computed from real pixel values (original LR imagery and the "
            "model's SR output respectively) - no NDVI values are fabricated. The SR-derived NDVI "
            "is based on reconstructed, super-resolved imagery, not directly observed satellite "
            "pixels, and should be interpreted with that in mind. The two rasters are at different "
            "resolutions (LR vs. LR * model.scale) and are not pixel-aligned or differenced here - "
            "see src/downstream/README.md's known limitations."
        ),
    }

    processed_dir = Path(config["paths"]["data_processed"])
    report_filename = config.get("downstream", {}).get("output_report_filename", "ndvi_report.json")
    report_path = processed_dir / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["report_path"] = str(report_path)
    return report
