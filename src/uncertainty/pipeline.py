"""
Phase 7 orchestration: flip-based TTA ensemble uncertainty estimation.

Runs super-resolution inference on several geometrically-augmented copies
of the demo scene's real LR imagery (see tta.py for which augmentations
and why), undoes each augmentation on its SR output so all predictions are
back in the original orientation, and computes the per-band, per-pixel
standard deviation across the ensemble. That standard deviation map IS the
uncertainty output - not a calibrated statistical confidence, but a
technically defensible signal of where the model's reconstruction is less
stable under equivalent input views (see README.md and the project's
scientific honesty notice).

This always uses the scene's real LR imagery (not the real-HR /
synthetic-proxy distinction from Phase 6 - that distinction is specific to
Phase 6's validation method and does not apply here). It does not
overwrite demo_data/sr/scene_sr.tif - Phase 5/6 already own that file; this
module only adds demo_data/uncertainty/scene_uncertainty.tif alongside it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
from rasterio import Affine

from src.dataset.scene import read_raster_metadata
from src.inference.scene_inference import run_tiled_inference, select_scene
from src.preprocessing.bands import load_band_stack

from .tta import apply_augmentation, invert_augmentation, validate_augmentation_names


def _load_scene_lr_stack(scene: Dict[str, Any], config: Dict[str, Any]):
    """Returns (band_stack_array, crs, transform_6, nodata) for a scene's real LR imagery.

    Duplicated (deliberately, in miniature) from
    src.inference.scene_inference._load_scene_lr_stack, which is private to
    that module. src.validation.pipeline reloads the same way for its
    synthetic-proxy path; keeping this as a small local helper here (rather
    than making the private function public across two unrelated phases)
    keeps each phase's coupling to Phase 5 minimal and explicit.
    """
    lr_metas = [read_raster_metadata(f["path"]) for f in scene["lr_files"]]
    sentinel2_cfg = config.get("sentinel2", {})
    requested_bands = {"red": sentinel2_cfg.get("red_band", "B4"), "nir": sentinel2_cfg.get("nir_band", "B8")}
    band_stack = load_band_stack(lr_metas, requested_bands)
    if not band_stack.ok:
        raise ValueError(f"Could not load LR band stack for scene '{scene['scene_id']}': {band_stack.issues}")
    lr_nodata = lr_metas[0].nodata if lr_metas else None
    return band_stack.data, band_stack.crs, band_stack.transform, lr_nodata


def run_uncertainty_estimation(
    dataset_index_path: Path,
    config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Estimate per-band, per-pixel uncertainty for the demo scene via a
    flip-based TTA ensemble, and write demo_data/uncertainty/scene_uncertainty.tif
    plus data/processed/uncertainty_report.json.

    Returns the report dict that was written to uncertainty_report.json.
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

    uncertainty_cfg = config.get("uncertainty", {})
    augmentation_names: List[str] = uncertainty_cfg.get(
        "augmentations", ["identity", "hflip", "vflip", "hvflip"]
    )
    validate_augmentation_names(augmentation_names)
    if len(augmentation_names) < 2:
        raise ValueError(
            f"uncertainty.augmentations must list at least 2 augmentations to measure "
            f"disagreement across; got {augmentation_names}"
        )

    lr_array, lr_crs, lr_transform_6, lr_nodata = _load_scene_lr_stack(scene, config)

    predictions = []
    sr_transform_6 = None
    for name in augmentation_names:
        augmented = apply_augmentation(lr_array, name)
        sr_array, sr_transform_6 = run_tiled_inference(augmented, lr_crs, lr_transform_6, model, config)
        restored = invert_augmentation(sr_array, name)
        predictions.append(restored)

    shapes = {p.shape for p in predictions}
    if len(shapes) != 1:
        raise ValueError(
            f"TTA ensemble predictions have inconsistent shapes {shapes} - this should not happen "
            "for flip-only augmentations and indicates a bug in run_tiled_inference() or the "
            "augmentation/inversion pair."
        )

    stacked = np.stack(predictions, axis=0)  # (num_augmentations, bands, H, W)
    uncertainty = stacked.std(axis=0)  # (bands, H, W)
    ensemble_mean = stacked.mean(axis=0)  # (bands, H, W) - for report stats only, not written as SR

    demo_dir = Path(config["paths"]["demo_data"])
    contract = config["demo_data_contract"]
    uncertainty_path = demo_dir / contract["uncertainty"]
    uncertainty_path.parent.mkdir(parents=True, exist_ok=True)

    num_bands, out_height, out_width = uncertainty.shape
    with rasterio.open(
        uncertainty_path, "w", driver="GTiff",
        height=out_height, width=out_width, count=num_bands,
        dtype="float32", nodata=lr_nodata,
        crs=lr_crs, transform=Affine(*sr_transform_6),
    ) as dst:
        dst.write(uncertainty.astype(np.float32))

    per_band_stats = [
        {
            "band_index": b,
            "mean_uncertainty": float(uncertainty[b].mean()),
            "max_uncertainty": float(uncertainty[b].max()),
            "mean_prediction": float(ensemble_mean[b].mean()),
        }
        for b in range(num_bands)
    ]

    report = {
        "scene_id": scene["scene_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "augmentations_used": augmentation_names,
        "num_augmentations": len(augmentation_names),
        "shape": list(uncertainty.shape),
        "per_band_stats": per_band_stats,
        "uncertainty_file": str(uncertainty_path),
        "note": (
            "Uncertainty derived from disagreement across a flip-based test-time-augmentation "
            "ensemble - the per-pixel standard deviation of predictions from geometrically "
            "equivalent input views. This is NOT a calibrated statistical confidence interval; "
            "high-uncertainty areas indicate regions where the model's reconstruction is less "
            "stable and should be interpreted cautiously, not as proof that a feature is real or fake."
        ),
    }

    processed_dir = Path(config["paths"]["data_processed"])
    report_filename = uncertainty_cfg.get("output_report_filename", "uncertainty_report.json")
    report_path = processed_dir / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["report_path"] = str(report_path)
    return report