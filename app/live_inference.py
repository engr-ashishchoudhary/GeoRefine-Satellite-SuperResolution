"""
Phase 16: live inference - run real super-resolution on a user-uploaded
scene, on demand, rather than reading precomputed demo_data/ files.

This is the one module in app/ that actually invokes the model
(src.model.super_resolution.load_model,
src.inference.scene_inference.run_tiled_inference) rather than reading
files data_adapter.py already prepared. Kept separate from
data_adapter.py (read-only, no inference) and rendering.py (pixel-to-PNG
only, no inference) since running the actual model is a distinct concern.

Simplifications, documented rather than hidden (prototype scope, single-
process localhost dashboard - see app/README.md's Phase 16 known
limitations):
  - Accepts one uploaded GeoTIFF with at least 2 bands (band 1 = red, band
    2 = nir), the same convention used throughout src/ and
    app/rendering.py - not two separate single-band files.
  - The model is loaded once and cached in a module-level variable, not
    reloaded per request.
  - Only ONE live result is held in memory at a time (the most recent
    upload) - this is not a per-session/per-user store. A second upload
    (by the same or a different user) overwrites the previous live
    result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio import Affine

from src.inference.scene_inference import run_tiled_inference
from src.model.super_resolution import load_model

_cached_model = None
_cached_model_checkpoint_path: Optional[str] = None

_live_result: Dict[str, Any] = {
    "input_path": None,
    "sr_path": None,
    "scene_shape": None,
    "sr_shape": None,
    "original_filename": None,
}


def get_live_result() -> Dict[str, Any]:
    """Return the most recent live inference result's file paths/shapes, or
    a dict of Nones if no live inference has been run yet in this process.
    """
    return dict(_live_result)


def _get_model(config: Dict[str, Any]):
    """Load the pretrained model once and cache it for reuse across
    requests. Re-loads only if config.yaml's models path has changed since
    the last load.
    """
    global _cached_model, _cached_model_checkpoint_path
    models_dir = config["paths"]["models"]
    if _cached_model is not None and _cached_model_checkpoint_path == models_dir:
        return _cached_model
    model = load_model(models_dir=models_dir)  # raises FileNotFoundError if checkpoint missing
    _cached_model = model
    _cached_model_checkpoint_path = models_dir
    return model


def run_live_inference(uploaded_file_bytes: bytes, original_filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Save an uploaded scene, run real SR inference on it, and cache the
    result for the dashboard's live-mode render endpoints.

    Raises FileNotFoundError if the pretrained checkpoint is not available
    (propagated from load_model). Raises ValueError for invalid uploads
    (wrong band count, unreadable file).

    Returns the same shape as get_live_result().
    """
    live_cfg = config.get("live", {})
    upload_dir = Path(live_cfg.get("upload_dir", "data/live/uploads"))
    output_dir = Path(live_cfg.get("output_dir", "data/live/outputs"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = upload_dir / "current_upload.tif"
    with open(input_path, "wb") as f:
        f.write(uploaded_file_bytes)

    try:
        with rasterio.open(input_path) as src:
            if src.count < 2:
                raise ValueError(
                    f"Uploaded file has {src.count} band(s); at least 2 are required "
                    "(band 1 = red, band 2 = nir, matching this project's Sentinel-2 convention)."
                )
            array = src.read()
            crs = src.crs.to_string() if src.crs else None
            transform_6 = tuple(src.transform)[:6]
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - any rasterio read failure becomes a clear ValueError
        raise ValueError(f"Could not read '{original_filename}' as a GeoTIFF: {exc}")

    model = _get_model(config)

    sr_array, sr_transform_6 = run_tiled_inference(array, crs, transform_6, model, config)

    sr_path = output_dir / "current_sr.tif"
    num_bands, out_height, out_width = sr_array.shape
    with rasterio.open(
        sr_path, "w", driver="GTiff",
        height=out_height, width=out_width, count=num_bands,
        dtype="float32", crs=crs, transform=Affine(*sr_transform_6),
    ) as dst:
        dst.write(sr_array.astype(np.float32))

    global _live_result
    _live_result = {
        "input_path": str(input_path),
        "sr_path": str(sr_path),
        "scene_shape": list(array.shape),
        "sr_shape": list(sr_array.shape),
        "original_filename": original_filename,
    }
    return dict(_live_result)