"""
Standard demo output contract: manifest generation and validation.

Phase 0 established demo_data/manifest.json as a hand-written static file
listing relative paths to each pipeline output, matching config.yaml's
demo_data_contract section. This module replaces "hand-written" with
"derived from config.yaml" - the manifest's keys and values are identical
(this is not a redesign, see README.md), but generating it from the single
source of truth (config.yaml) instead of maintaining two copies by hand
means they can never silently drift apart.

validate_demo_outputs() is the other half of Phase 9's job: after every
phase has run, confirm every file the manifest points to actually exists
and is openable, so a broken or partial pipeline run is caught here rather
than surfacing later as a confusing dashboard error.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import rasterio


def build_manifest(config: Dict[str, Any]) -> Dict[str, str]:
    """Return the manifest dict - identical in shape to config.yaml's
    demo_data_contract section, which is itself identical to Phase 0's
    original demo_data/manifest.json.
    """
    return dict(config["demo_data_contract"])


def write_manifest(config: Dict[str, Any], output_path: Path) -> Dict[str, str]:
    """Build and write the manifest to output_path. Returns the manifest dict."""
    manifest = build_manifest(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


_RASTER_KEYS = {"scene", "sr", "uncertainty", "original_ndvi", "sr_ndvi"}
_JSON_KEYS = {"metrics"}
_REQUIRED_METRICS_KEYS = {"psnr", "ssim", "rmse", "sam"}


def validate_demo_outputs(config: Dict[str, Any]) -> List[str]:
    """Check that every file listed in the manifest actually exists and is
    openable. Returns a list of human-readable issues - empty means every
    contract file is present and valid.

    Raster entries (scene, sr, uncertainty, original_ndvi, sr_ndvi) are
    opened with rasterio to catch a truncated/corrupt file, not just a
    missing one. The metrics entry is checked as valid JSON containing the
    required keys (psnr, ssim, rmse, sam) per the project's metrics.json
    contract - not that the values are non-placeholder, since a freshly
    computed 0.0 (a real, legitimate metric value) is indistinguishable
    from a placeholder by value alone.
    """
    issues: List[str] = []
    demo_dir = Path(config["paths"]["demo_data"])
    manifest = build_manifest(config)

    for key, relative_path in manifest.items():
        full_path = demo_dir / relative_path

        if not full_path.exists():
            issues.append(f"'{key}': expected file not found at {full_path}")
            continue

        if key in _RASTER_KEYS:
            try:
                with rasterio.open(full_path) as src:
                    if src.width <= 0 or src.height <= 0 or src.count <= 0:
                        issues.append(f"'{key}': raster at {full_path} has invalid dimensions/band count")
            except Exception as exc:  # noqa: BLE001 - report any read failure, do not crash validation
                issues.append(f"'{key}': failed to open raster at {full_path}: {exc}")
        elif key in _JSON_KEYS:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                missing_keys = _REQUIRED_METRICS_KEYS - set(data.keys())
                if missing_keys:
                    issues.append(f"'{key}': {full_path} is missing required key(s) {missing_keys}")
            except Exception as exc:  # noqa: BLE001
                issues.append(f"'{key}': failed to read/parse JSON at {full_path}: {exc}")

    return issues