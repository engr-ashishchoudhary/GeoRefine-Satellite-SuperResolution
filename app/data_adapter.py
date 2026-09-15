"""
Phase 10 data adapter: the ONLY module in app/ that knows about
demo_data/'s file layout and config.yaml's paths. Every other dashboard
component (API routes, frontend JS) works with the normalized dicts this
module returns - not raw file paths - per the project's dashboard
independence requirement (see README.md's "Dashboard Independence"
section).

This module is read-only: it never runs inference, computes metrics, or
writes any file. If demo_data/ has not been populated yet (scripts/
run_full_pipeline.py has not been run), every function here reports that
clearly rather than raising an unhandled exception or fabricating
placeholder data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import rasterio


def _demo_dir(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["demo_data"])


def load_manifest(config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Return the manifest dict from demo_data/manifest.json, or None if it
    doesn't exist yet (the demo pipeline has not been run).
    """
    manifest_filename = config.get("pipeline", {}).get("manifest_filename", "manifest.json")
    manifest_path = _demo_dir(config) / manifest_filename
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_demo_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Report which demo_data contract files actually exist on disk right
    now. Uses config.yaml's demo_data_contract directly (not the possibly-
    missing manifest), so status reporting works even before manifest.json
    has ever been generated.

    Returns a dict with:
      - "manifest_available": bool
      - "files": {key: {"path": str, "exists": bool}} for every contract entry
      - "ready": bool - True only if every contract file exists
    """
    demo_dir = _demo_dir(config)
    contract = config.get("demo_data_contract", {})
    manifest = load_manifest(config)

    files = {}
    for key, relative_path in contract.items():
        full_path = demo_dir / relative_path
        files[key] = {"path": str(full_path), "exists": full_path.exists()}

    return {
        "manifest_available": manifest is not None,
        "files": files,
        "ready": bool(files) and all(f["exists"] for f in files.values()),
    }


def load_metrics(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return demo_data/metrics/metrics.json's contents, or None if missing."""
    contract = config.get("demo_data_contract", {})
    if "metrics" not in contract:
        return None
    metrics_path = _demo_dir(config) / contract["metrics"]
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raster_summary(config: Dict[str, Any], contract_key: str) -> Optional[Dict[str, Any]]:
    """Return lightweight metadata (no pixel data) for a raster contract
    entry - width, height, band count, CRS - or None if the file doesn't
    exist. Used to confirm a raster is present and readable without
    loading its full pixel array; later visualization phases (11-14) will
    load actual pixel data through their own logic, not this function.
    """
    contract = config.get("demo_data_contract", {})
    if contract_key not in contract:
        return None
    raster_path = _demo_dir(config) / contract[contract_key]
    if not raster_path.exists():
        return None
    with rasterio.open(raster_path) as src:
        return {
            "path": str(raster_path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": src.crs.to_string() if src.crs else None,
        }