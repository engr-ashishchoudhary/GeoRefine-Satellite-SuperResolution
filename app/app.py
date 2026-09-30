"""
Phase 10-15: FastAPI dashboard.

Phase 10 established the foundation. Phases 11-14 added before/after,
metrics, NDVI, and uncertainty visualization. Phase 15 adds explicit mode
labeling ("demo" vs. a future "live" mode from Phase 16) so cached,
precomputed data is never presented as if it were live processing (see
app/README.md's Phase 15 section and the project's section 22/42
scientific-honesty requirements). This module never reads demo_data/ or
data/processed/ file paths directly for status/metadata - it only calls
app.data_adapter functions - but DOES call app.rendering (which reads
pixel data) for the render endpoints, since pixel rendering is that
phase's job.

Config path is overridable via the GEOREFINE_CONFIG environment variable
so tests can point at an isolated temporary config without touching the
real project config.yaml.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_adapter import (
    describe_metrics,
    get_app_mode,
    get_demo_status,
    load_ndvi_report,
    load_raster_summary,
    load_uncertainty_report,
)
from app.rendering import (
    render_ndvi_raster_file_as_png,
    render_raster_file_as_png,
    render_uncertainty_raster_file_as_png,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR.parent / "config.yaml"

_FALSE_COLOR_KEYS = ("input", "sr")
_NDVI_KEYS = ("original_ndvi", "sr_ndvi")

_MODE_DESCRIPTIONS = {
    "demo": (
        "Demo Mode: all data shown is precomputed and committed to the repository. "
        "No live inference occurs in this view."
    ),
    "live": "Live Mode: results are computed from a scene provided in this session.",
}


def load_config() -> dict:
    config_path = Path(os.environ.get("GEOREFINE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


app = FastAPI(title="GeoRefine Dashboard", version="0.6.0")


@app.get("/api/mode")
def api_mode():
    config = load_config()
    mode = get_app_mode(config)
    return {"mode": mode, "description": _MODE_DESCRIPTIONS.get(mode, "")}


@app.get("/api/status")
def api_status():
    config = load_config()
    return get_demo_status(config)


@app.get("/api/metrics")
def api_metrics():
    config = load_config()
    return describe_metrics(config)


@app.get("/api/raster-info")
def api_raster_info():
    config = load_config()
    return {key: load_raster_summary(config, key) for key in _FALSE_COLOR_KEYS}


@app.get("/api/ndvi-info")
def api_ndvi_info():
    config = load_config()
    return {
        "original_ndvi": load_raster_summary(config, "original_ndvi"),
        "sr_ndvi": load_raster_summary(config, "sr_ndvi"),
        "report": load_ndvi_report(config),
    }


@app.get("/api/uncertainty-info")
def api_uncertainty_info():
    config = load_config()
    return {
        "uncertainty": load_raster_summary(config, "uncertainty"),
        "report": load_uncertainty_report(config),
    }


def _resolve_raster_path(config: dict, key: str) -> Path:
    contract = config.get("demo_data_contract", {})
    if key not in contract:
        raise HTTPException(status_code=404, detail=f"'{key}' is not in demo_data_contract")
    demo_dir = Path(config["paths"]["demo_data"])
    raster_path = demo_dir / contract[key]
    if not raster_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{key}' raster not found at {raster_path}. Run scripts/run_full_pipeline.py first.",
        )
    return raster_path


@app.get("/api/render/{key}")
def api_render(key: str):
    if key not in _FALSE_COLOR_KEYS:
        raise HTTPException(
            status_code=404, detail=f"Unknown render key '{key}'. Expected one of {_FALSE_COLOR_KEYS}"
        )
    config = load_config()
    raster_path = _resolve_raster_path(config, key)

    viz_cfg = config.get("visualization", {})
    low = viz_cfg.get("stretch_low_percentile", 2.0)
    high = viz_cfg.get("stretch_high_percentile", 98.0)

    try:
        png_bytes = render_raster_file_as_png(raster_path, low, high)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/render-ndvi/{key}")
def api_render_ndvi(key: str):
    if key not in _NDVI_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown NDVI render key '{key}'. Expected one of {_NDVI_KEYS}")
    config = load_config()
    raster_path = _resolve_raster_path(config, key)

    viz_cfg = config.get("visualization", {})
    ndvi_min = viz_cfg.get("ndvi_min", -1.0)
    ndvi_max = viz_cfg.get("ndvi_max", 1.0)

    try:
        png_bytes = render_ndvi_raster_file_as_png(raster_path, ndvi_min, ndvi_max)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/render-uncertainty")
def api_render_uncertainty():
    config = load_config()
    raster_path = _resolve_raster_path(config, "uncertainty")

    viz_cfg = config.get("visualization", {})
    low = viz_cfg.get("stretch_low_percentile", 2.0)
    high = viz_cfg.get("stretch_high_percentile", 98.0)

    try:
        png_bytes = render_uncertainty_raster_file_as_png(raster_path, low, high)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=png_bytes, media_type="image/png")


app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(APP_DIR / "static" / "index.html"))


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    app_cfg = cfg.get("app", {})
    uvicorn.run(
        "app.app:app",
        host=app_cfg.get("host", "127.0.0.1"),
        port=app_cfg.get("port", 8000),
        reload=True,
    )