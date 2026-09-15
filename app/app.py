"""
Phase 10/11: FastAPI dashboard.

Phase 10 established the foundation (status/metrics endpoints, static
page). Phase 11 adds before/after raster rendering. This module never
reads demo_data/ file paths directly for status/metadata - it only calls
app.data_adapter functions - but DOES call app.rendering (which reads
pixel data) for the two /api/render/* endpoints, since pixel rendering is
this phase's job (see app/README.md).

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

from app.data_adapter import get_demo_status, load_metrics, load_raster_summary
from app.rendering import render_raster_file_as_png

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR.parent / "config.yaml"

_RENDERABLE_KEYS = ("input", "sr")


def load_config() -> dict:
    config_path = Path(os.environ.get("GEOREFINE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


app = FastAPI(title="GeoRefine Dashboard", version="0.2.0")


@app.get("/api/status")
def api_status():
    config = load_config()
    return get_demo_status(config)


@app.get("/api/metrics")
def api_metrics():
    config = load_config()
    metrics = load_metrics(config)
    if metrics is None:
        return {"available": False, "metrics": None}
    return {"available": True, "metrics": metrics}


@app.get("/api/raster-info")
def api_raster_info():
    config = load_config()
    return {key: load_raster_summary(config, key) for key in _RENDERABLE_KEYS}


@app.get("/api/render/{key}")
def api_render(key: str):
    if key not in _RENDERABLE_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown render key '{key}'. Expected one of {_RENDERABLE_KEYS}")

    config = load_config()
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

    viz_cfg = config.get("visualization", {})
    low = viz_cfg.get("stretch_low_percentile", 2.0)
    high = viz_cfg.get("stretch_high_percentile", 98.0)

    try:
        png_bytes = render_raster_file_as_png(raster_path, low, high)
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