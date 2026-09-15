"""
Phase 10: FastAPI dashboard foundation.

Intentionally minimal - proves the server runs, serves a status page, and
correctly reports demo_data/ availability via app.data_adapter. No
visualizations are built here; those are Phases 11-14. This module never
reads demo_data/ file paths directly - it only calls app.data_adapter
functions, keeping the dashboard decoupled from the pipeline's internal
file layout (see app/README.md).

Config path is overridable via the GEOREFINE_CONFIG environment variable
so tests can point at an isolated temporary config without touching the
real project config.yaml.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_adapter import get_demo_status, load_metrics

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR.parent / "config.yaml"


def load_config() -> dict:
    config_path = Path(os.environ.get("GEOREFINE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


app = FastAPI(title="GeoRefine Dashboard", version="0.1.0")


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