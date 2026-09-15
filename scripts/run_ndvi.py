"""
CLI entry point: run Phase 8 NDVI analysis on the demo scene, writing
demo_data/ndvi/original_ndvi.tif, demo_data/ndvi/sr_ndvi.tif, and
data/processed/ndvi_report.json.

Usage (from repo root):
    python scripts/run_ndvi.py
    python scripts/run_ndvi.py --config config.yaml

Requires:
  - data/processed/dataset_index.json (Phase 1)
  - models/RealESRGAN_x4plus.pth (see scripts/download_pretrained_model.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/run_ndvi.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.downstream.pipeline import run_ndvi_analysis
from src.model.super_resolution import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRefine Phase 8 NDVI analysis.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["paths"]["data_processed"])
    dataset_index_filename = config.get("dataset", {}).get("index_filename", "dataset_index.json")
    dataset_index_path = processed_dir / dataset_index_filename

    if not dataset_index_path.exists():
        print(
            f"dataset_index.json not found at {dataset_index_path}.\n"
            "Run `python scripts/build_dataset_index.py` first (Phase 1)."
        )
        sys.exit(1)

    print("Loading pretrained model...")
    try:
        model = load_model(models_dir=config["paths"]["models"])
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    print("Running NDVI analysis...")
    try:
        report = run_ndvi_analysis(dataset_index_path, config, model)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Scene:              {report['scene_id']}")
    print(f"Original NDVI file: {report['original_ndvi']['file']}")
    print(f"  mean/min/max:     {report['original_ndvi']['stats']['mean']}, "
          f"{report['original_ndvi']['stats']['min']}, {report['original_ndvi']['stats']['max']}")
    print(f"SR NDVI file:       {report['sr_ndvi']['file']}")
    print(f"  mean/min/max:     {report['sr_ndvi']['stats']['mean']}, "
          f"{report['sr_ndvi']['stats']['min']}, {report['sr_ndvi']['stats']['max']}")
    print(f"Report written to:  {report['report_path']}")


if __name__ == "__main__":
    main()
