"""
CLI entry point: run Phase 6 validation on the demo scene, writing
demo_data/metrics/metrics.json and data/processed/validation_report.json.

Usage (from repo root):
    python scripts/run_validation.py
    python scripts/run_validation.py --config config.yaml

Requires:
  - data/processed/preprocessing_index.json (Phase 2)
  - models/RealESRGAN_x4plus.pth (see scripts/download_pretrained_model.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/run_validation.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.super_resolution import load_model
from src.validation.pipeline import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRefine Phase 6 validation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["paths"]["data_processed"])
    dataset_index_filename = config.get("dataset", {}).get("index_filename", "dataset_index.json")
    dataset_index_path = processed_dir / dataset_index_filename

    preprocessing_index_filename = config.get("preprocessing", {}).get(
        "output_index_filename", "preprocessing_index.json"
    )
    preprocessing_index_path = processed_dir / preprocessing_index_filename

    if not dataset_index_path.exists():
        print(
            f"dataset_index.json not found at {dataset_index_path}.\n"
            "Run `python scripts/build_dataset_index.py` first (Phase 1)."
        )
        sys.exit(1)

    if not preprocessing_index_path.exists():
        print(
            f"preprocessing_index.json not found at {preprocessing_index_path}.\n"
            "Run `python scripts/run_preprocessing.py` first (Phase 2)."
        )
        sys.exit(1)

    print("Loading pretrained model...")
    try:
        model = load_model(models_dir=config["paths"]["models"])
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    print("Running validation...")
    try:
        report = run_validation(dataset_index_path, preprocessing_index_path, config, model)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Scene validated:     {report['scene_id']}")
    print(f"Comparison type:     {report['comparison_type']}")
    print(f"PSNR:                {report['metrics']['psnr']}")
    print(f"SSIM:                {report['metrics']['ssim']}")
    print(f"RMSE:                {report['metrics']['rmse']}")
    print(f"SAM (degrees):       {report['metrics']['sam']}")
    print(f"Valid pixel fraction:{report['valid_pixel_fraction']:.1%}")
    print(f"Report written to:   {report['report_path']}")
    print(f"Metrics written to:  {report['metrics_path']}")


if __name__ == "__main__":
    main()