"""
CLI entry point: run Phase 7 TTA-ensemble uncertainty estimation on the
demo scene, writing demo_data/uncertainty/scene_uncertainty.tif and
data/processed/uncertainty_report.json.

Usage (from repo root):
    python scripts/run_uncertainty.py
    python scripts/run_uncertainty.py --config config.yaml

Requires:
  - data/processed/dataset_index.json (Phase 1)
  - models/RealESRGAN_x4plus.pth (see scripts/download_pretrained_model.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/run_uncertainty.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.super_resolution import load_model
from src.uncertainty.pipeline import run_uncertainty_estimation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRefine Phase 7 uncertainty estimation.")
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

    print("Running TTA-ensemble uncertainty estimation...")
    try:
        report = run_uncertainty_estimation(dataset_index_path, config, model)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Scene:                {report['scene_id']}")
    print(f"Augmentations used:   {report['augmentations_used']}")
    for band_stat in report["per_band_stats"]:
        print(
            f"  Band {band_stat['band_index']}: mean_uncertainty={band_stat['mean_uncertainty']:.4f}, "
            f"max_uncertainty={band_stat['max_uncertainty']:.4f}"
        )
    print(f"Uncertainty written to: {report['uncertainty_file']}")
    print(f"Report written to:      {report['report_path']}")


if __name__ == "__main__":
    main()