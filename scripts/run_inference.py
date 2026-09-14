"""
CLI entry point: run full-scene super-resolution inference and write the
project's stable demo_data/ output contract (demo_data/input/scene.tif,
demo_data/sr/scene_sr.tif).
Usage (from repo root):
    python scripts/run_inference.py
    python scripts/run_inference.py --config config.yaml
Requires:
  - data/processed/dataset_index.json (Phase 1)
  - models/RealESRGAN_x4plus.pth (see scripts/download_pretrained_model.py)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.scene_inference import run_demo_output
from src.model.super_resolution import load_model
def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRefine Phase 5 full-scene inference.")
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
    print("Running full-scene inference...")
    try:
        result = run_demo_output(dataset_index_path, config, model)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)
    print(f"Scene used:  {result['scene_id']}")
    print(f"Input written to: {result['input']}")
    print(f"SR written to:    {result['sr']}")
if __name__ == "__main__":
    main()
