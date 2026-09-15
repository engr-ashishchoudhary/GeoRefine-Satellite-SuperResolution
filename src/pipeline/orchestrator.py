"""
Phase 9 orchestration: run Phases 5 through 8 for one scene, then generate
and validate the standard demo output contract.

This module introduces no new inference/analysis logic of its own - it
only calls each existing phase's already-tested entry point in sequence:

    Phase 5: run_demo_output()          -> demo_data/input/, demo_data/sr/
    Phase 6: run_validation()           -> demo_data/metrics/, refreshes sr/
    Phase 7: run_uncertainty_estimation() -> demo_data/uncertainty/
    Phase 8: run_ndvi_analysis()        -> demo_data/ndvi/, refreshes sr/

Phase 5 must run first: it is the only one of the four that writes
demo_data/input/scene.tif (the others only refresh demo_data/sr/scene_sr.tif
or write their own phase-specific outputs). The final demo_data/sr/scene_sr.tif
reflects whichever phase last refreshed it (Phase 6 for a real-HR scene, or
Phase 8 always) - both are SR computed from the scene's real LR imagery via
run_scene_inference(), so the file's content is equivalent regardless of
which phase most recently wrote it.

After all four phases have run, this module builds manifest.json from
config.yaml's demo_data_contract (see manifest.py) and validates that
every contract file actually exists and is openable - raising ValueError
with the full list of issues if anything is missing or corrupt, rather
than reporting a false "success".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.downstream.pipeline import run_ndvi_analysis
from src.inference.scene_inference import run_demo_output
from src.uncertainty.pipeline import run_uncertainty_estimation
from src.validation.pipeline import run_validation

from .manifest import validate_demo_outputs, write_manifest


def run_full_demo_pipeline(
    dataset_index_path: Path,
    preprocessing_index_path: Path,
    config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Run the full demo pipeline (Phases 5-8) for the demo scene, then
    write and validate manifest.json.

    Returns a report dict summarizing every sub-phase's result. Raises
    ValueError (propagated from the underlying phase, or raised here for
    contract validation failures) if any step cannot complete - this
    function never returns a "success" report for a partially-broken run.
    """
    demo_output_result = run_demo_output(dataset_index_path, config, model)
    validation_report = run_validation(dataset_index_path, preprocessing_index_path, config, model)
    uncertainty_report = run_uncertainty_estimation(dataset_index_path, config, model)
    ndvi_report = run_ndvi_analysis(dataset_index_path, config, model)

    demo_dir = Path(config["paths"]["demo_data"])
    manifest_filename = config.get("pipeline", {}).get("manifest_filename", "manifest.json")
    manifest_path = demo_dir / manifest_filename
    manifest = write_manifest(config, manifest_path)

    issues = validate_demo_outputs(config)
    if issues:
        raise ValueError(
            "Demo output contract validation failed after running the full pipeline:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
        )

    return {
        "scene_id": demo_output_result["scene_id"],
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "validation": validation_report,
        "uncertainty": uncertainty_report,
        "ndvi": ndvi_report,
        "issues": issues,
    }