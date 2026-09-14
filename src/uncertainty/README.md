\# src/uncertainty — Phase 7: Uncertainty Estimation



Estimates per-band, per-pixel uncertainty for the demo scene's SR output

using a flip-based test-time-augmentation (TTA) ensemble.



\## Method: TTA ensemble variance



\*\*Why not Monte Carlo Dropout:\*\* the pretrained RRDBNet architecture

(`src/model/rrdbnet.py`) has no dropout layers, and adding them would

require retraining — out of scope per the project's model-strategy rules

(prefer pretrained, do not over-invest in architecture changes).



\*\*What this does instead:\*\* runs super-resolution on 4 geometrically

augmented copies of the scene's real LR imagery — identity, horizontal

flip, vertical flip, and both (`src/uncertainty/tta.py`) — undoes each

augmentation on its SR output so every prediction is back in the original

orientation, then computes the per-band, per-pixel \*\*standard deviation\*\*

across those 4 predictions. That standard deviation map is the uncertainty

output.



90-degree rotations are deliberately excluded: they swap height and width,

which would complicate `src.inference.scene\_inference.run\_tiled\_inference()`'s

tiling math for non-square scenes without adding much beyond what flips

already capture.



\## What this uncertainty means (and doesn't)



Disagreement across predictions from geometrically-equivalent input views

is a legitimate, technically defensible signal that the model's

reconstruction is less stable in a region — \*\*it is not a calibrated

statistical confidence interval\*\*. Per the project's top-level scientific

honesty notice:



> High-uncertainty areas represent regions where the model's reconstruction

> is less stable and should be interpreted cautiously.



Uncertainty does \*\*not\*\* prove a generated object is real, and does not

prove it is fake. It flags where the SR output is less trustworthy, not

what the ground truth actually is.



\## Output



\- \*\*`demo\_data/uncertainty/scene\_uncertainty.tif`\*\* — same band count,

&#x20; order, CRS, and grid as the SR output (`demo\_data/sr/scene\_sr.tif`), so

&#x20; the two can be directly overlaid. This module does \*\*not\*\* overwrite

&#x20; `scene\_sr.tif` — Phase 5/6 own that file; the ensemble's identity-flip

&#x20; prediction is mathematically equivalent to running

&#x20; `run\_scene\_inference()` directly, but is not written out separately here

&#x20; to avoid two phases claiming ownership of the same file.

\- \*\*`data/processed/uncertainty\_report.json`\*\* — audit trail: which

&#x20; augmentations were used, per-band mean/max uncertainty and ensemble

&#x20; mean, shape, and the same cautious-interpretation note.



\## Config (`config.yaml`, additive)



```yaml

uncertainty:

&#x20; augmentations: \["identity", "hflip", "vflip", "hvflip"]

&#x20; output\_report\_filename: "uncertainty\_report.json"

```



`augmentations` must list at least 2 entries (there must be more than one

prediction to measure disagreement across) and only unrecognized names

raise an error — see `tta.py`'s `AUGMENTATION\_NAMES`.



\## CLI

python scripts/run\_uncertainty.py



Requires `data/processed/dataset\_index.json` (Phase 1) and

`models/RealESRGAN\_x4plus.pth` (Phase 4).



\## Dependencies



None added — this module only uses `numpy` and existing project code

(`src.inference.scene\_inference.run\_tiled\_inference`, introduced in

Phase 6's refactor).



\## Known limitations



\- TTA-ensemble variance is an approximate, practical uncertainty proxy —

&#x20; not a rigorously calibrated one (e.g. not validated against known-error

&#x20; regions with real ground truth, since none exists yet in this project).

\- Only 4 augmentations are used; a larger ensemble (e.g. adding 90-degree

&#x20; rotations with corrected tiling logic, or multi-scale crops) would give

&#x20; a smoother estimate at proportionally higher compute cost — deferred

&#x20; since the current cost is already 4x a single inference pass.

\- Only exercised against synthetic test GeoTIFFs so far — no real

&#x20; Sentinel-2 imagery has been run through this pipeline yet, same caveat

&#x20; as every prior phase.



