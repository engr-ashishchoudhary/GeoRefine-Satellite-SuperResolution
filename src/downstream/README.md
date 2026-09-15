\# src/downstream — Phase 8: NDVI \& Downstream Analysis



Computes NDVI (Normalized Difference Vegetation Index) for the demo

scene's original (real LR) imagery and its SR output, from actual raster

calculations.

NDVI = (NIR - Red) / (NIR + Red)



\## What this produces



\- \*\*`original\_ndvi`\*\* — computed from the scene's real LR imagery, at LR

&#x20; resolution.

\- \*\*`sr\_ndvi`\*\* — computed from the SR output (refreshed via

&#x20; `run\_scene\_inference()` so it is never stale from a prior run), at SR

&#x20; resolution (LR resolution × `model.scale`).



Both are genuinely calculated from real pixel values — nothing here is

fabricated or interpolated as a placeholder. Per the project's scientific

honesty notice: \*\*the SR-derived NDVI is based on reconstructed,

super-resolved imagery, not directly observed satellite pixels\*\*, and

every output (`ndvi\_report.json`'s `note`) says so explicitly.



\## NDVI edge case: zero denominator



Where `nir + red == 0`, NDVI is mathematically undefined (0/0). This

module writes `NaN` for those pixels, not `0` — reporting `0` would imply

"no vegetation," which is a different, fabricated claim. GeoTIFFs are

written with `nodata=NaN` accordingly.



\## Known limitations



\- \*\*No direct NDVI difference map is produced in this phase.\*\*

&#x20; `original\_ndvi` (LR resolution) and `sr\_ndvi` (LR resolution ×

&#x20; `model.scale`) are at different pixel grids — a genuine pixel-wise diff

&#x20; would require the same resampling machinery

&#x20; `src.validation.pipeline.resample\_to\_reference()` provides. Reusing that

&#x20; here would create a `src.downstream` → `src.validation` dependency for

&#x20; what is really a dashboard visualization concern. Phase 13 (NDVI

&#x20; Visualization) owns reconciling the two rasters for side-by-side or

&#x20; differenced display. Both rasters remain independently valid, real NDVI

&#x20; and are suitable for visual/statistical comparison as-is.

\- \*\*Requires a known band order.\*\* NDVI needs to know which array index is

&#x20; red vs. nir. This works for the common case (matched single-band LR

&#x20; files) but raises a clear error for a pre-merged multi-band composite

&#x20; file, whose band order `src.preprocessing.bands.load\_band\_stack()`

&#x20; cannot verify — same restriction documented in

&#x20; `src/preprocessing/README.md`, not a new one introduced here.

\- Only exercised against synthetic test GeoTIFFs so far — no real

&#x20; Sentinel-2 imagery has been run through this pipeline yet, same caveat

&#x20; as every prior phase.



\## Output



\- \*\*`demo\_data/ndvi/original\_ndvi.tif`\*\*, \*\*`demo\_data/ndvi/sr\_ndvi.tif`\*\*

&#x20; — single-band float32 GeoTIFFs, `nodata=NaN`.

\- \*\*`data/processed/ndvi\_report.json`\*\* — per-raster mean/min/max,

&#x20; valid-pixel fraction, shape, and the note above.



\## Config (`config.yaml`, additive)



```yaml

downstream:

&#x20; output\_report\_filename: "ndvi\_report.json"

```



\## CLI

python scripts/run\_ndvi.py



Requires `data/processed/dataset\_index.json` (Phase 1) and

`models/RealESRGAN\_x4plus.pth` (Phase 4).



\## Dependencies



None added.



