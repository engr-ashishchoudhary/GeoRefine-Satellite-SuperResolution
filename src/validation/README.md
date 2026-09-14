\# src/validation — Phase 6: Validation \& Metrics



Computes PSNR, SSIM, RMSE, and SAM for the demo scene selected the same

way as Phase 5 (`config.yaml`'s `inference.scene\_id`, or the first scene

with LR files). Reads `data/processed/preprocessing\_index.json` (Phase 2)

to determine which of two mutually-exclusive comparison paths applies.



\## Two comparison paths



\### 1. Real HR reference (`comparison\_type: "real\_hr\_reference"`)



Used when the scene has an HR file that passed Phase 2's `check\_alignment`

(metadata/bounds-level only — see `src/preprocessing/README.md`'s known

limitations). This module:



1\. Re-runs SR inference on the scene's real LR imagery, refreshing

&#x20;  `demo\_data/sr/scene\_sr.tif` (so the SR raster being validated is never

&#x20;  stale from a previous run).

2\. Resamples the real HR reference onto the \*\*SR output's exact grid\*\*

&#x20;  (`resample\_to\_reference()`) — a pixel-level step Phase 2 never performs.

&#x20;  Destination pixels outside the HR raster's coverage become `NaN` and are

&#x20;  excluded from every metric, not treated as zero error.

3\. Computes all four metrics between SR output and resampled real HR.



This is the only path that measures reconstruction against genuine,

independently-sourced ground truth.



\*\*Documented assumption:\*\* the real HR reference's band count and order

match the SR output's (red, nir). This has not been verified against a

real Sentinel-2 real-HR pair, since none has been added to the project yet

(same caveat as `src/preprocessing/README.md`). A mismatch raises a clear

error rather than comparing misaligned bands.



\### 2. Synthetic degradation proxy (`comparison\_type: "synthetic\_degradation\_proxy"`)



Used when the scene has no real HR and Phase 2's `synthetic\_hr\_fallback`

was used instead. This module:



1\. Runs SR inference directly on `synthetic\_lr.tif` (Phase 2's

&#x20;  deliberately blurred + downsampled raster) via

&#x20;  `src.inference.scene\_inference.run\_tiled\_inference()`.

2\. Reloads the scene's real LR imagery (`proxy\_hr` — the same array

&#x20;  `synthetic\_hr.py` degraded from) and center-crops both arrays to a

&#x20;  common shape (block-mean downsampling can drop a remainder row/column

&#x20;  that exact integer upsampling doesn't recreate).

3\. Computes all four metrics between the two.



\*\*This does not measure real-world HR recovery.\*\* It measures how well the

model reconstructs a \*known, deliberately-applied degradation\* of real

imagery — the same distinction `synthetic\_hr.py` and the project's

top-level scientific honesty notice make. Every output (`metrics.json`,

`validation\_report.json`) carries this label and note explicitly.



If a scene has neither an aligned real HR entry nor a synthetic entry,

`run\_validation()` raises `ValueError` — no placeholder result is written.



\## Metric definitions (`metrics.py`)



\- \*\*RMSE\*\* — root-mean-square error, computed per band, averaged.

\- \*\*PSNR\*\* — computed per band (using that band's own dynamic range unless

&#x20; overridden), averaged. Returns `math.inf` for a band with zero error —

&#x20; documented as mathematically correct, not a bug.

\- \*\*SSIM\*\* — `skimage.metrics.structural\_similarity`, computed \*\*per band\*\*

&#x20; and averaged, not via skimage's multichannel mode. Sentinel-2 red/nir

&#x20; bands are not a natural 3-channel image the way RGB is, so multichannel

&#x20; SSIM's cross-channel weighting doesn't apply here.

\- \*\*SAM\*\* (Spectral Angle Mapper) — per-pixel angle (degrees) between the

&#x20; two images' spectral vectors across bands, averaged over valid pixels.

&#x20; With only 2 bands (red, nir), SAM has much less discriminative power

&#x20; than its typical hyperspectral use case — included because the project

&#x20; spec requires it, not presented as a strong standalone indicator here.



All four functions accept an optional `(H, W)` boolean mask so nodata and

out-of-coverage pixels are excluded rather than silently biasing results.



\## Nodata / invalid-pixel handling



Two mask sources are combined:



1\. \*\*Always applied:\*\* a "finite" mask excluding any pixel that is `NaN`

&#x20;  in either array (this is how `resample\_to\_reference()` marks

&#x20;  out-of-coverage pixels — a correctness requirement, not a policy

&#x20;  choice).

2\. \*\*Controlled by `config.yaml`'s `validation.nodata\_exclude`:\*\* an

&#x20;  additional mask excluding pixels equal to the raster's declared

&#x20;  `nodata` value, when one is set.



\## Outputs



\- \*\*`demo\_data/metrics/metrics.json`\*\* — the stable, minimal contract

&#x20; (Phase 0). Keeps the required `psnr`/`ssim`/`rmse`/`sam` keys (now real

&#x20; computed values, replacing the Phase 0 placeholder `0.0`s), plus

&#x20; `scene\_id`, `comparison\_type`, and `note`.

\- \*\*`data/processed/validation\_report.json`\*\* — the full audit trail:

&#x20; which files were compared, valid-pixel fraction, shapes, and the same

&#x20; `note`. This is additive and does not replace `metrics.json`'s contract.



\## Config (`config.yaml`, additive)



```yaml

validation:

&#x20; output\_report\_filename: "validation\_report.json"

&#x20; nodata\_exclude: true

```



\## CLI

python scripts/run\_validation.py



Requires `data/processed/preprocessing\_index.json` (Phase 2) and

`models/RealESRGAN\_x4plus.pth` (Phase 4) to already exist.



\## Known limitations



\- The real-HR path assumes matching band count/order with the SR output

&#x20; (see above) — untested against real Sentinel-2 real-HR imagery.

\- SAM's utility with only 2 spectral bands is limited; do not treat it as

&#x20; a strong standalone quality indicator until more bands are available.

\- SSIM's per-band-then-averaged approach is a reasonable prototype choice,

&#x20; not a spectral-aware SSIM variant — worth revisiting if literature-grade

&#x20; multispectral SSIM becomes a project requirement later.

\- Only exercised against synthetic test GeoTIFFs so far — no real

&#x20; Sentinel-2 imagery or real HR reference has been run through this

&#x20; pipeline yet, same caveat as every prior phase.

