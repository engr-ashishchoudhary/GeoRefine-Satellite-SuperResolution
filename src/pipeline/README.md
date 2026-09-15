\# src/pipeline — Phase 9: Standardized Demo Outputs



Runs Phases 5 through 8 for one scene in sequence, then generates and

validates the project's standard `demo\_data/` output contract (established

in Phase 0).



This module adds no new inference or analysis logic — it only orchestrates

each existing phase's already-tested entry point:



Phase 5: run\_demo\_output() -> demo\_data/input/, demo\_data/sr/

Phase 6: run\_validation() -> demo\_data/metrics/, refreshes sr/

Phase 7: run\_uncertainty\_estimation() -> demo\_data/uncertainty/

Phase 8: run\_ndvi\_analysis() -> demo\_data/ndvi/, refreshes sr/





Phase 5 must run first — it is the only one that writes

`demo\_data/input/scene.tif`.



\## Manifest generation



Phase 0 established `demo\_data/manifest.json` as a hand-written static

file, matching `config.yaml`'s `demo\_data\_contract` section. This module

\*\*generates\*\* that file from `config.yaml` instead — identical keys and

values, so this is not a redesign of the contract (see section 20 of the

project's master instructions: "do not repeatedly redesign this

interface"). It removes the risk of the hand-written file and

`config.yaml` silently drifting apart, since there is now exactly one

source of truth.



\## Contract validation



After all four phases run, `validate\_demo\_outputs()` checks that every

file the manifest points to:



\- \*\*Exists\*\* on disk.

\- \*\*Opens successfully\*\* — raster entries (`scene`, `sr`, `uncertainty`,

&#x20; `original\_ndvi`, `sr\_ndvi`) via `rasterio`, with valid width/height/band

&#x20; count; the `metrics` entry as valid JSON containing the required

&#x20; `psnr`/`ssim`/`rmse`/`sam` keys.



This does \*\*not\*\* check that metric values are non-placeholder — a freshly

computed `0.0` is a legitimate real value indistinguishable from a

placeholder by value alone. It only confirms the contract's \*structure\* is

satisfied, catching a broken or partial pipeline run before it's reported

as "done."



If any file is missing or fails to open, `run\_full\_demo\_pipeline()` raises

`ValueError` listing every issue found — it never returns a success report

for a partially-broken run.



\## Config (`config.yaml`, additive)



```yaml

pipeline:

&#x20; manifest\_filename: "manifest.json"

```



\## CLI

python scripts/run\_full\_pipeline.py



Requires `data/processed/dataset\_index.json` (Phase 1),

`data/processed/preprocessing\_index.json` (Phase 2), and

`models/RealESRGAN\_x4plus.pth` (Phase 4).



\## Dependencies



None added.



\## Known limitations



\- Only orchestrates the single demo scene selected by

&#x20; `config.yaml`'s `inference.scene\_id` (or the first scene with LR files)

&#x20; — batch-processing multiple scenes through this pipeline is not

&#x20; implemented and is out of scope for the demo output contract.

\- Only exercised against synthetic test GeoTIFFs so far — no real

&#x20; Sentinel-2 imagery has been run through this pipeline yet, same caveat

&#x20; as every prior phase.

