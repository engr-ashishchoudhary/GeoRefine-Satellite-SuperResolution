\# app — Phase 10: Dashboard Foundation \& Data Adapter



A minimal FastAPI server proving the dashboard can run and correctly

report `demo\_data/` availability. \*\*No visualizations are built here\*\* —

that starts in Phase 11 (Before/After) through Phase 14 (Uncertainty

Visualization).



\## Dashboard independence



Per the project's dashboard independence requirement: `app/data\_adapter.py`

is the \*\*only\*\* module in `app/` that knows `demo\_data/`'s file layout or

`config.yaml`'s paths. `app/app.py` and the frontend JS only ever call

`data\_adapter` functions or fetch `/api/\*` JSON — they never construct a

`demo\_data/` path themselves. This keeps the dashboard decoupled from the

pipeline's internals (model architecture, preprocessing, uncertainty

method) — it only consumes the standardized output contract Phase 9

established.



\## Data adapter (`data\_adapter.py`)



\- `get\_demo\_status(config)` — reports which contract files exist right now

&#x20; (using `config.yaml`'s `demo\_data\_contract` directly, so it works even

&#x20; before `manifest.json` has ever been generated).

\- `load\_metrics(config)` — returns `metrics.json`'s contents, or `None`.

\- `load\_raster\_summary(config, key)` — lightweight metadata (width,

&#x20; height, band count, CRS) for a raster contract entry, without loading

&#x20; pixel data.

\- `load\_manifest(config)` — the raw manifest dict, or `None`.



Every function returns `None`/a clearly-marked "not ready" state rather

than raising when `demo\_data/` hasn't been populated yet — since as of

this phase, no real pipeline run has been done, this is the expected

default state, not an error condition.



\## API



\- `GET /api/status` — `{"manifest\_available": bool, "files": {...}, "ready": bool}`

\- `GET /api/metrics` — `{"available": bool, "metrics": {...} | null}`

\- `GET /` — the status page (`static/index.html`), which fetches

&#x20; `/api/status` and renders a green/red dot per contract file, or a clear

&#x20; "run the pipeline" message if nothing has been generated yet.



\## Config (`config.yaml`, additive)



```yaml

app:

&#x20; host: "127.0.0.1"

&#x20; port: 8000

```



\## Running locally

python -m uvicorn app.app:app --reload





or



python app/app.py



Then open `http://127.0.0.1:8000/` in a browser.



\## Testing



`app/app.py` reads its config path from the `GEOREFINE\_CONFIG` environment

variable (falling back to the real `config.yaml`), so tests can point it

at an isolated temporary config + `demo\_data/` directory without touching

the real project files. See `tests/test\_app.py`.



\## Dependencies added



\- `fastapi` — web framework

\- `uvicorn` — ASGI server

\- `httpx` — required by FastAPI's `TestClient` for tests

\- `aiofiles` — required by `StaticFiles` for serving `static/`



\## Known limitations



\- No visualizations yet — this phase only proves the server + adapter +

&#x20; empty-state handling work correctly.

\- The status page is intentionally plain; visual design work per the

&#x20; project's UX principles (sections 37-39) begins with the actual

&#x20; visualization phases.


## Phase 11: Before/After Visualization

`app/rendering.py` renders raster pixel data as PNG for the dashboard's
before/after comparison slider. This is a new module, separate from
`data_adapter.py`, since pixel rendering is a genuinely different concern
from file-presence/metadata reporting (`data_adapter.py`'s own docstring
says it never loads pixel data).

### Rendering choice

Sentinel-2 imagery here is 2-band (red, nir) — there is no green band, so
no true-color RGB is possible. This module renders a **false-color
composite**: R channel = NIR, G/B channels = Red. This is a documented
visualization convention for display purposes only, not a claim about the
underlying data.

### Band order assumption

Every module in `src/` builds `requested_bands` as `{"red": ..., "nir":
...}` in that order, so band index 0 is always red and index 1 is always
nir in every raster this pipeline writes. `rendering.py` relies on this
same project-wide convention — it does not introduce a new assumption.

### New endpoints

- `GET /api/render/{input|sr}` — returns a PNG. `404` with a clear message
  if the raster doesn't exist yet.
- `GET /api/raster-info` — `{"input": {...} | null, "sr": {...} | null}`,
  width/height/band-count/CRS for each, without loading pixel data
  (reuses `data_adapter.load_raster_summary`).

### Config (`config.yaml`, additive)

```yaml
visualization:
  stretch_low_percentile: 2.0
  stretch_high_percentile: 98.0
```

### Dependencies added

- `Pillow` — PNG encoding. Already installed transitively via
  `torchvision`/`scikit-image`, now declared explicitly since
  `app/rendering.py` imports it directly.

### Known limitations

- The before/after slider displays both images at the same on-screen size
  (browser-scaled) — the LR image is visibly blockier before the slider
  reveals the SR side. This is an honest side-effect of the display
  method, not a rendering artifact to hide.
- Percentile stretch parameters are global (one setting for all scenes),
  not per-scene auto-tuned.


## Phase 12: Metrics Visualization

Adds `data_adapter.describe_metrics()` and a Validation Metrics panel to
the dashboard.

### Placeholder detection

Phase 0's original `demo_data/metrics/metrics.json` is a hand-written
placeholder (`{"psnr": 0.0, ..., "note": "Placeholder values..."}`). Phase
6's real output (`src.validation.pipeline.run_validation()`) always adds
`scene_id` and `comparison_type` keys the placeholder never had.
`is_placeholder_metrics()` checks for the presence of both — this is the
single place that distinguishes "a real computed result" from "the
original placeholder," so the dashboard never presents a placeholder
`0.0` as if it were a genuine measurement.

### `/api/metrics` response shape (changed, additive)

```json
{
  "available": true,
  "is_placeholder": false,
  "metrics": { "psnr": 5.0, "ssim": 0.9, "rmse": 2.0, "sam": 1.0,
               "scene_id": "...", "comparison_type": "...", "note": "..." }
}
```

If `is_placeholder` is `true`, the dashboard shows an explicit amber
banner rather than the metric cards — never silently showing `0.0`s as a
real result.

### Metric display

PSNR (dB), SSIM (unitless), RMSE (unitless, same scale as pixel values),
SAM (degrees) are shown as four cards, plus `comparison_type` and the
scientific `note` from `metrics.json` (e.g. "synthetic_degradation_proxy"
scenes get their cautionary note surfaced directly in the UI, not just in
the JSON).

### Known limitations

- No historical/multi-run comparison — only the single most recent
  `metrics.json` is shown.
- PSNR `Infinity` (a mathematically valid result for an exact-match band,
  see `src/validation/metrics.py`) is displayed as `∞`, not hidden or
  replaced with a large finite number.


## Phase 13: NDVI Visualization

Adds `data_adapter.load_ndvi_report()`, `rendering.render_ndvi_png()`, and
an NDVI panel to the dashboard.

### Colormap, not percentile stretch

NDVI has a defined physical range (-1 to 1), unlike the arbitrary-range
radiance values Phase 11's false-color rendering handles. This module
uses a **fixed** brown → yellow → green colormap over that fixed range —
not a percentile stretch, which would rescale each scene's NDVI
differently and make values incomparable across scenes.

### NaN handling

`src.downstream.ndvi.compute_ndvi()` writes real `NaN` for pixels where
NDVI is undefined (zero denominator). These render **fully transparent**
(RGBA alpha=0) — a checkerboard pattern shows through in the dashboard —
so "no data" is visually distinct from "low NDVI" (which is still a solid
brown), never conflated.

### No diff/overlay between original and SR NDVI

Per Phase 8's documented limitation, `original_ndvi` (LR resolution) and
`sr_ndvi` (LR resolution × `model.scale`) are different resolutions and
are never pixel-aligned. This panel shows them **side by side**, not with
a comparison slider — a slider would misleadingly imply pixel
correspondence that doesn't exist between the two rasters.

### New data source: `data/processed/`

`load_ndvi_report()` is the first `data_adapter` function to read from
`data/processed/` rather than `demo_data/` — still entirely config-driven
(`paths.data_processed`, `downstream.output_report_filename`), so the
"adapter is the only module that knows file layout" rule is preserved.

### New endpoints

- `GET /api/render-ndvi/{original_ndvi|sr_ndvi}` — colormapped PNG. `404`
  with a clear message if the raster doesn't exist yet.
- `GET /api/ndvi-info` — `{"original_ndvi": {...}|null, "sr_ndvi": {...}|null, "report": {...}|null}`.

### Config (`config.yaml`, additive)

```yaml
visualization:
  ndvi_min: -1.0
  ndvi_max: 1.0
```

### Known limitations

- No pixel-wise NDVI difference/change map — same limitation documented in
  `src/downstream/README.md`, not newly introduced by this phase.
- The colormap's exact color stops are a display choice, not a
  standardized index (e.g. not matching any specific published NDVI color
  ramp standard).