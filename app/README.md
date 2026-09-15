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
