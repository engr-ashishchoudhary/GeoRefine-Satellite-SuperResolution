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

