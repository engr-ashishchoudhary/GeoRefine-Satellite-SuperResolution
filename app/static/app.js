async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function main() {
  const appEl = document.getElementById("app");
  let status;
  try {
    status = await fetchJSON("/api/status");
  } catch (err) {
    appEl.innerHTML = `<div class="panel"><p class="empty-state">Could not reach the GeoRefine API: ${err}</p></div>`;
    return;
  }

  const sections = [renderStatusPanel(status), renderContractPanel(status)];

  if (status.files.input && status.files.input.exists && status.files.sr && status.files.sr.exists) {
    sections.push(await renderBeforeAfterPanel());
  }

  sections.push(await renderMetricsPanel());

  appEl.innerHTML = sections.join("");

  if (status.files.input && status.files.input.exists && status.files.sr && status.files.sr.exists) {
    wireCompareSlider();
  }
}

function renderStatusPanel(status) {
  if (status.ready) {
    return `<div class="panel"><h2>Status</h2><p>Demo outputs are present.</p></div>`;
  }
  return `<div class="panel"><h2>Status</h2><div class="empty-state">
      <p>Demo data has not been fully generated yet.</p>
      <p>Run the full pipeline locally:</p>
      <p><code>python scripts/run_full_pipeline.py</code></p>
    </div></div>`;
}

function renderContractPanel(status) {
  const rows = Object.entries(status.files)
    .map(([key, info]) => {
      const cls = info.exists ? "ok" : "missing";
      return `<div class="file-row">
        <span class="dot ${cls}"></span>
        <span>${key}</span>
        <span class="file-path">${info.path}</span>
      </div>`;
    })
    .join("");
  return `<div class="panel"><h2>Demo Output Contract</h2>${rows}</div>`;
}

async function renderBeforeAfterPanel() {
  let info = { input: null, sr: null };
  try {
    info = await fetchJSON("/api/raster-info");
  } catch (err) {
    // fall through with nulls - the panel still renders images even without metadata
  }

  let metaLine = "";
  if (info.input && info.sr) {
    const scaleX = (info.sr.width / info.input.width).toFixed(0);
    metaLine = `LR: ${info.input.width}×${info.input.height} &rarr; SR: ${info.sr.width}×${info.sr.height} (${scaleX}x). False-color render: R=NIR, G/B=Red.`;
  }

  return `
    <div class="panel">
      <h2>Before / After</h2>
      <p class="compare-meta">${metaLine}</p>
      <div class="compare-wrap" id="compareWrap">
        <img src="/api/render/sr" alt="Super-resolved output" />
        <div class="compare-before-layer" id="compareBeforeLayer">
          <img id="compareBeforeImg" src="/api/render/input" alt="Original low-resolution input" />
        </div>
        <div class="compare-divider" id="compareDivider"></div>
      </div>
      <input type="range" class="compare-slider" id="compareSlider" min="0" max="100" value="50" />
      <div class="compare-labels"><span>Before (LR input)</span><span>After (SR output)</span></div>
    </div>
  `;
}

function wireCompareSlider() {
  const slider = document.getElementById("compareSlider");
  const beforeLayer = document.getElementById("compareBeforeLayer");
  const divider = document.getElementById("compareDivider");
  const wrap = document.getElementById("compareWrap");
  const beforeImg = document.getElementById("compareBeforeImg");

  function setBeforeImgWidth() {
    beforeImg.style.width = `${wrap.clientWidth}px`;
  }
  setBeforeImgWidth();
  window.addEventListener("resize", setBeforeImgWidth);

  slider.addEventListener("input", () => {
    const pct = slider.value;
    beforeLayer.style.width = `${pct}%`;
    divider.style.left = `${pct}%`;
  });
}

const METRIC_DEFS = [
  { key: "psnr", label: "PSNR", unit: "dB", higherIsBetter: true },
  { key: "ssim", label: "SSIM", unit: "", higherIsBetter: true },
  { key: "rmse", label: "RMSE", unit: "", higherIsBetter: false },
  { key: "sam", label: "SAM", unit: "°", higherIsBetter: false },
];

function formatMetricValue(value) {
  if (value === null || value === undefined) return "—";
  if (value === Infinity || value === "Infinity") return "∞";
  if (typeof value === "number") return value.toFixed(3);
  return String(value);
}

async function renderMetricsPanel() {
  let data;
  try {
    data = await fetchJSON("/api/metrics");
  } catch (err) {
    return `<div class="panel"><h2>Validation Metrics</h2><p class="empty-state">Could not load metrics: ${err}</p></div>`;
  }

  if (!data.available) {
    return `<div class="panel"><h2>Validation Metrics</h2><div class="empty-state">
        <p>No metrics have been computed yet.</p>
        <p>Run: <code>python scripts/run_validation.py</code></p>
      </div></div>`;
  }

  const m = data.metrics;

  if (data.is_placeholder) {
    return `<div class="panel">
      <h2>Validation Metrics</h2>
      <div class="placeholder-banner">
        These are Phase 0 placeholder values, not yet computed from real data.
        Run <code>python scripts/run_validation.py</code> to generate a real result.
      </div>
    </div>`;
  }

  const cards = METRIC_DEFS.map(
    (def) => `
      <div class="metric-card">
        <div class="metric-label">${def.label}</div>
        <div class="metric-value">${formatMetricValue(m[def.key])}<span class="metric-unit">${def.unit}</span></div>
      </div>`
  ).join("");

  const comparisonLine = m.comparison_type
    ? `<p class="metrics-comparison-type">Comparison type: <strong>${m.comparison_type}</strong></p>`
    : "";
  const noteLine = m.note ? `<p class="metrics-note">${m.note}</p>` : "";

  return `<div class="panel">
      <h2>Validation Metrics</h2>
      ${comparisonLine}
      <div class="metrics-grid">${cards}</div>
      ${noteLine}
    </div>`;
}

main();