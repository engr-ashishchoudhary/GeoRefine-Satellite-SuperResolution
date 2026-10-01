async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function main() {
  const appEl = document.getElementById("app");

  await renderModeBadge();

  let status;
  try {
    status = await fetchJSON("/api/status");
  } catch (err) {
    appEl.innerHTML = `<div class="panel"><p class="empty-state">Could not reach the GeoRefine API: ${err}</p></div>`;
    return;
  }

  const sections = [await renderModeBanner(), renderStatusPanel(status), renderContractPanel(status)];

  const hasBeforeAfter = status.files.input && status.files.input.exists && status.files.sr && status.files.sr.exists;
  if (hasBeforeAfter) {
    sections.push(await renderBeforeAfterPanel());
  }

  sections.push(await renderMetricsPanel());

  const hasNdvi =
    status.files.original_ndvi && status.files.original_ndvi.exists && status.files.sr_ndvi && status.files.sr_ndvi.exists;
  if (hasNdvi) {
    sections.push(await renderNdviPanel());
  }

  const hasUncertainty = status.files.uncertainty && status.files.uncertainty.exists;
  if (hasUncertainty) {
    sections.push(await renderUncertaintyPanel());
  }

  sections.push(renderLiveInferencePanel());

  appEl.innerHTML = sections.join("");

  if (hasBeforeAfter) {
    wireCompareSlider();
  }
  wireLiveInferenceForm();
}

async function renderModeBadge() {
  const badge = document.getElementById("modeBadge");
  try {
    const data = await fetchJSON("/api/mode");
    badge.textContent = data.mode === "demo" ? "DEMO MODE" : data.mode.toUpperCase() + " MODE";
    badge.classList.remove("mode-badge-hidden");
  } catch (err) {
    // If /api/mode is unreachable, leave the badge hidden rather than
    // showing a wrong or misleading mode label.
  }
}

async function renderModeBanner() {
  try {
    const data = await fetchJSON("/api/mode");
    return `<div class="mode-banner">${data.description}</div>`;
  } catch (err) {
    return "";
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

function formatNdviStat(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(3);
}

function ndviStatsBlock(label, stats) {
  if (!stats) return `<div class="ndvi-stats"><strong>${label}</strong><p class="empty-state">No stats available.</p></div>`;
  return `<div class="ndvi-stats">
    <strong>${label}</strong>
    <div class="ndvi-stat-row"><span>Mean</span><span>${formatNdviStat(stats.mean)}</span></div>
    <div class="ndvi-stat-row"><span>Min</span><span>${formatNdviStat(stats.min)}</span></div>
    <div class="ndvi-stat-row"><span>Max</span><span>${formatNdviStat(stats.max)}</span></div>
    <div class="ndvi-stat-row"><span>Valid pixels</span><span>${
      stats.valid_pixel_fraction != null ? (stats.valid_pixel_fraction * 100).toFixed(1) + "%" : "—"
    }</span></div>
  </div>`;
}

async function renderNdviPanel() {
  let info = { original_ndvi: null, sr_ndvi: null, report: null };
  try {
    info = await fetchJSON("/api/ndvi-info");
  } catch (err) {
    // fall through - panel still renders images even without metadata/report
  }

  const report = info.report;
  const originalStats = report ? report.original_ndvi && report.original_ndvi.stats : null;
  const srStats = report ? report.sr_ndvi && report.sr_ndvi.stats : null;
  const noteLine = report && report.note ? `<p class="metrics-note">${report.note}</p>` : "";

  return `
    <div class="panel">
      <h2>NDVI</h2>
      <p class="compare-meta">
        Original (LR-resolution) and SR-derived NDVI are shown side by side, not
        pixel-differenced - they are at different resolutions and are not
        pixel-aligned. Colormap: brown = low/no vegetation, green = healthy vegetation,
        transparent = undefined (NDVI denominator was zero).
      </p>
      <div class="ndvi-grid">
        <div class="ndvi-column">
          <div class="ndvi-image-wrap"><img src="/api/render-ndvi/original_ndvi" alt="Original NDVI" /></div>
          <p class="ndvi-caption">Original (from real LR imagery)</p>
          ${ndviStatsBlock("Original NDVI", originalStats)}
        </div>
        <div class="ndvi-column">
          <div class="ndvi-image-wrap"><img src="/api/render-ndvi/sr_ndvi" alt="SR-derived NDVI" /></div>
          <p class="ndvi-caption">SR-derived (from reconstructed imagery)</p>
          ${ndviStatsBlock("SR NDVI", srStats)}
        </div>
      </div>
      ${noteLine}
    </div>
  `;
}

const UNCERTAINTY_BAND_LABELS = ["Red", "NIR"];

function uncertaintyBandRows(perBandStats) {
  if (!perBandStats || perBandStats.length === 0) {
    return `<p class="empty-state">No per-band stats available.</p>`;
  }
  return perBandStats
    .map((s) => {
      const label = UNCERTAINTY_BAND_LABELS[s.band_index] || `Band ${s.band_index}`;
      return `<div class="ndvi-stat-row">
        <span>${label} — mean / max uncertainty</span>
        <span>${Number(s.mean_uncertainty).toFixed(4)} / ${Number(s.max_uncertainty).toFixed(4)}</span>
      </div>`;
    })
    .join("");
}

async function renderUncertaintyPanel() {
  let info = { uncertainty: null, report: null };
  try {
    info = await fetchJSON("/api/uncertainty-info");
  } catch (err) {
    // fall through - panel still renders the image even without a report
  }

  const report = info.report;
  const augmentationsLine = report
    ? `<p class="compare-meta">TTA ensemble: ${report.num_augmentations} augmentations (${report.augmentations_used.join(", ")}).</p>`
    : "";
  const noteLine = report && report.note ? `<p class="metrics-note">${report.note}</p>` : "";
  const bandRows = report ? uncertaintyBandRows(report.per_band_stats) : "";

  return `
    <div class="panel">
      <h2>Uncertainty</h2>
      <p class="compare-meta">
        Heatmap shows per-pixel disagreement across a flip-based test-time-augmentation
        ensemble, averaged across bands for display. This is NOT a calibrated confidence
        interval - see the note below.
      </p>
      ${augmentationsLine}
      <div class="uncertainty-image-wrap"><img src="/api/render-uncertainty" alt="Uncertainty heatmap" /></div>
      <div class="uncertainty-legend">
        <span>Low uncertainty</span>
        <div class="uncertainty-gradient-bar"></div>
        <span>High uncertainty</span>
      </div>
      <div class="ndvi-stats">${bandRows}</div>
      ${noteLine}
    </div>
  `;
}

function renderLiveInferencePanel() {
  return `
    <div class="panel">
      <h2>Live Inference <span class="mode-badge mode-badge-live">LIVE</span></h2>
      <p class="compare-meta">
        Upload your own GeoTIFF with at least 2 bands (band 1 = red, band 2 = nir) to run
        the real pretrained model on it now. This is a genuine inference run performed in
        this session, not precomputed demo data.
      </p>
      <form id="liveUploadForm" class="live-upload-form">
        <input type="file" id="liveFileInput" accept=".tif,.tiff" required />
        <button type="submit" id="liveSubmitBtn">Run Inference</button>
      </form>
      <div id="liveResultContainer"></div>
    </div>
  `;
}

function wireLiveInferenceForm() {
  const form = document.getElementById("liveUploadForm");
  if (!form) return;

  const resultContainer = document.getElementById("liveResultContainer");
  const submitBtn = document.getElementById("liveSubmitBtn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("liveFileInput");
    if (!fileInput.files.length) return;

    submitBtn.disabled = true;
    submitBtn.textContent = "Running inference...";
    resultContainer.innerHTML = `<p class="empty-state">Running the pretrained model on your upload - this may take a moment...</p>`;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const res = await fetch("/api/live/infer", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        resultContainer.innerHTML = `<p class="empty-state">Error: ${err.detail || "inference failed"}</p>`;
        return;
      }
      const result = await res.json();
      const cacheBust = Date.now();
      resultContainer.innerHTML = `
        <p class="compare-meta">
          File: ${result.original_filename} — LR shape: ${result.scene_shape.join("×")},
          SR shape: ${result.sr_shape.join("×")}
        </p>
        <div class="compare-wrap" id="liveCompareWrap">
          <img src="/api/live/render/sr?t=${cacheBust}" alt="Live SR output" />
          <div class="compare-before-layer" id="liveBeforeLayer">
            <img id="liveBeforeImg" src="/api/live/render/input?t=${cacheBust}" alt="Live LR input" />
          </div>
          <div class="compare-divider" id="liveDivider"></div>
        </div>
        <input type="range" class="compare-slider" id="liveSlider" min="0" max="100" value="50" />
        <div class="compare-labels"><span>Before (your upload)</span><span>After (live SR)</span></div>
      `;
      wireLiveCompareSlider();
    } catch (err) {
      resultContainer.innerHTML = `<p class="empty-state">Request failed: ${err}</p>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Run Inference";
    }
  });
}

function wireLiveCompareSlider() {
  const slider = document.getElementById("liveSlider");
  const beforeLayer = document.getElementById("liveBeforeLayer");
  const divider = document.getElementById("liveDivider");
  const wrap = document.getElementById("liveCompareWrap");
  const beforeImg = document.getElementById("liveBeforeImg");

  function setWidth() {
    beforeImg.style.width = `${wrap.clientWidth}px`;
  }
  setWidth();
  window.addEventListener("resize", setWidth);

  slider.addEventListener("input", () => {
    beforeLayer.style.width = `${slider.value}%`;
    divider.style.left = `${slider.value}%`;
  });
}

main();