async function loadStatus() {
  const appEl = document.getElementById("app");
  try {
    const res = await fetch("/api/status");
    const status = await res.json();
    renderStatus(status);
  } catch (err) {
    appEl.innerHTML = `<div class="panel"><p class="empty-state">Could not reach the GeoRefine API: ${err}</p></div>`;
  }
}

function renderStatus(status) {
  const appEl = document.getElementById("app");
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

  const readyBanner = status.ready
    ? `<div class="panel"><h2>Status</h2><p>Demo outputs are present. Visualizations will appear here in later phases.</p></div>`
    : `<div class="panel"><h2>Status</h2><div class="empty-state">
        <p>Demo data has not been generated yet.</p>
        <p>Run the full pipeline locally:</p>
        <p><code>python scripts/run_full_pipeline.py</code></p>
      </div></div>`;

  appEl.innerHTML = `
    ${readyBanner}
    <div class="panel">
      <h2>Demo Output Contract</h2>
      ${rows}
    </div>
  `;
}

loadStatus();