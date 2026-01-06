import { fetchResults } from "../api/client.js";

let rootEl = null;
let results = null;

function renderContent() {
  const header = `
    <div class="table-header">
      <span>Risk text</span>
      <span>Predicted level</span>
      <span>Department</span>
      <span>Confidence</span>
    </div>`;

  if (!results || !results.rows || !results.rows.length) {
    return `
      <h2>Results</h2>
      <p>Results will appear here after a classify job runs.</p>
      <div class="table-placeholder">
        ${header}
        <div class="table-row muted">No results yet</div>
      </div>
    `;
  }

  const body = results.rows
    .map((row) => {
      const conf = Math.max(Number(row.conf_level || 0), Number(row.conf_dept || 0));
      return `
        <div class="table-row">
          <span>${row.risk_text || ""}</span>
          <span>${row.pred_level ?? ""}</span>
          <span>${row.pred_dept ?? ""}</span>
          <span>${conf.toFixed(2)}</span>
        </div>`;
    })
    .join("");

  return `
    <h2>Results</h2>
    <p>Aggregated predictions from enabled methods.</p>
    <div class="table-placeholder">
      ${header}
      ${body}
    </div>
  `;
}

function mount(containerEl) {
  const paneEl = document.createElement("section");
  paneEl.className = "pane";
  containerEl.innerHTML = "";
  containerEl.appendChild(paneEl);
  return paneEl;
}

export default {
  async mount(containerEl) {
    rootEl = mount(containerEl);
    results = await fetchResults(50, 0);
  },
  async refresh() {
    if (!rootEl) return;
    try {
      results = await fetchResults(50, 0);
    } catch (_) {
      // ignore
    }
    rootEl.innerHTML = renderContent();
  },
  unmount() {
    if (rootEl && rootEl.parentElement) {
      rootEl.parentElement.removeChild(rootEl);
    }
    rootEl = null;
    results = null;
  },
};
