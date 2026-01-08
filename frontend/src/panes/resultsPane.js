import { fetchResults } from "../api/client.js";

let rootEl = null;
let results = null;

function renderEvidenceList(items, formatter) {
  if (!items || !items.length) {
    return `<p class="muted">None</p>`;
  }
  return `<ul class="stats">${items.map(formatter).join("")}</ul>`;
}

function renderWhy(trace) {
  if (!trace) {
    return `<p class="muted">No trace available.</p>`;
  }
  const steps = Array.isArray(trace.steps) ? trace.steps : [];
  const winner = trace.winner ? `Winner: ${trace.winner}` : "Winner: n/a";
  const selected = steps.filter((step) => step.selected).map((step) => step.step).join(", ");
  const evidence = trace.evidence || {};
  const rules = evidence.rules || {};
  const model = evidence.model || {};
  const vector = evidence.vector || {};

  const levelTerms = renderEvidenceList(model.level_top_terms, (term) => `<li>${term.term} (${Number(term.weight).toFixed(3)})</li>`);
  const deptTerms = renderEvidenceList(model.dept_top_terms, (term) => `<li>${term.term} (${Number(term.weight).toFixed(3)})</li>`);
  const neighbors = renderEvidenceList(vector.neighbors, (neighbor) => {
    const label = `${neighbor.label_dept ?? "?"}/${neighbor.label_level ?? "?"}`;
    return `<li>Row ${neighbor.source_row ?? "?"}: ${label} (sim ${Number(neighbor.similarity ?? 0).toFixed(2)})</li>`;
  });
  return `
    <div class="why-details">
      <p><strong>${winner}</strong></p>
      ${selected ? `<p class="muted">Selected steps: ${selected}</p>` : ""}
      <div class="why-section">
        <h4>Rules evidence</h4>
        <p class="muted">Hard: ${rules.is_hard ? "yes" : "no"}</p>
        ${renderEvidenceList(rules.matched, (hit) => `<li>${hit}</li>`)}
      </div>
      <div class="why-section">
        <h4>Model evidence</h4>
        <p class="muted">Level proba: ${model.level_proba ? JSON.stringify(model.level_proba) : "n/a"}</p>
        <p class="muted">Dept proba: ${model.dept_proba ? JSON.stringify(model.dept_proba) : "n/a"}</p>
        <div class="why-columns">
          <div>
            <strong>Level terms</strong>
            ${levelTerms}
          </div>
          <div>
            <strong>Dept terms</strong>
            ${deptTerms}
          </div>
        </div>
      </div>
      <div class="why-section">
        <h4>Vector evidence</h4>
        <p class="muted">Top similarity: ${Number(vector.top_similarity || 0).toFixed(2)} | Margin: ${Number(vector.margin || 0).toFixed(2)}</p>
        ${neighbors}
      </div>
    </div>
  `;
}

function renderContent() {
  const header = `
    <div class="table-header">
      <span>Risk text</span>
      <span>Predicted level</span>
      <span>Department</span>
      <span>Confidence</span>
      <span>Why</span>
    </div>`;

  if (!results || !results.rows || !results.rows.length) {
    return `
      <h2>Results</h2>
      <p>Results will appear here after a classify job runs.</p>
      <div class="table-placeholder table-5">
        ${header}
        <div class="table-row muted">No results yet</div>
      </div>
    `;
  }

  const body = results.rows
    .map((row) => {
      const conf = Math.max(Number(row.conf_level || 0), Number(row.conf_dept || 0));
      let trace = null;
      try {
        trace = row.trace ? JSON.parse(row.trace) : null;
      } catch (_) {
        trace = null;
      }
      const winner = trace?.winner ? `Winner: ${trace.winner}` : "Winner: n/a";
      return `
        <div class="table-row">
          <span>${row.risk_text || ""}</span>
          <span>${row.pred_level ?? ""}</span>
          <span>${row.pred_dept ?? ""}</span>
          <span>${conf.toFixed(2)}</span>
          <span>
            ${winner}
            <details class="why-toggle">
              <summary>Why?</summary>
              ${renderWhy(trace)}
            </details>
          </span>
        </div>`;
    })
    .join("");

  return `
    <h2>Results</h2>
    <p>Aggregated predictions from enabled methods.</p>
    <div class="table-placeholder table-5">
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
