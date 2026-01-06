import {
  fetchPreview,
  loadDataset,
  fetchRules,
  saveRules,
  testRules,
  startTraining,
  fetchTrainingStatus,
  cancelTraining,
  fetchTrainingMetrics,
  buildVectorStore,
} from "../api/client.js";

let rootEl = null;
let summary = null;
let preview = [];
let loading = false;
let trainingFile = null;
let rulesConfig = null;
let testResult = null;
let trainingStatus = null;
let trainingMetrics = null;
let vectorBuildStatus = null;
let trainingParams = {
  oversample_enabled: false,
  oversample_cap_ratio: 0.3,
  min_recall_per_class: 0.5,
  calibration_method: "sigmoid",
};

function render(state) {
  if (!rootEl) return;
  rootEl.innerHTML = `
    <h2>Train</h2>
    <p>Load a training file, configure rules, train models, and build a vector store.</p>
    <div class="card">
      <label class="field">
        <span>Training file (uses native OS picker)</span>
        <input type="file" id="train-file" accept=".csv,.xlsx,.xls" />
        ${trainingFile ? `<small>Selected: ${trainingFile.name}</small>` : ""}
      </label>
      <button id="train-load" ${loading ? "disabled" : ""}>${loading ? "Loading..." : "Load"}</button>
    </div>
    <div class="card-grid">
      <div class="card">
        <h3>Summary</h3>
        ${summary ? `
          <ul class="stats">
            <li>Total rows: ${summary.total_rows}</li>
            <li>Missing risk text: ${summary.missing_risk_text}</li>
            <li>Missing labels: ${summary.missing_labels}</li>
            <li>Invalid levels: ${summary.invalid_levels}</li>
            <li>Invalid departments: ${summary.invalid_departments}</li>
          </ul>
        ` : `<p>No dataset loaded.</p>`}
      </div>
      <div class="card">
        <h3>Preview</h3>
        ${preview.length ? renderPreview(preview) : `<p>No preview available.</p>`}
      </div>
    </div>
    <div class="card">
      <h3>Training parameters</h3>
      <div class="param-grid">
        <label class="field">
          <span>Oversample enabled</span>
          <input type="checkbox" id="oversample-enabled" ${trainingParams.oversample_enabled ? "checked" : ""} />
        </label>
        <label class="field">
          <span>Oversample cap ratio</span>
          <input type="number" step="0.1" min="0" max="1" id="oversample-cap" value="${trainingParams.oversample_cap_ratio}" />
        </label>
        <label class="field">
          <span>Min recall per class</span>
          <input type="number" step="0.05" min="0" max="1" id="min-recall" value="${trainingParams.min_recall_per_class}" />
        </label>
        <label class="field">
          <span>Calibration method</span>
          <select id="calibration-method">
            <option value="sigmoid" ${trainingParams.calibration_method === "sigmoid" ? "selected" : ""}>sigmoid</option>
            <option value="isotonic" ${trainingParams.calibration_method === "isotonic" ? "selected" : ""}>isotonic</option>
          </select>
        </label>
      </div>
      <div class="rule-actions">
        <button id="train-start">Train</button>
        <button id="train-cancel">Cancel</button>
        <span class="chip">Status: ${trainingStatus?.status || "idle"}</span>
      </div>
      ${trainingMetrics ? renderMetrics(trainingMetrics) : ""}
    </div>
    <div class="card">
      <h3>Vector store</h3>
      <p>Build embeddings and ANN index for vector-based predictions.</p>
      <div class="rule-actions">
        <button id="vector-build">Build vector store</button>
        <span class="chip">${vectorBuildStatus || "Not built"}</span>
      </div>
    </div>
    <div class="card">
      <h3>Rules configuration</h3>
      <p>Edit JSON rules per department. Test a sample risk text below.</p>
      <div class="field">
        <textarea id="rules-json" rows="10" class="code-area">${rulesConfig ? JSON.stringify(rulesConfig, null, 2) : ""}</textarea>
      </div>
      <div class="rule-actions">
        <button id="rules-save">Save rules</button>
        <input type="text" id="rule-test-text" placeholder="Enter risk text to test" />
        <button id="rules-test">Test rule</button>
      </div>
      <div class="rule-result">${renderRuleResult()}</div>
    </div>
  `;

  const loadBtn = rootEl.querySelector("#train-load");
  const fileInput = rootEl.querySelector("#train-file");
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      trainingFile = fileInput.files?.[0] || null;
      render(state);
    });
  }
  if (loadBtn) {
    loadBtn.addEventListener("click", async () => {
      if (!trainingFile) {
        alert("Please select a training file.");
        return;
      }
      loading = true;
      render(state);
      try {
        summary = await loadDataset("train", trainingFile);
        const previewResp = await fetchPreview("train", 10, 0);
        preview = previewResp.rows || [];
      } catch (error) {
        alert(error.message);
      } finally {
        loading = false;
        trainingFile = null;
        render(state);
      }
    });
  }

  const oversampleCheckbox = rootEl.querySelector("#oversample-enabled");
  const oversampleCapInput = rootEl.querySelector("#oversample-cap");
  const minRecallInput = rootEl.querySelector("#min-recall");
  const calibrationSelect = rootEl.querySelector("#calibration-method");
  const startBtn = rootEl.querySelector("#train-start");
  const cancelBtn = rootEl.querySelector("#train-cancel");
  const vectorBtn = rootEl.querySelector("#vector-build");

  const syncParams = () => {
    trainingParams.oversample_enabled = oversampleCheckbox?.checked || false;
    trainingParams.oversample_cap_ratio = parseFloat(oversampleCapInput?.value || trainingParams.oversample_cap_ratio);
    trainingParams.min_recall_per_class = parseFloat(minRecallInput?.value || trainingParams.min_recall_per_class);
    trainingParams.calibration_method = calibrationSelect?.value || trainingParams.calibration_method;
  };

  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      try {
        syncParams();
        await startTraining(trainingParams);
        trainingStatus = await fetchTrainingStatus();
        try {
          trainingMetrics = await fetchTrainingMetrics();
        } catch (_) {
          trainingMetrics = null;
        }
        render(state);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (cancelBtn) {
    cancelBtn.addEventListener("click", async () => {
      await cancelTraining();
      trainingStatus = await fetchTrainingStatus();
      render(state);
    });
  }

  if (vectorBtn) {
    vectorBtn.addEventListener("click", async () => {
      try {
        const resp = await buildVectorStore();
        vectorBuildStatus = resp.built ? `Built at ${resp.path}` : "Not built";
        render(state);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  const saveBtn = rootEl.querySelector("#rules-save");
  const rulesTextArea = rootEl.querySelector("#rules-json");
  const testBtn = rootEl.querySelector("#rules-test");
  const testInput = rootEl.querySelector("#rule-test-text");

  if (saveBtn && rulesTextArea) {
    saveBtn.addEventListener("click", async () => {
      try {
        const parsed = JSON.parse(rulesTextArea.value);
        rulesConfig = await saveRules(parsed);
        alert("Rules saved");
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (testBtn && testInput) {
    testBtn.addEventListener("click", async () => {
      try {
        const parsed = rulesTextArea?.value ? JSON.parse(rulesTextArea.value) : rulesConfig;
        testResult = await testRules(testInput.value || "", parsed);
        render(state);
      } catch (error) {
        alert(error.message);
      }
    });
  }
}

function renderPreview(rows) {
  const header = `
    <div class="table-header">
      <span>Row</span>
      <span>Risk text</span>
      <span>Level</span>
      <span>Department</span>
    </div>`;
  const body = rows
    .map(
      (row) => `
        <div class="table-row">
          <span>${row.source_row}</span>
          <span>${row.risk_text || ""}</span>
          <span>${row.label_level || ""}</span>
          <span>${row.label_dept || ""}</span>
        </div>`
    )
    .join("");
  return `<div class="table-placeholder">${header}${body}</div>`;
}

function renderRuleResult() {
  if (!testResult) return "No test run.";
  const { dept_pred, dept_conf, matched } = testResult;
  const matchedList = matched
    ? Object.entries(matched)
        .map(([dept, hits]) => `${dept}: ${hits.join(", ")}`)
        .join(" | ")
    : "";
  return `<strong>Prediction:</strong> ${dept_pred ?? "Abstain"} (conf ${Number(dept_conf).toFixed(2)})<br/><small>Matched: ${matchedList}</small>`;
}

function renderMetrics(metrics) {
  if (!metrics) return "";
  const level = metrics.level || {};
  const dept = metrics.dept || {};
  const recallRow = (data) =>
    Object.entries(data || {})
      .map(([cls, rec]) => `<li>${cls}: ${Number(rec).toFixed(2)}</li>`)
      .join("");
  return `
    <div class="metrics">
      <div>
        <h4>Risk level</h4>
        <p>Macro F1: ${Number(level.macro_f1 || 0).toFixed(2)} | Balanced acc: ${Number(level.balanced_accuracy || 0).toFixed(2)}</p>
        <ul class="stats">${recallRow(level.per_class_recall)}</ul>
      </div>
      <div>
        <h4>Department</h4>
        <p>Macro F1: ${Number(dept.macro_f1 || 0).toFixed(2)} | Balanced acc: ${Number(dept.balanced_accuracy || 0).toFixed(2)}</p>
        <ul class="stats">${recallRow(dept.per_class_recall)}</ul>
      </div>
    </div>
  `;
}

export default {
  async mount(containerEl) {
    rootEl = document.createElement("section");
    rootEl.className = "pane";
    containerEl.innerHTML = "";
    containerEl.appendChild(rootEl);
    try {
      rulesConfig = await fetchRules();
    } catch (error) {
      console.error("Failed to fetch rules", error);
      rulesConfig = null;
    }
  },
  refresh(state) {
    render(state);
  },
  unmount() {
    if (rootEl && rootEl.parentElement) {
      rootEl.parentElement.removeChild(rootEl);
    }
    rootEl = null;
    summary = null;
    preview = [];
    loading = false;
    trainingFile = null;
    testResult = null;
    trainingStatus = null;
    trainingMetrics = null;
    vectorBuildStatus = null;
  },
};
