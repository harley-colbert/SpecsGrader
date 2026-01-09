import {
  fetchState,
  testVector,
  fetchSettings,
  saveSettings,
  fetchLabelPolicy,
  testLlm,
  loadDataset,
  fetchPreview,
  startClassify,
  fetchClassifyStatus,
  cancelClassify,
} from "../api/client.js";

let rootEl = null;
let stateCache = null;
let vectorResult = null;
let neverSend = false;
let llmResult = null;
let classifyFile = null;
let classifySummary = null;
let classifyPreview = [];
let classifyStatus = null;
let statusInterval = null;
let labelPolicy = null;
let labelPolicyError = "";
let labelPolicyLoading = false;
let thresholds = { level: 0.0, dept: 0.0, k: 5 };
let enabledMethods = { rules: true, vector: true, llm: true, model: true };
let llmModel = "openrouter/auto";
let mode = "production";
let overwritePredictions = true;
let overwriteSpecificRisk = true;
let policy = {
  model_conf_threshold: 0.75,
  vector_similarity_threshold: 0.45,
  vector_margin_threshold: 0.1,
  allow_llm: false,
  abstain_enabled: true,
};

function renderVectorResult(result) {
  const neighbors = (result.neighbors || [])
    .map((n) => `<li>${n.row.risk_text} (${n.row.label_dept}/${n.row.label_level}) sim=${Number(n.similarity).toFixed(2)}</li>`)
    .join("");
  return `
    <div class="metrics">
      <div><strong>Dept:</strong> ${result.dept_pred ?? "abstain"} (${Number(result.dept_conf).toFixed(2)})</div>
      <div><strong>Level:</strong> ${result.level_pred ?? "abstain"} (${Number(result.level_conf).toFixed(2)})</div>
      <div><strong>Top similarity:</strong> ${Number(result.top_similarity || 0).toFixed(2)}</div>
      <div><strong>Margin:</strong> ${Number(result.margin || 0).toFixed(2)}</div>
      <ul class="stats">${neighbors}</ul>
    </div>
  `;
}

function renderLlmResult(result) {
  return `
    <div class="metrics">
      <div><strong>Department:</strong> ${result.department ?? "abstain"}</div>
      <div><strong>Risk level:</strong> ${result.risk_level ?? "abstain"}</div>
      <div><strong>Confidence:</strong> ${result.confidence != null ? Number(result.confidence).toFixed(2) : "n/a"}</div>
      <div><strong>Reason:</strong> ${result.reason ?? ""}</div>
    </div>
  `;
}

function renderLabelPolicyList(items) {
  if (!Array.isArray(items) || !items.length) {
    return `<p class="muted">No definitions available.</p>`;
  }
  return `
    <ul class="label-policy-list">
      ${items
        .map(
          (item) => `
            <li>
              <strong>${item.label || item.id}</strong>
              <span class="muted">${item.description || ""}</span>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function renderLabelPolicy(policy) {
  if (!policy) {
    if (labelPolicyLoading) {
      return `<p class="muted">Loading label policy...</p>`;
    }
    if (labelPolicyError) {
      return `<p class="muted">Unable to load label policy: ${labelPolicyError}</p>`;
    }
    return `<p class="muted">Label policy not available.</p>`;
  }
  return `
    <details class="label-policy">
      <summary>Label definitions</summary>
      <div class="label-policy-grid">
        <div>
          <h5>Risk levels</h5>
          ${renderLabelPolicyList(policy.risk_levels)}
        </div>
        <div>
          <h5>Departments</h5>
          ${renderLabelPolicyList(policy.departments)}
        </div>
      </div>
    </details>
  `;
}

async function refreshLabelPolicy() {
  labelPolicyLoading = true;
  labelPolicyError = "";
  try {
    const resp = await fetchLabelPolicy();
    labelPolicy = resp?.policy || resp || null;
  } catch (error) {
    labelPolicy = null;
    labelPolicyError = error.message || "Unknown error";
  } finally {
    labelPolicyLoading = false;
  }
}

function render() {
  if (!rootEl) return;
  rootEl.innerHTML = `
    <h2>Classify</h2>
    <p>Load a classify file and run a job using enabled methods.</p>
    <div class="card-grid">
      <div class="card">
        <h3>Settings</h3>
        <label class="toggle">
          <input type="checkbox" id="never-send" ${neverSend ? "checked" : ""} />
          Never send externally (disables LLM calls)
        </label>
        ${neverSend ? `<p class="warning">LLM is disabled by Never Send mode.</p>` : ""}
      </div>
      <div class="card">
        <h3>Excel Mapping</h3>
        <p class="muted">Column D = input spec text, Column E = specific risk (auto-generated for medium+), Column F = predicted risk level, Column G = predicted department.</p>
      </div>
      <div class="card">
        <h3>Classify dataset</h3>
        <label class="field">
          <span>Select file (native OS picker)</span>
          <input type="file" id="classify-file" accept=".csv,.xlsx,.xls" />
          ${classifyFile ? `<small>Selected: ${classifyFile.name}</small>` : ""}
        </label>
        <div class="rule-actions">
          <button id="classify-load">Load file</button>
          ${classifySummary ? `<span class="chip">Rows: ${classifySummary.total_rows} | Missing text: ${classifySummary.missing_risk_text}</span>` : ""}
        </div>
        <div class="rule-actions">
          <label class="toggle"><input type="checkbox" id="overwrite-predictions" ${overwritePredictions ? "checked" : ""} />Overwrite existing F/G values</label>
        </div>
        <div class="rule-actions">
          <label class="toggle"><input type="checkbox" id="overwrite-specific-risk" ${overwriteSpecificRisk ? "checked" : ""} />Overwrite Column E specific risk (medium+)</label>
        </div>
        <p class="muted">Specific risk notes are auto-generated for medium+ risk levels only.</p>
        ${classifyPreview.length ? renderPreview(classifyPreview) : "<p>No preview loaded.</p>"}
      </div>
      <div class="card">
        <h3>Run classify job</h3>
        <label class="field">
          <span>Mode</span>
          <select id="mode-select">
            <option value="production" ${mode === "production" ? "selected" : ""}>production</option>
            <option value="evaluate" ${mode === "evaluate" ? "selected" : ""}>evaluate</option>
            <option value="sanity" ${mode === "sanity" ? "selected" : ""}>sanity</option>
          </select>
        </label>
        <div class="param-grid">
          <label class="field"><span>Level threshold</span><input type="number" step="0.05" id="thresh-level" value="${thresholds.level}" /></label>
          <label class="field"><span>Department threshold</span><input type="number" step="0.05" id="thresh-dept" value="${thresholds.dept}" /></label>
          <label class="field"><span>Neighbors (k)</span><input type="number" min="1" max="20" id="thresh-k" value="${thresholds.k}" /></label>
          <label class="field"><span>LLM model</span><input type="text" id="llm-model" value="${llmModel}" ${neverSend ? "disabled" : ""} /></label>
          <label class="field"><span>Model conf threshold</span><input type="number" step="0.05" min="0" max="1" id="policy-model-conf" value="${policy.model_conf_threshold}" /></label>
          <label class="field"><span>Vector similarity threshold</span><input type="number" step="0.05" min="0" max="1" id="policy-vector-sim" value="${policy.vector_similarity_threshold}" /></label>
          <label class="field"><span>Vector margin threshold</span><input type="number" step="0.05" min="0" max="1" id="policy-vector-margin" value="${policy.vector_margin_threshold}" /></label>
        </div>
        <div class="rule-actions">
          <label class="toggle"><input type="checkbox" id="method-model" ${enabledMethods.model ? "checked" : ""} />Model</label>
          <label class="toggle"><input type="checkbox" id="method-rules" ${enabledMethods.rules ? "checked" : ""} />Rules</label>
          <label class="toggle"><input type="checkbox" id="method-vector" ${enabledMethods.vector ? "checked" : ""} />Vector</label>
          <label class="toggle"><input type="checkbox" id="method-llm" ${enabledMethods.llm && !neverSend ? "checked" : ""} ${neverSend ? "disabled" : ""} />LLM</label>
          <label class="toggle"><input type="checkbox" id="policy-allow-llm" ${policy.allow_llm && !neverSend ? "checked" : ""} ${neverSend ? "disabled" : ""} />Allow LLM fallback</label>
          <label class="toggle"><input type="checkbox" id="policy-abstain" ${policy.abstain_enabled ? "checked" : ""} />Allow abstain</label>
        </div>
        <p class="muted">Defaults for mode: ${mode}</p>
        <div class="rule-actions">
          <button id="classify-start">Start classify</button>
          <button id="classify-cancel">Cancel</button>
          <span class="chip">Status: ${classifyStatus?.status || "idle"} (${classifyStatus?.processed || 0}/${classifyStatus?.total || 0})</span>
        </div>
      </div>
      <div class="card">
        <h3>Vector test</h3>
        <input type="text" id="vector-text" placeholder="Enter text to test vector" />
        <button id="vector-run">Test vector</button>
        ${vectorResult ? renderVectorResult(vectorResult) : ""}
      </div>
      <div class="card">
        <h3>LLM test</h3>
        <input type="text" id="llm-text" placeholder="Enter text to test LLM" />
        <button id="llm-run" ${neverSend ? "disabled" : ""}>Test LLM</button>
        ${llmResult ? renderLlmResult(llmResult) : ""}
      </div>
      <div class="card">
        <h3>Label definitions</h3>
        ${renderLabelPolicy(labelPolicy)}
      </div>
    </div>
  `;

  const neverSendBox = rootEl.querySelector("#never-send");
  neverSendBox?.addEventListener("change", async () => {
    try {
      neverSend = !!neverSendBox.checked;
      await saveSettings(neverSend);
      llmResult = null;
      if (neverSend) {
        enabledMethods.llm = false;
      }
      render();
    } catch (error) {
      alert(error.message);
    }
  });

  const classifyInput = rootEl.querySelector("#classify-file");
  classifyInput?.addEventListener("change", () => {
    classifyFile = classifyInput.files?.[0] || null;
    render();
  });

  const classifyLoadBtn = rootEl.querySelector("#classify-load");
  classifyLoadBtn?.addEventListener("click", async () => {
    if (!classifyFile) {
      alert("Please select a classify file.");
      return;
    }
    try {
      classifySummary = await loadDataset("classify", classifyFile);
      const previewResp = await fetchPreview("classify", 10, 0);
      classifyPreview = previewResp.rows || [];
      classifyFile = null;
      render();
    } catch (error) {
      alert(error.message);
    }
  });

  const methodModel = rootEl.querySelector("#method-model");
  const methodRules = rootEl.querySelector("#method-rules");
  const methodVector = rootEl.querySelector("#method-vector");
  const methodLlm = rootEl.querySelector("#method-llm");
  const allowLlmToggle = rootEl.querySelector("#policy-allow-llm");
  const abstainToggle = rootEl.querySelector("#policy-abstain");
  const modeSelect = rootEl.querySelector("#mode-select");
  const policyModelConf = rootEl.querySelector("#policy-model-conf");
  const policyVectorSim = rootEl.querySelector("#policy-vector-sim");
  const policyVectorMargin = rootEl.querySelector("#policy-vector-margin");

  modeSelect?.addEventListener("change", () => {
    mode = modeSelect.value;
    if (mode === "sanity") {
      enabledMethods = { rules: false, vector: false, llm: false, model: true };
    } else if (mode === "evaluate") {
      enabledMethods = { rules: true, vector: true, llm: !neverSend, model: true };
    } else {
      enabledMethods = { rules: true, vector: true, llm: !neverSend, model: true };
    }
    render();
  });
  [methodModel, methodRules, methodVector, methodLlm].forEach((checkbox) => {
    checkbox?.addEventListener("change", () => {
      enabledMethods = {
        ...enabledMethods,
        model: !!methodModel?.checked,
        rules: !!methodRules?.checked,
        vector: !!methodVector?.checked,
        llm: !!methodLlm?.checked && !neverSend,
      };
      render();
    });
  });

  allowLlmToggle?.addEventListener("change", () => {
    policy.allow_llm = !!allowLlmToggle.checked && !neverSend;
    render();
  });

  abstainToggle?.addEventListener("change", () => {
    policy.abstain_enabled = !!abstainToggle.checked;
    render();
  });

  const threshLevel = rootEl.querySelector("#thresh-level");
  const threshDept = rootEl.querySelector("#thresh-dept");
  const threshK = rootEl.querySelector("#thresh-k");
  const llmModelInput = rootEl.querySelector("#llm-model");
  const overwriteToggle = rootEl.querySelector("#overwrite-predictions");
  const overwriteSpecificToggle = rootEl.querySelector("#overwrite-specific-risk");

  const classifyStartBtn = rootEl.querySelector("#classify-start");
  classifyStartBtn?.addEventListener("click", async () => {
    thresholds = {
      level: parseFloat(threshLevel?.value ?? thresholds.level) || 0,
      dept: parseFloat(threshDept?.value ?? thresholds.dept) || 0,
      k: parseInt(threshK?.value ?? thresholds.k, 10) || 5,
    };
    policy = {
      ...policy,
      model_conf_threshold: parseFloat(policyModelConf?.value ?? policy.model_conf_threshold) || 0,
      vector_similarity_threshold: parseFloat(policyVectorSim?.value ?? policy.vector_similarity_threshold) || 0,
      vector_margin_threshold: parseFloat(policyVectorMargin?.value ?? policy.vector_margin_threshold) || 0,
      allow_llm: policy.allow_llm && !neverSend,
      abstain_enabled: policy.abstain_enabled,
    };
    llmModel = llmModelInput?.value?.trim() || llmModel;
    try {
      classifyStatus = await startClassify({
        mode,
        thresholds,
        policy,
        enabled_methods: enabledMethods,
        k: thresholds.k,
        llm_model: llmModel,
        overwrite_predictions: overwritePredictions,
      });
      startStatusPolling();
      render();
    } catch (error) {
      alert(error.message);
    }
  });

  const classifyCancelBtn = rootEl.querySelector("#classify-cancel");
  classifyCancelBtn?.addEventListener("click", async () => {
    try {
      await cancelClassify();
      classifyStatus = await fetchClassifyStatus();
      stopStatusPolling();
      render();
    } catch (error) {
      alert(error.message);
    }
  });

  overwriteToggle?.addEventListener("change", () => {
    overwritePredictions = !!overwriteToggle.checked;
  });
  overwriteSpecificToggle?.addEventListener("change", () => {
    overwriteSpecificRisk = !!overwriteSpecificToggle.checked;
  });

  const vectorBtn = rootEl.querySelector("#vector-run");
  const vectorInput = rootEl.querySelector("#vector-text");
  vectorBtn?.addEventListener("click", async () => {
    const text = vectorInput?.value.trim();
    if (!text) return;
    try {
      vectorResult = await testVector(text, 3);
      render();
    } catch (error) {
      alert(error.message);
    }
  });

  const llmBtn = rootEl.querySelector("#llm-run");
  const llmInput = rootEl.querySelector("#llm-text");
  llmBtn?.addEventListener("click", async () => {
    const text = llmInput?.value.trim();
    if (!text || neverSend) return;
    try {
      llmResult = await testLlm(text, "openrouter/auto");
      render();
    } catch (error) {
      alert(error.message);
    }
  });
}

function renderPreview(rows) {
  const header = `
    <div class="table-header">
      <span>Row</span>
      <span>Spec text</span>
      <span>Specific risk</span>
      <span>Risk level</span>
      <span>Department</span>
    </div>`;
  const body = rows
    .map(
      (row) => `
        <div class="table-row">
          <span>${row.source_row}</span>
          <span>${row.spec_text || ""}</span>
          <span>${row.specific_risk_existing || ""}</span>
          <span>${row.risk_level_existing || ""}</span>
          <span>${row.dept_existing || ""}</span>
        </div>`
    )
    .join("");
  return `<div class="table-placeholder">${header}${body}</div>`;
}

function stopStatusPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

function startStatusPolling() {
  stopStatusPolling();
  statusInterval = setInterval(async () => {
    try {
      classifyStatus = await fetchClassifyStatus();
      if (classifyStatus?.status !== "running") {
        stopStatusPolling();
      }
      render();
    } catch (error) {
      console.error("Failed to poll classify status", error);
      stopStatusPolling();
    }
  }, 1000);
}

export default {
  async mount(containerEl) {
    rootEl = document.createElement("section");
    rootEl.className = "pane";
    containerEl.innerHTML = "";
    containerEl.appendChild(rootEl);

    stateCache = await fetchState();
    if (stateCache?.production_policy) {
      policy = { ...policy, ...stateCache.production_policy };
    }
    const settings = await fetchSettings();
    neverSend = !!settings.never_send_externally;
    await refreshLabelPolicy();
    render();
  },
  refresh() {
    render();
  },
  unmount() {
    stopStatusPolling();
    if (rootEl && rootEl.parentElement) {
      rootEl.parentElement.removeChild(rootEl);
    }
    rootEl = null;
    stateCache = null;
    vectorResult = null;
    llmResult = null;
    classifyFile = null;
    classifySummary = null;
    classifyPreview = [];
    classifyStatus = null;
    labelPolicy = null;
    labelPolicyError = "";
    labelPolicyLoading = false;
  },
};
