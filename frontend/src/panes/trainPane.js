import {
  fetchState,
  fetchPreview,
  loadDataset,
  fetchDatasetHealth,
  fetchLabelPolicy,
  fetchRules,
  saveRules,
  testRules,
  setActivePane,
  startTraining,
  fetchTrainingStatus,
  cancelTraining,
  runSanityCheck,
  runEvaluate,
  buildVectorStore,
  listModelSets,
  createModelSet,
  updateModelSet,
  deleteModelSet,
  saveModelSetVersion,
  deleteModelSetVersion,
  loadModelSet,
  exportModelSetUrl,
  importModelSet,
  fetchModelInsights,
} from "../api/client.js";

let rootEl = null;
let lastState = null;
let storeRef = null;
let summary = null;
let preview = [];
let loading = false;
let trainingFile = null;
let datasetHealth = null;
let labelPolicy = null;
let labelPolicyError = "";
let labelPolicyLoading = false;
let rulesConfig = null;
let testResult = null;
let trainingStatus = null;
let trainingMetrics = null;
let insightsData = null;
let insightsLoading = false;
let insightsError = "";
let insightsType = "level";
let insightsClass = "";
let insightsTopN = 20;
let vectorBuildStatus = null;
let trainingPollTimer = null;
let sanityReport = null;
let sanityLoading = false;
let evaluationReport = null;
let evaluationLoading = false;
let validationErrorMessage = "";
let validationCompleted = false;
let validationFile = null;
let validationSummary = null;
let validationPreview = [];
let validationSanityReport = null;
let validationEvaluationReport = null;
let validationSanityLoading = false;
let validationEvaluationLoading = false;
let validationAccordionOpen = false;
let validationError = "";
let rulesEditorOpen = false;
let rulesDraft = "";
let rulesDirty = false;
let rulesModified = false;
let rulesErrorMessage = "";
let rulesFocusTest = false;
let trainingAdvancedOpen = false;

let modelsets = [];
let modelsetsLoading = false;
let modelsetSelectId = "";
let modelsetVersionSelectId = "";
let newModelsetId = "";
let newModelsetName = "";
let newModelsetDescription = "";
let editModelsetName = "";
let editModelsetDescription = "";
let editModelsetTags = "";
let importFile = null;
let mode = "use-existing";
let modelsetLoadTab = "local";
let trainingParams = {
  oversample_enabled: false,
  oversample_cap_ratio: 0.3,
  min_recall_per_class: 0.5,
  calibration_method: "sigmoid",
  cv_folds: 5,
  use_class_weight_balanced: true,
};

function stopTrainingPoll() {
  if (trainingPollTimer) {
    clearInterval(trainingPollTimer);
    trainingPollTimer = null;
  }
}

async function refreshModelSets() {
  modelsetsLoading = true;
  try {
    const resp = await listModelSets();
    modelsets = resp.modelsets || [];
    // Keep selection stable when possible
    if (!modelsetSelectId && modelsets.length) {
      modelsetSelectId = modelsets[0].modelset_id;
    }
    const selected = modelsets.find((m) => m.modelset_id === modelsetSelectId) || null;
    if (selected) {
      const versions = selected.versions || [];
      if (!modelsetVersionSelectId && versions.length) {
        modelsetVersionSelectId = versions[0].version_id;
      }
      if (modelsetVersionSelectId && versions.length && !versions.find((v) => v.version_id === modelsetVersionSelectId)) {
        modelsetVersionSelectId = versions[0].version_id;
      }
    } else {
      modelsetVersionSelectId = "";
    }
    syncModelsetEditor();
  } catch (error) {
    console.error("Failed to refresh modelsets", error);
  } finally {
    modelsetsLoading = false;
  }
}

async function syncState() {
  try {
    const s = await fetchState();
    if (storeRef && typeof storeRef.setState === "function") {
      storeRef.setState(s);
    }
    lastState = s;
    return s;
  } catch (error) {
    console.error("Failed to fetch state", error);
    return lastState;
  }
}

async function pollTrainingStatus() {
  try {
    trainingStatus = await fetchTrainingStatus();
    trainingMetrics = trainingStatus?.metrics || null;
    if (trainingStatus?.stats) {
      datasetHealth = trainingStatus.stats;
    }
  } catch (error) {
    console.error("Failed to poll training status", error);
  }

  if (trainingStatus && trainingStatus.status !== "running") {
    stopTrainingPoll();
  }

  render(lastState);
}

function startTrainingPoll() {
  stopTrainingPoll();
  trainingPollTimer = setInterval(pollTrainingStatus, 900);
}

async function refreshDatasetHealth() {
  try {
    datasetHealth = await fetchDatasetHealth("train");
  } catch (error) {
    datasetHealth = null;
  }
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

function safeString(value) {
  return value === null || value === undefined ? "" : String(value);
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0%";
  }
  return `${Number(value).toFixed(1)}%`;
}

function renderDistributionTable(title, counts, pct) {
  const entries = Object.entries(counts || {});
  if (!entries.length) {
    return `
      <div class="card card-inset">
        <h4>${title}</h4>
        <p class="muted">No labeled rows to summarize.</p>
      </div>
    `;
  }
  return `
    <div class="card card-inset">
      <h4>${title}</h4>
      <table class="table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Count</th>
            <th>%</th>
          </tr>
        </thead>
        <tbody>
          ${entries
            .map(([label, count]) => {
              const pctValue = pct?.[label] ?? 0;
              return `
                <tr>
                  <td>${safeString(label)}</td>
                  <td>${count}</td>
                  <td>${formatPct(pctValue)}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
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
              <strong>${safeString(item.label || item.id)}</strong>
              <span class="muted">${safeString(item.description)}</span>
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
      return `<p class="muted">Unable to load label policy: ${safeString(labelPolicyError)}</p>`;
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

function renderDecisionPolicy(policy) {
  if (!policy) {
    return `<p class="muted">Decision policy unavailable for this version.</p>`;
  }
  const layers = Array.isArray(policy.layers) ? policy.layers : [];
  const weights = policy.weights || {};
  if (!layers.length) {
    return `<p class="muted">Decision policy has no layers.</p>`;
  }
  const layerRows = layers
    .map((layer, index) => {
      const type = safeString(layer.type);
      const id = safeString(layer.id || type || `layer-${index + 1}`);
      const enabled = layer.enabled === false ? "Disabled" : "Enabled";
      const details = [];
      if (type === "model_confidence") {
        details.push(`Min confidence ≥ ${safeString(layer.min_confidence ?? 0)}`);
      }
      if (type === "model_vector_consensus") {
        details.push(`Min similarity ≥ ${safeString(layer.min_similarity ?? 0)}`);
      }
      if (type === "vector_confidence") {
        details.push(`Min similarity ≥ ${safeString(layer.min_similarity ?? 0)}`);
        details.push(`Min margin ≥ ${safeString(layer.min_margin ?? 0)}`);
      }
      if (type === "weighted") {
        details.push(
          `Weights: model ${safeString(weights.model ?? 0)}, vector ${safeString(weights.vector ?? 0)}, llm ${safeString(
            weights.llm ?? 0
          )}, rules ${safeString(weights.rules ?? 0)}`
        );
      }
      if (type === "llm" || type === "abstain") {
        details.push(enabled);
      }
      const criteria = details.length ? details.join(" · ") : enabled;
      return `
        <tr>
          <td>${index + 1}. ${id}</td>
          <td>${safeString(type)}</td>
          <td>${criteria}</td>
        </tr>
      `;
    })
    .join("");

  return `
    <div class="card card-inset">
      <h4>Decision policy</h4>
      <table class="table">
        <thead>
          <tr>
            <th>Order</th>
            <th>Layer type</th>
            <th>Criteria</th>
          </tr>
        </thead>
        <tbody>
          ${layerRows}
        </tbody>
      </table>
    </div>
  `;
}

function formatTs(ts) {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return String(ts);
    return d.toLocaleString();
  } catch (_) {
    return String(ts);
  }
}

function formatSecondsSince(ts) {
  if (!ts) return "";
  try {
    const start = new Date(ts).getTime();
    if (Number.isNaN(start)) return "";
    const seconds = Math.max(0, Math.floor((Date.now() - start) / 1000));
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m <= 0) return `${s}s`;
    return `${m}m ${s}s`;
  } catch (_) {
    return "";
  }
}

function updateVectorBuildStatusFromState(state) {
  const vectorState = state?.vector_store || null;
  if (!vectorState) {
    vectorBuildStatus = null;
    return;
  }
  if (vectorState.built) {
    vectorBuildStatus = vectorState.path ? `Built at ${vectorState.path}` : "Built";
    return;
  }
  vectorBuildStatus = "Not built";
}

function renderMiniJson(obj) {
  try {
    return `<pre class="mini-code">${safeString(JSON.stringify(obj, null, 2))}</pre>`;
  } catch (_) {
    return `<pre class="mini-code">(Unable to format)</pre>`;
  }
}

function renderCvSummary(cvMetrics, cvStatus) {
  if (!cvMetrics) {
    if (cvStatus?.label) {
      const current = Number(cvStatus.current_fold || 0);
      const total = Number(cvStatus.total_folds || 0);
      return `
        <div class="card card-inset">
          <h4>Cross-validation</h4>
          <p class="muted">Running ${safeString(cvStatus.label)} fold ${current} of ${total}...</p>
        </div>
      `;
    }
    return "";
  }
  const renderBlock = (label, metrics) => {
    const averages = metrics?.averages || {};
    return `
      <div class="card card-inset">
        <h4>CV summary — ${safeString(label)}</h4>
        <ul class="stats">
          <li>Macro F1: ${Number(averages.macro_f1 || 0).toFixed(2)}</li>
          <li>Weighted F1: ${Number(averages.weighted_f1 || 0).toFixed(2)}</li>
          <li>Balanced accuracy: ${Number(averages.balanced_accuracy || 0).toFixed(2)}</li>
        </ul>
      </div>
    `;
  };
  return `
    <div class="cv-summary">
      ${renderBlock("Risk level", cvMetrics.level)}
      ${renderBlock("Department", cvMetrics.dept)}
    </div>
  `;
}

function renderImbalanceDistributions(preDist, postDist) {
  if (!preDist && !postDist) {
    return "";
  }
  const renderSection = (title, dist) => {
    if (!dist) {
      return `<p class="muted">No ${title.toLowerCase()} distribution recorded.</p>`;
    }
    const levelEntries = Object.entries(dist.level || {});
    const deptEntries = Object.entries(dist.dept || {});
    return `
      <div class="card card-inset">
        <h4>${title} distribution</h4>
        <div class="card-grid">
          <div class="card">
            <h5>Risk levels</h5>
            ${levelEntries.length
              ? `<ul class="stats">${levelEntries
                  .map(([label, count]) => `<li>${safeString(label)}: ${count}</li>`)
                  .join("")}</ul>`
              : `<p class="muted">No level distribution recorded.</p>`}
          </div>
          <div class="card">
            <h5>Departments</h5>
            ${deptEntries.length
              ? `<ul class="stats">${deptEntries
                  .map(([label, count]) => `<li>${safeString(label)}: ${count}</li>`)
                  .join("")}</ul>`
              : `<p class="muted">No department distribution recorded.</p>`}
          </div>
        </div>
      </div>
    `;
  };
  return `
    <div class="imbalance-summary">
      ${renderSection("Pre-oversample", preDist)}
      ${renderSection("Post-oversample", postDist)}
    </div>
  `;
}

function renderTrainingDetails(job) {
  if (!job) return "";
  const params = job.params || null;
  const stats = job.stats || null;
  const events = Array.isArray(job.events) ? job.events : [];
  const recent = events.slice(-12).reverse();

  return `
    <div class="train-telemetry">
      <div class="telemetry-grid">
        <div class="telemetry-block">
          <h4>Effective params</h4>
          ${params ? renderMiniJson(params) : `<p class="muted">No params recorded yet.</p>`}
        </div>
        <div class="telemetry-block">
          <h4>Dataset stats</h4>
          ${stats ? renderMiniJson(stats) : `<p class="muted">No stats available yet.</p>`}
        </div>
      </div>
      <div class="telemetry-block">
        <h4>Recent events</h4>
        ${recent.length ? `
          <ul class="event-log">
            ${recent
              .map((ev) => {
                const ts = formatTs(ev.ts);
                const lvl = safeString(ev.level || "info");
                const msg = safeString(ev.message || "");
                const data = ev.data ? ` <span class="muted">${safeString(JSON.stringify(ev.data))}</span>` : "";
                return `<li><span class="event-ts">${ts}</span> <span class="event-level">${lvl}</span> ${msg}${data}</li>`;
              })
              .join("")}
          </ul>
        ` : `<p class="muted">No events yet.</p>`}
      </div>
    </div>
  `;
}

function getSelectedModelset() {
  return modelsets.find((ms) => ms.modelset_id === modelsetSelectId) || null;
}

function getSelectedModelsetVersion() {
  const ms = getSelectedModelset();
  const versions = ms?.versions || [];
  return versions.find((version) => version.version_id === modelsetVersionSelectId) || null;
}

function renderModelsetSelectOptions() {
  if (!modelsets.length) {
    return `<option value="">(none)</option>`;
  }
  return modelsets
    .map((ms) => {
      const selected = ms.modelset_id === modelsetSelectId ? "selected" : "";
      const label = `${ms.name} (${ms.modelset_id})`;
      return `<option value="${ms.modelset_id}" ${selected}>${label}</option>`;
    })
    .join("");
}

function renderModelsetVersionOptions() {
  const ms = getSelectedModelset();
  const versions = ms?.versions || [];
  if (!versions.length) {
    return `<option value="">(none)</option>`;
  }
  return versions
    .map((v) => {
      const vid = v.version_id;
      const selected = vid === modelsetVersionSelectId ? "selected" : "";
      const note = v.note || v.notes ? ` - ${v.note || v.notes}` : "";
      return `<option value="${vid}" ${selected}>${vid}${note}</option>`;
    })
    .join("");
}

function syncModelsetEditor() {
  const selected = getSelectedModelset();
  if (!selected) {
    editModelsetName = "";
    editModelsetDescription = "";
    editModelsetTags = "";
    return;
  }
  editModelsetName = selected.name || "";
  editModelsetDescription = selected.description || "";
  editModelsetTags = (selected.tags || []).join(", ");
}

function render(state) {
  if (!rootEl) return;
  lastState = state;

  const status = trainingStatus?.status || "idle";
  const phase = trainingStatus?.phase || "";
  const progress = Number(trainingStatus?.progress || 0);
  const progressPercent = Math.round(progress * 100);
  const isUseExisting = mode === "use-existing";
  const capabilities = state.capabilities || {};
  const hasModels = Boolean(capabilities.model);
  const hasRules = Boolean(capabilities.rules);
  const hasVector = Boolean(capabilities.vector);
  const activeModelsetId = state.active_modelset_id;
  const activeModelsetVersionId = state.active_modelset_version_id;
  const selectedModelsetVersion = getSelectedModelsetVersion();
  const selectedDecisionPolicy = selectedModelsetVersion?.decision_policy || null;
  const healthStats = datasetHealth || trainingStatus?.stats || null;
  const hasActiveModelset = Boolean(activeModelsetId);
  const insightsModelsetId = modelsetSelectId || activeModelsetId;
  const insightsVersionId = modelsetVersionSelectId || activeModelsetVersionId;
  const insightsAvailable = Boolean(trainingStatus?.status === "completed" && insightsModelsetId && insightsVersionId);
  const trainingTotalRows = Number(summary?.total_rows || healthStats?.total_rows || 0);
  const trainingMissingLabels = Number(summary?.missing_labels || 0);
  const labeledRows = Number(healthStats?.labeled_rows ?? Math.max(trainingTotalRows - trainingMissingLabels, 0));
  const hasTrainingData = trainingTotalRows > 0;
  const trainingDataError = hasTrainingData && labeledRows === 0;
  const blockingErrors = healthStats?.blocking_errors || [];
  const warnings = healthStats?.warnings || [];
  const hasBlockingErrors = blockingErrors.length > 0;
  const step1Complete = Boolean(modelsetSelectId || activeModelsetId);
  const step2Complete = hasTrainingData && labeledRows > 0;
  const step4Complete = trainingStatus?.status === "completed";
  const trainingInProgress = trainingStatus?.status === "running";
  const step3Skipped = step4Complete && !rulesModified;
  const canTrain = step1Complete && step2Complete && !trainingDataError && !hasBlockingErrors;
  const canProceedAfterTrain = step4Complete;
  const step5Complete = validationCompleted;
  const headlineLevelF1 = Number(trainingStatus?.metrics?.level?.macro_f1 || 0).toFixed(2);
  const headlineDeptF1 = Number(trainingStatus?.metrics?.dept?.macro_f1 || 0).toFixed(2);
  const validationTotalRows = Number(validationSummary?.total_rows || 0);
  const validationMissingLabels = Number(validationSummary?.missing_labels || 0);
  const validationLabeledRows = Math.max(validationTotalRows - validationMissingLabels, 0);
  const validationReady = validationTotalRows > 0 && validationLabeledRows > 0;
  const canValidate = step1Complete && step2Complete && step4Complete;
  const canValidatePathA = hasActiveModelset && validationReady && hasModels;
  const readinessLabels = {
    activeModelset: activeModelsetId ? `Active ModelSet: ${safeString(activeModelsetId)}` : "Active ModelSet: (none)",
    models: `Models: ${hasModels ? "✅" : "⭕"}`,
    rules: `Rules: ${hasRules ? "✅" : "⭕"}`,
    vector: `Vector store: ${hasVector ? "✅" : "⭕"}`,
    training: `Training data loaded: ${step2Complete ? `✅ (${labeledRows})` : "⭕"}`,
  };
  const stepStatus = (index, { complete = false, error = false, skipped = false } = {}) => {
    if (error) return "error";
    if (complete) return "completed";
    if (skipped) return "skipped";
    const firstIncomplete =
      !step1Complete
        ? 1
        : !step2Complete || trainingDataError || hasBlockingErrors
        ? 2
        : !step4Complete && !trainingInProgress
        ? 3
        : trainingInProgress
        ? 4
        : !step5Complete
        ? 5
        : 6;
    return index === firstIncomplete ? "active" : "inactive";
  };
  rootEl.innerHTML = `
    <h2>Train</h2>
    <div class="readiness-strip">
      <span>${readinessLabels.activeModelset} ${activeModelsetId ? "✅" : "⭕"}</span>
      <span>${readinessLabels.models}</span>
      <span>${readinessLabels.rules}</span>
      <span>${readinessLabels.vector}</span>
      <span>${readinessLabels.training}</span>
    </div>
    <p>Load a training file, configure rules, train models, and build a vector store.</p>
    <div class="card quickstart-card">
      <div class="quickstart-header">
        <h3>What do you want to do?</h3>
        <p class="muted">Choose a workflow to guide which sections appear below.</p>
      </div>
      <div class="quickstart-options">
        <button class="quickstart-option ${isUseExisting ? "active" : ""}" data-mode="use-existing" type="button">
          <span class="quickstart-title">Use an existing ModelSet and classify</span>
          <span class="quickstart-desc">Load a saved .sgm (or pick a local ModelSet version) and go straight to Classify.</span>
          ${isUseExisting ? `<span class="quickstart-chip">Selected</span>` : ""}
        </button>
        <button class="quickstart-option ${!isUseExisting ? "active" : ""}" data-mode="build-update" type="button">
          <span class="quickstart-title">Build / update a ModelSet (advanced)</span>
          <span class="quickstart-desc">Load labeled training data, train models, validate, build vector store, then save a new version.</span>
          ${!isUseExisting ? `<span class="quickstart-chip">Selected</span>` : ""}
        </button>
      </div>
    </div>
    ${isUseExisting ? `
    <div data-mode-section="use-existing">
      <div class="card modelset-card">
      <div class="modelset-header">
        <h3>Step 1 — Load a ModelSet</h3>
        <p class="modelset-description">
          Load a saved ModelSet to classify immediately. You can also import a <code>.sgm</code> bundle.
        </p>
        <p class="muted">Training data is only needed if you want to validate models, not to classify with an existing ModelSet.</p>
      </div>

      <div class="modelset-tabs">
        <button class="modelset-tab ${modelsetLoadTab === "local" ? "active" : ""}" data-modelset-tab="local" type="button">
          Local ModelSets
        </button>
        <button class="modelset-tab ${modelsetLoadTab === "import" ? "active" : ""}" data-modelset-tab="import" type="button">
          Import .sgm
        </button>
      </div>

      <div class="${modelsetLoadTab === "local" ? "" : "mode-hidden"}">
        <div class="modelset-grid">
          <label class="field modelset-field">
            <span>Existing ModelSet</span>
            <select id="modelset-select" ${modelsetsLoading ? "disabled" : ""}>
              ${renderModelsetSelectOptions()}
            </select>
          </label>
          <label class="field modelset-field">
            <span>Version</span>
            <select id="modelset-version-select" ${modelsetsLoading ? "disabled" : ""}>
              ${renderModelsetVersionOptions()}
            </select>
          </label>
          <div class="field modelset-field">
            <span>Active in app</span>
            <div class="chip-row">
              <span class="chip">${activeModelsetId ? activeModelsetId : "(none)"}</span>
              <span class="chip chip-muted">${activeModelsetVersionId ? activeModelsetVersionId : "(none)"}</span>
            </div>
          </div>
        </div>

        <div class="modelset-actions">
          <button id="modelset-refresh" ${modelsetsLoading ? "disabled" : ""}>${modelsetsLoading ? "Refreshing..." : "Refresh"}</button>
          <button id="modelset-load" ${modelsetSelectId && modelsetVersionSelectId ? "" : "disabled"}>Load selected version</button>
          <button id="modelset-export" ${modelsetSelectId && modelsetVersionSelectId ? "" : "disabled"}>Export .sgm</button>
        </div>

        <div class="modelset-policy">
          ${renderDecisionPolicy(selectedDecisionPolicy)}
        </div>
      </div>

      <div class="${modelsetLoadTab === "import" ? "" : "mode-hidden"}">
        <label class="field">
          <span>Choose .sgm file</span>
          <input type="file" id="modelset-import-file" accept=".sgm" />
          ${importFile ? `<small>Selected: ${importFile.name}</small>` : ""}
        </label>
        <div class="modelset-actions">
          <button id="modelset-import" ${importFile ? "" : "disabled"}>Import & load .sgm</button>
        </div>
      </div>

      ${hasActiveModelset
        ? `
        <div class="modelset-ready">
          <div class="modelset-banner">✅ Active ModelSet: ${safeString(activeModelsetId)} / ${safeString(activeModelsetVersionId || "(latest)")}</div>
          <div class="modelset-summary">
            <span>Models (level/dept): ${hasModels ? "✅" : "⭕"}</span>
            <span>Rules: ${hasRules ? "✅" : "⭕"}</span>
            <span>Vector store: ${hasVector ? "✅" : "⭕"}</span>
          </div>
        </div>
        `
        : `<p class="muted">No active ModelSet loaded yet.</p>`}
    </div>

    ${hasActiveModelset
      ? `
      <div class="card next-step-card">
        <h3>Next step</h3>
        ${hasModels
          ? `
          <p>You’re ready to classify using the active ModelSet. Training data is not required for this workflow.</p>
          <button id="go-classify" class="primary">Go to Classify</button>
          `
          : hasRules
          ? `
          <p>Only rules are available in this ModelSet. You can still classify using rules.</p>
          <button id="go-classify" class="primary">Go to Classify (rules-only)</button>
          `
          : `
          <p>This ModelSet has no trained models. You can still use rules (if present), or switch to the Build/Update workflow to train models.</p>
          <button id="switch-build-update">Switch to Build / update ModelSet</button>
          `}
      </div>
      `
      : ""}

    <div class="card validation-accordion">
      <button class="accordion-toggle" id="validation-toggle" type="button">
        Optional — Validate this ModelSet
      </button>
      ${validationAccordionOpen
        ? `
        <div class="accordion-body">
          <p class="muted">To validate this ModelSet, load a labeled dataset. Validation does not change the ModelSet; it only computes metrics.</p>
          <label class="field">
            <span>Validation dataset</span>
            <input type="file" id="validation-file" accept=".csv,.xlsx,.xls" />
            ${validationFile ? `<small>Selected: ${validationFile.name}</small>` : ""}
          </label>
          <button id="validation-load" ${validationFile ? "" : "disabled"}>Load validation dataset</button>
          ${validationSummary
            ? `
            <div class="card-grid">
              <div class="card">
                <h4>Summary</h4>
                <ul class="stats">
                  <li>Total rows: ${validationSummary.total_rows}</li>
                  <li>Labeled rows: ${validationLabeledRows}</li>
                  <li>Missing risk text: ${validationSummary.missing_risk_text}</li>
                  <li>Missing labels: ${validationSummary.missing_labels}</li>
                  <li>Invalid levels: ${validationSummary.invalid_levels}</li>
                  <li>Invalid departments: ${validationSummary.invalid_departments}</li>
                </ul>
              </div>
              <div class="card">
                <h4>Preview</h4>
                ${validationPreview.length ? renderPreview(validationPreview) : `<p>No preview available.</p>`}
              </div>
            </div>
            `
            : `<p class="muted">No validation dataset loaded (optional). Load one to compute metrics for this ModelSet.</p>`}
          <div class="rule-actions">
            <button id="validation-sanity-run" ${validationSanityLoading || !canValidatePathA ? "disabled" : ""}>
              ${validationSanityLoading ? "Running..." : "Run sanity check (fast)"}
            </button>
            <button id="validation-evaluate-run" ${validationEvaluationLoading || !canValidatePathA ? "disabled" : ""}>
              ${validationEvaluationLoading ? "Running..." : "Run holdout evaluation"}
            </button>
          </div>
          ${validationError ? `<div class="train-error">${safeString(validationError)}</div>` : ""}
          <div class="validation-results">
            <div class="card">
              <h4>Sanity check</h4>
              ${validationSanityReport ? renderSanityReport(validationSanityReport) : `<p class="muted">No sanity report yet.</p>`}
            </div>
            <div class="card">
              <h4>Holdout evaluation</h4>
              ${validationEvaluationReport ? renderEvaluateReport(validationEvaluationReport) : `<p class="muted">No evaluation report yet.</p>`}
            </div>
          </div>
        </div>
        `
        : ""}
    </div>
    </div>
    ` : ""}
    ${isUseExisting ? "" : `
    <div data-mode-section="build-update">
      ${step1Complete ? "" : `<p class="muted">No active ModelSet. Load one to classify, or switch to the Build/Update workflow to create one.</p>`}
      <div class="stepper">
        <button class="stepper-step ${stepStatus(1, { complete: step1Complete })}" data-step-target="step-modelset">
          <span class="stepper-index">1</span>
          <span>ModelSet</span>
        </button>
        <button class="stepper-step ${stepStatus(2, { complete: step2Complete, error: trainingDataError })}" data-step-target="step-training-data">
          <span class="stepper-index">2</span>
          <span>Training data</span>
        </button>
        <button class="stepper-step ${stepStatus(3, { complete: rulesModified, skipped: step3Skipped })}" data-step-target="step-rules">
          <span class="stepper-index">3</span>
          <span>Rules</span>
        </button>
        <button class="stepper-step ${stepStatus(4, { complete: step4Complete })}" data-step-target="step-train">
          <span class="stepper-index">4</span>
          <span>Train</span>
        </button>
        <button class="stepper-step ${stepStatus(5, { complete: step5Complete, error: Boolean(validationErrorMessage) })}" data-step-target="step-validate">
          <span class="stepper-index">5</span>
          <span>Validate</span>
        </button>
        <button class="stepper-step ${stepStatus(6)}" data-step-target="step-vector">
          <span class="stepper-index">6</span>
          <span>Vector store</span>
        </button>
        <button class="stepper-step ${stepStatus(7)}" data-step-target="step-save">
          <span class="stepper-index">7</span>
          <span>Save</span>
        </button>
      </div>

      <div class="card" id="step-modelset">
        <h3>Step 1 — Select or create ModelSet</h3>
        <p class="muted">Choose which ModelSet you want to train and save new versions into.</p>
        <div class="modelset-grid">
          <label class="field modelset-field">
            <span>Existing ModelSet</span>
            <select id="modelset-select" ${modelsetsLoading ? "disabled" : ""}>
              ${renderModelsetSelectOptions()}
            </select>
          </label>
          <label class="field modelset-field">
            <span>Target version (optional)</span>
            <select id="modelset-version-select" ${modelsetsLoading ? "disabled" : ""}>
              ${renderModelsetVersionOptions()}
            </select>
          </label>
        </div>
        <div class="modelset-actions">
          <button id="modelset-refresh" ${modelsetsLoading ? "disabled" : ""}>${modelsetsLoading ? "Refreshing..." : "Refresh"}</button>
        </div>
        <div class="card-grid modelset-subgrid">
          <div class="card card-inset modelset-inset">
            <h4>Create new ModelSet</h4>
            <div class="param-grid modelset-param-grid">
              <label class="field">
                <span>ModelSet ID (optional)</span>
                <input type="text" id="modelset-new-id" placeholder="e.g. customer_a_risk_v1" value="${safeString(newModelsetId)}" />
              </label>
              <label class="field">
                <span>Name</span>
                <input type="text" id="modelset-new-name" placeholder="e.g. Customer A – Risk" value="${safeString(newModelsetName)}" />
              </label>
              <label class="field">
                <span>Description</span>
                <input type="text" id="modelset-new-desc" placeholder="optional" value="${safeString(newModelsetDescription)}" />
              </label>
            </div>
            <div class="rule-actions">
              <button id="modelset-create">Create</button>
            </div>
          </div>
        </div>
        ${step1Complete ? `<p class="step-status success">✅ ModelSet selected.</p>` : `<p class="step-status">Select or create a ModelSet to continue.</p>`}
      </div>

      <div class="card" id="step-training-data">
        <h3>Step 2 — Load labeled training data</h3>
        <p class="muted">Load labeled examples so the system can learn to predict risk level and department.</p>
        <label class="field">
          <span>Training file (uses native OS picker)</span>
          <input type="file" id="train-file" accept=".csv,.xlsx,.xls" />
          ${trainingFile ? `<small>Selected: ${trainingFile.name}</small>` : ""}
        </label>
        <button id="train-load" ${loading ? "disabled" : ""}>${loading ? "Loading..." : "Load training data"}</button>
        <div class="card-grid">
          <div class="card">
            <h4>Summary</h4>
            ${summary ? `
              <ul class="stats">
                <li>Total rows: ${summary.total_rows}</li>
                <li>Labeled rows: ${labeledRows}</li>
                <li>Missing risk text: ${summary.missing_risk_text}</li>
                <li>Missing labels: ${summary.missing_labels}</li>
                <li>Invalid levels: ${summary.invalid_levels}</li>
                <li>Invalid departments: ${summary.invalid_departments}</li>
              </ul>
            ` : `<p>No dataset loaded.</p>`}
          </div>
          <div class="card">
            <h4>Preview</h4>
            ${preview.length ? renderPreview(preview) : `<p>No preview available.</p>`}
          </div>
        </div>
        <div class="card">
          <h4>Dataset health</h4>
          ${healthStats
            ? `
              <div class="card-grid">
                ${renderDistributionTable(
                  "Risk level distribution",
                  healthStats.label_distribution?.level || {},
                  healthStats.label_distribution_pct?.level || {}
                )}
                ${renderDistributionTable(
                  "Department distribution",
                  healthStats.label_distribution?.dept || {},
                  healthStats.label_distribution_pct?.dept || {}
                )}
              </div>
              ${warnings.length ? `
                <div class="train-warning">
                  <strong>Warnings</strong>
                  <ul>
                    ${warnings.map((warning) => `<li>${safeString(warning)}</li>`).join("")}
                  </ul>
                </div>
              ` : `<p class="muted">No imbalance warnings detected.</p>`}
              ${hasBlockingErrors ? `
                <div class="train-error">
                  <strong>Blocking issues</strong>
                  <ul>
                    ${blockingErrors.map((error) => `<li>${safeString(error)}</li>`).join("")}
                  </ul>
                </div>
              ` : ""}
            `
            : `<p class="muted">Load a training dataset to view health checks.</p>`}
        </div>
        <div class="card">
          <h4>Label definitions</h4>
          ${renderLabelPolicy(labelPolicy)}
        </div>
        ${trainingDataError
          ? `
        <div class="train-error">
            We found data rows but no labels in columns F and G starting at row 5. Please ensure the file includes labeled rows with valid risk level and department values.
          </div>
          `
          : hasBlockingErrors
          ? `
        <div class="train-error">
            Resolve the blocking dataset issues above before training.
          </div>
          `
          : step2Complete
          ? `<p class="step-status success">✅ Labeled training data ready.</p>`
          : `<p class="step-status">Load labeled rows to continue.</p>`}
      </div>

      <div class="card" id="step-rules">
        <h3>Step 3 — Rules (optional)</h3>
        <p class="muted">Define rule-based overrides (e.g., keywords) that complement the machine learning model.</p>
        <p class="step-status">${rulesModified ? "✅ Rules status: Modified" : "Rules status: Default rules loaded"}</p>
        <div class="rule-actions">
          <button id="rules-edit-toggle">${rulesEditorOpen ? "Close editor" : "Edit rules"}</button>
          <button id="rules-test-toggle">${rulesEditorOpen ? "Test a phrase" : "Test a phrase"}</button>
        </div>
        ${rulesEditorOpen
          ? `
          <div class="rules-editor">
            <div class="field">
              <textarea id="rules-json" rows="10" class="code-area">${safeString(rulesDraft || "")}</textarea>
            </div>
            ${rulesErrorMessage ? `<div class="train-error">${safeString(rulesErrorMessage)}</div>` : ""}
            <div class="rule-actions">
              <button id="rules-save">Save rules</button>
              <button id="rules-revert" class="ghost">Revert</button>
            </div>
            <div class="rule-actions">
              <input type="text" id="rule-test-text" placeholder="Enter risk text to test" />
              <button id="rules-test">Test rule</button>
            </div>
            <div class="rule-result">${renderRuleResult()}</div>
          </div>
          `
          : ""}
      </div>

      <div class="card" id="step-train">
        <h3>Step 4 — Train models</h3>
        <p class="muted">Train ML models for risk level and department using the loaded dataset.</p>
        <div class="train-actions">
        <button id="train-start" ${status === "running" || !canTrain ? "disabled" : ""}>Train models</button>
          <button id="train-cancel" ${status === "running" ? "" : "disabled"}>Cancel</button>
          <span class="chip">Status: ${status}</span>
          ${phase ? `<span class="chip chip-muted">Phase: ${phase}</span>` : ""}
          <span class="chip chip-muted">${progressPercent}%</span>
        </div>
        <button id="train-advanced-toggle" class="ghost">${trainingAdvancedOpen ? "Hide advanced settings" : "Show advanced settings"}</button>
        ${trainingAdvancedOpen
          ? `
          <div class="advanced-settings">
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
              <label class="field">
                <span>CV folds</span>
                <select id="cv-folds">
                  <option value="3" ${trainingParams.cv_folds === 3 ? "selected" : ""}>3</option>
                  <option value="5" ${trainingParams.cv_folds === 5 ? "selected" : ""}>5</option>
                  <option value="10" ${trainingParams.cv_folds === 10 ? "selected" : ""}>10</option>
                </select>
              </label>
              <label class="field">
                <span>Class weight balanced</span>
                <input type="checkbox" id="class-weight-balanced" ${trainingParams.use_class_weight_balanced ? "checked" : ""} />
              </label>
            </div>
          </div>
          ${renderImbalanceDistributions(
            trainingStatus?.stats?.preprocess_distribution,
            trainingStatus?.stats?.training_distribution
          )}
          `
          : ""}
        <div class="progress-row">
          <progress class="progress" max="1" value="${progress}"></progress>
          <div class="train-status">
            ${trainingStatus?.message ? `<div><strong>${safeString(trainingStatus.message)}</strong></div>` : ""}
            <div class="muted">
              ${trainingStatus?.started_at ? `Started: ${formatTs(trainingStatus.started_at)} (${formatSecondsSince(trainingStatus.started_at)} elapsed)` : ""}
              ${trainingStatus?.finished_at ? ` | Finished: ${formatTs(trainingStatus.finished_at)}` : ""}
              ${trainingStatus?.last_updated_at ? ` | Last update: ${formatTs(trainingStatus.last_updated_at)}` : ""}
            </div>
          </div>
        </div>
        ${trainingStatus?.error ? `<div class="train-error">Error: ${safeString(trainingStatus.error)}</div>` : ""}
        ${renderTrainingDetails(trainingStatus)}
        ${renderCvSummary(trainingStatus?.cv_metrics, trainingStatus?.cv_status)}
        ${trainingMetrics ? renderMetrics(trainingMetrics) : ""}
        ${step4Complete ? `<p class="step-status success">✅ Training complete. Level macro F1 ${headlineLevelF1} | Dept macro F1 ${headlineDeptF1}. Next: Validate.</p>` : ""}
      </div>

      ${renderInsightsPanel({
        available: insightsAvailable,
        type: insightsType,
        className: insightsClass,
        topN: insightsTopN,
        data: insightsData,
        loading: insightsLoading,
        errorMessage: insightsError,
      })}

      <div class="card" id="step-validate">
        <h3>Step 5 — Validate</h3>
        <p class="muted">Check how the models perform on your labeled data.</p>
        ${canValidate
          ? ""
          : `<p class="muted">Load labeled training data in Step 2 and train models in Step 4 to enable validation.</p>`}
        <div class="rule-actions">
          <button id="sanity-run" ${sanityLoading || !canValidate ? "disabled" : ""}>
            ${sanityLoading ? "Running..." : "Run sanity check (fast)"}
          </button>
          <button id="evaluate-run" ${evaluationLoading || !canValidate ? "disabled" : ""}>
            ${evaluationLoading ? "Running..." : "Run holdout evaluation (slower)"}
          </button>
        </div>
        ${validationErrorMessage ? `<div class="train-error">${safeString(validationErrorMessage)}</div>` : ""}
        <div class="validation-results">
          <div class="card">
            <h4>Sanity check</h4>
            ${sanityReport ? renderSanityReport(sanityReport) : `<p class="muted">No sanity report yet.</p>`}
          </div>
          <div class="card">
            <h4>Holdout evaluation</h4>
            ${evaluationReport ? renderEvaluateReport(evaluationReport) : `<p class="muted">No evaluation report yet.</p>`}
          </div>
        </div>
      </div>

      <div class="card placeholder-card" id="step-vector">
        <h3>Step 6 — Vector store</h3>
        <p class="muted">Build a similarity index so new text can be compared to known examples.</p>
        <button disabled>Build vector store</button>
      </div>

      <div class="card placeholder-card" id="step-save">
        <h3>Step 7 — Save snapshot / Export</h3>
        <p class="muted">Save everything to a new ModelSet version you can reuse in the Classify tab.</p>
        <button disabled>Save snapshot</button>
      </div>
    </div>
    `}
  `;

  const modeOptions = Array.from(rootEl.querySelectorAll("[data-mode]"));
  modeOptions.forEach((button) => {
    button.addEventListener("click", () => {
      const nextMode = button.dataset.mode;
      if (!nextMode || nextMode === mode) return;
      mode = nextMode;
      render(state);
    });
  });

  const tabOptions = Array.from(rootEl.querySelectorAll("[data-modelset-tab]"));
  tabOptions.forEach((button) => {
    button.addEventListener("click", () => {
      const nextTab = button.dataset.modelsetTab;
      if (!nextTab || nextTab === modelsetLoadTab) return;
      modelsetLoadTab = nextTab;
      render(state);
    });
  });

  const goClassifyBtn = rootEl.querySelector("#go-classify");
  if (goClassifyBtn) {
    goClassifyBtn.addEventListener("click", async () => {
      const navButton = document.querySelector('[data-pane="classify"]');
      if (navButton instanceof HTMLElement) {
        navButton.click();
        return;
      }
      try {
        await setActivePane("classify");
        const nextState = await syncState();
        render(nextState || state);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  const switchBuildBtn = rootEl.querySelector("#switch-build-update");
  if (switchBuildBtn) {
    switchBuildBtn.addEventListener("click", () => {
      mode = "build-update";
      render(state);
    });
  }

  const validationToggle = rootEl.querySelector("#validation-toggle");
  if (validationToggle) {
    validationToggle.addEventListener("click", () => {
      validationAccordionOpen = !validationAccordionOpen;
      render(state);
    });
  }

  const stepperButtons = Array.from(rootEl.querySelectorAll(".stepper-step"));
  stepperButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.dataset.stepTarget;
      if (!targetId) return;
      const target = rootEl.querySelector(`#${targetId}`);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  const loadBtn = rootEl.querySelector("#train-load");
  const fileInput = rootEl.querySelector("#train-file");
  const validationFileInput = rootEl.querySelector("#validation-file");
  const validationLoadBtn = rootEl.querySelector("#validation-load");
  const validationSanityBtn = rootEl.querySelector("#validation-sanity-run");
  const validationEvaluateBtn = rootEl.querySelector("#validation-evaluate-run");
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
        await refreshDatasetHealth();
        sanityReport = null;
        evaluationReport = null;
        validationCompleted = false;
        validationErrorMessage = "";
      } catch (error) {
        alert(error.message);
      } finally {
        loading = false;
        trainingFile = null;
        render(state);
      }
    });
  }

  if (validationFileInput) {
    validationFileInput.addEventListener("change", () => {
      validationFile = validationFileInput.files?.[0] || null;
      render(state);
    });
  }

  if (validationLoadBtn) {
    validationLoadBtn.addEventListener("click", async () => {
      if (!validationFile) {
        alert("Please select a validation file.");
        return;
      }
      try {
        validationSummary = await loadDataset("train", validationFile);
        const previewResp = await fetchPreview("train", 10, 0);
        validationPreview = previewResp.rows || [];
        validationFile = null;
        validationError = "";
        validationSanityReport = null;
        validationEvaluationReport = null;
        render(state);
      } catch (error) {
        validationError = error.message;
        render(state);
      }
    });
  }

  const oversampleCheckbox = rootEl.querySelector("#oversample-enabled");
  const oversampleCapInput = rootEl.querySelector("#oversample-cap");
  const minRecallInput = rootEl.querySelector("#min-recall");
  const calibrationSelect = rootEl.querySelector("#calibration-method");
  const cvFoldsSelect = rootEl.querySelector("#cv-folds");
  const classWeightCheckbox = rootEl.querySelector("#class-weight-balanced");
  const startBtn = rootEl.querySelector("#train-start");
  const cancelBtn = rootEl.querySelector("#train-cancel");
  const vectorBtn = rootEl.querySelector("#vector-build");
  const sanityBtn = rootEl.querySelector("#sanity-run");
  const evaluateBtn = rootEl.querySelector("#evaluate-run");
  const insightsTypeSelect = rootEl.querySelector("#insights-type");
  const insightsClassSelect = rootEl.querySelector("#insights-class");
  const insightsTopNInput = rootEl.querySelector("#insights-top-n");
  const insightsRefreshBtn = rootEl.querySelector("#insights-refresh");

  const syncParams = () => {
    trainingParams.oversample_enabled = oversampleCheckbox?.checked || false;
    trainingParams.oversample_cap_ratio = parseFloat(oversampleCapInput?.value || trainingParams.oversample_cap_ratio);
    trainingParams.min_recall_per_class = parseFloat(minRecallInput?.value || trainingParams.min_recall_per_class);
    trainingParams.calibration_method = calibrationSelect?.value || trainingParams.calibration_method;
    trainingParams.cv_folds = parseInt(cvFoldsSelect?.value || trainingParams.cv_folds, 10);
    trainingParams.use_class_weight_balanced = classWeightCheckbox?.checked ?? trainingParams.use_class_weight_balanced;
  };

  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      try {
        syncParams();
        await startTraining(trainingParams);
        trainingStatus = await fetchTrainingStatus();
        trainingMetrics = trainingStatus?.metrics || null;
        if (trainingStatus?.status === "running") {
          startTrainingPoll();
          // Update immediately (don't wait for the first interval tick)
          await pollTrainingStatus();
        } else {
          stopTrainingPoll();
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
      trainingMetrics = trainingStatus?.metrics || null;
      stopTrainingPoll();
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

  if (sanityBtn) {
    sanityBtn.addEventListener("click", async () => {
      sanityLoading = true;
      validationErrorMessage = "";
      render(state);
      try {
        sanityReport = await runSanityCheck();
        validationCompleted = true;
      } catch (error) {
        validationErrorMessage = error.message;
      } finally {
        sanityLoading = false;
        render(state);
      }
    });
  }

  if (evaluateBtn) {
    evaluateBtn.addEventListener("click", async () => {
      evaluationLoading = true;
      validationErrorMessage = "";
      render(state);
      try {
        evaluationReport = await runEvaluate();
        validationCompleted = true;
      } catch (error) {
        validationErrorMessage = error.message;
      } finally {
        evaluationLoading = false;
        render(state);
      }
    });
  }

  if (insightsTypeSelect) {
    insightsTypeSelect.addEventListener("change", () => {
      insightsType = insightsTypeSelect.value;
      insightsClass = "";
      render(state);
    });
  }

  if (insightsClassSelect) {
    insightsClassSelect.addEventListener("change", () => {
      insightsClass = insightsClassSelect.value;
    });
  }

  if (insightsTopNInput) {
    insightsTopNInput.addEventListener("change", () => {
      insightsTopN = Number(insightsTopNInput.value || insightsTopN);
    });
  }

  if (insightsRefreshBtn) {
    insightsRefreshBtn.addEventListener("click", async () => {
      const modelsetId = modelsetSelectId || activeModelsetId;
      const versionId = modelsetVersionSelectId || activeModelsetVersionId;
      if (!modelsetId || !versionId) {
        insightsError = "Select a ModelSet version to load insights.";
        render(state);
        return;
      }
      insightsLoading = true;
      insightsError = "";
      render(state);
      try {
        const data = await fetchModelInsights(modelsetId, versionId, insightsTopN);
        insightsData = data;
        const labels = data?.insights?.[insightsType]?.labels || [];
        if (labels.length && !labels.includes(insightsClass)) {
          insightsClass = labels[0];
        }
      } catch (error) {
        insightsError = error.message;
      } finally {
        insightsLoading = false;
        render(state);
      }
    });
  }

  if (validationSanityBtn) {
    validationSanityBtn.addEventListener("click", async () => {
      validationSanityLoading = true;
      validationError = "";
      render(state);
      try {
        validationSanityReport = await runSanityCheck();
      } catch (error) {
        validationError = error.message;
      } finally {
        validationSanityLoading = false;
        render(state);
      }
    });
  }

  if (validationEvaluateBtn) {
    validationEvaluateBtn.addEventListener("click", async () => {
      validationEvaluationLoading = true;
      validationError = "";
      render(state);
      try {
        validationEvaluationReport = await runEvaluate();
      } catch (error) {
        validationError = error.message;
      } finally {
        validationEvaluationLoading = false;
        render(state);
      }
    });
  }

  const saveBtn = rootEl.querySelector("#rules-save");
  const rulesTextArea = rootEl.querySelector("#rules-json");
  const testBtn = rootEl.querySelector("#rules-test");
  const testInput = rootEl.querySelector("#rule-test-text");
  const rulesEditToggle = rootEl.querySelector("#rules-edit-toggle");
  const rulesTestToggle = rootEl.querySelector("#rules-test-toggle");
  const rulesRevertBtn = rootEl.querySelector("#rules-revert");
  const trainAdvancedToggle = rootEl.querySelector("#train-advanced-toggle");

  if (rulesEditToggle) {
    rulesEditToggle.addEventListener("click", () => {
      rulesEditorOpen = !rulesEditorOpen;
      rulesFocusTest = false;
      if (rulesEditorOpen) {
        rulesDraft = rulesConfig ? JSON.stringify(rulesConfig, null, 2) : "";
        rulesDirty = false;
        rulesErrorMessage = "";
      }
      render(state);
    });
  }

  if (rulesTestToggle) {
    rulesTestToggle.addEventListener("click", () => {
      rulesEditorOpen = true;
      rulesFocusTest = true;
      rulesDraft = rulesConfig ? JSON.stringify(rulesConfig, null, 2) : "";
      rulesDirty = false;
      rulesErrorMessage = "";
      render(state);
    });
  }

  if (saveBtn && rulesTextArea) {
    saveBtn.addEventListener("click", async () => {
      try {
        const parsed = JSON.parse(rulesTextArea.value);
        rulesConfig = await saveRules(parsed);
        rulesModified = true;
        rulesDirty = false;
        rulesErrorMessage = "";
        rulesDraft = JSON.stringify(rulesConfig, null, 2);
        alert("Rules saved");
        render(state);
      } catch (error) {
        rulesErrorMessage = error.message;
        render(state);
      }
    });
  }

  if (rulesTextArea) {
    rulesTextArea.addEventListener("input", () => {
      rulesDraft = rulesTextArea.value;
      rulesDirty = true;
    });
  }

  if (rulesRevertBtn) {
    rulesRevertBtn.addEventListener("click", () => {
      rulesDraft = rulesConfig ? JSON.stringify(rulesConfig, null, 2) : "";
      rulesDirty = false;
      rulesErrorMessage = "";
      render(state);
    });
  }

  if (testBtn && testInput) {
    testBtn.addEventListener("click", async () => {
      try {
        const parsed = rulesTextArea?.value ? JSON.parse(rulesTextArea.value) : rulesConfig;
        testResult = await testRules(testInput.value || "", parsed);
        render(state);
      } catch (error) {
        rulesErrorMessage = error.message;
        render(state);
      }
    });
  }

  if (rulesFocusTest && testInput) {
    testInput.focus();
    rulesFocusTest = false;
  }

  if (trainAdvancedToggle) {
    trainAdvancedToggle.addEventListener("click", () => {
      trainingAdvancedOpen = !trainingAdvancedOpen;
      render(state);
    });
  }

  // --- ModelSets ---

  const msSelect = rootEl.querySelector("#modelset-select");
  const msVersionSelect = rootEl.querySelector("#modelset-version-select");
  const msRefreshBtn = rootEl.querySelector("#modelset-refresh");
  const msSaveBtn = rootEl.querySelector("#modelset-save");
  const msLoadBtn = rootEl.querySelector("#modelset-load");
  const msExportBtn = rootEl.querySelector("#modelset-export");
  const msDeleteVersionBtn = rootEl.querySelector("#modelset-delete-version");
  const msDeleteBtn = rootEl.querySelector("#modelset-delete");
  const msNewId = rootEl.querySelector("#modelset-new-id");
  const msNewName = rootEl.querySelector("#modelset-new-name");
  const msNewDesc = rootEl.querySelector("#modelset-new-desc");
  const msEditName = rootEl.querySelector("#modelset-edit-name");
  const msEditDesc = rootEl.querySelector("#modelset-edit-desc");
  const msEditTags = rootEl.querySelector("#modelset-edit-tags");
  const msUpdateBtn = rootEl.querySelector("#modelset-update");
  const msCreateBtn = rootEl.querySelector("#modelset-create");
  const msImportFile = rootEl.querySelector("#modelset-import-file");
  const msImportBtn = rootEl.querySelector("#modelset-import");

  if (msSelect) {
    msSelect.addEventListener("change", async () => {
      modelsetSelectId = msSelect.value || "";
      modelsetVersionSelectId = "";
      syncModelsetEditor();
      render(lastState);
    });
  }

  if (msVersionSelect) {
    msVersionSelect.addEventListener("change", async () => {
      modelsetVersionSelectId = msVersionSelect.value || "";
      render(lastState);
    });
  }

  if (msRefreshBtn) {
    msRefreshBtn.addEventListener("click", async () => {
      await refreshModelSets();
      render(lastState);
    });
  }

  if (msSaveBtn) {
    msSaveBtn.addEventListener("click", async () => {
      if (!modelsetSelectId) return;
      const notes = window.prompt("Optional notes for this version:", "") || "";
      try {
        await saveModelSetVersion(modelsetSelectId, {
          notes,
          parent_version_id: modelsetVersionSelectId || null,
          include_bundle: true,
          include_vector_store: true,
          include_rules: true,
          include_training_snapshot: true,
        });
        await refreshModelSets();
        render(lastState);
        alert("ModelSet snapshot saved.");
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (msLoadBtn) {
    msLoadBtn.addEventListener("click", async () => {
      if (!modelsetSelectId || !modelsetVersionSelectId) return;
      try {
        await loadModelSet(modelsetSelectId, modelsetVersionSelectId);
        await refreshLabelPolicy();
        try {
          rulesConfig = await fetchRules();
        } catch (error) {
          console.error("Failed to fetch rules", error);
          rulesConfig = null;
        }
        rulesEditorOpen = false;
        rulesDraft = rulesConfig ? JSON.stringify(rulesConfig, null, 2) : "";
        rulesDirty = false;
        rulesModified = false;
        rulesErrorMessage = "";
        const loadedState = await syncState();
        const dataset = loadedState?.training_dataset || null;
        summary = dataset?.summary || null;
        updateVectorBuildStatusFromState(loadedState);
        if (Array.isArray(dataset?.rows)) {
          preview = dataset.rows.slice(0, 10);
        } else {
          preview = [];
        }
        render(lastState);
        alert("ModelSet loaded.");
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (msExportBtn) {
    msExportBtn.addEventListener("click", async () => {
      if (!modelsetSelectId || !modelsetVersionSelectId) return;
      const url = exportModelSetUrl(modelsetSelectId, modelsetVersionSelectId);

      const safePart = (s) => String(s || "").replace(/[^A-Za-z0-9._-]+/g, "_");
      const suggestedName = `${safePart(modelsetSelectId)}__${safePart(modelsetVersionSelectId)}.sgm`;

      // Best-effort Save-As prompt (supported in Chromium via File System Access API).
      // Falls back to a normal browser download when unavailable.
      if (typeof window.showSaveFilePicker === "function") {
        try {
          const res = await fetch(url);
          if (!res.ok) {
            throw new Error(`Export failed: ${res.status} ${res.statusText}`);
          }
          const blob = await res.blob();

          const handle = await window.showSaveFilePicker({
            suggestedName,
            types: [
              {
                description: "SpecGrader ModelSet",
                accept: {
                  "application/octet-stream": [".sgm"],
                  "application/zip": [".sgm"]
                }
              }
            ]
          });

          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
          alert("Export saved.");
          return;
        } catch (error) {
          // If the user cancels the dialog or the API fails, fall back to a normal download.
          // A user-cancel usually shows as an AbortError.
          console.warn("Save picker failed; falling back to download:", error);
        }
      }

      // Fallback: normal download (browser will save to default downloads folder unless configured to ask).
      const a = document.createElement("a");
      a.href = url;
      a.download = suggestedName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });
  }

  if (msDeleteVersionBtn) {
    msDeleteVersionBtn.addEventListener("click", async () => {
      if (!modelsetSelectId || !modelsetVersionSelectId) return;
      const isActive =
        lastState?.active_modelset_id === modelsetSelectId &&
        lastState?.active_modelset_version_id === modelsetVersionSelectId;
      if (!window.confirm(`Delete version '${modelsetVersionSelectId}'? This cannot be undone.`)) {
        return;
      }
      let force = false;
      if (isActive) {
        force = window.confirm("This version is active. Force delete and clear active version?");
        if (!force) return;
      }
      try {
        await deleteModelSetVersion(modelsetSelectId, modelsetVersionSelectId, force);
        modelsetVersionSelectId = "";
        await refreshModelSets();
        await syncState();
        render(lastState);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (msDeleteBtn) {
    msDeleteBtn.addEventListener("click", async () => {
      if (!modelsetSelectId) return;
      const ok = window.confirm(`Delete ModelSet '${modelsetSelectId}'? This cannot be undone.`);
      if (!ok) return;
      try {
        await deleteModelSet(modelsetSelectId);
        modelsetSelectId = "";
        modelsetVersionSelectId = "";
        await refreshModelSets();
        await syncState();
        render(lastState);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (msNewId) {
    msNewId.addEventListener("input", () => {
      newModelsetId = msNewId.value || "";
    });
  }
  if (msNewName) {
    msNewName.addEventListener("input", () => {
      newModelsetName = msNewName.value || "";
    });
  }
  if (msNewDesc) {
    msNewDesc.addEventListener("input", () => {
      newModelsetDescription = msNewDesc.value || "";
    });
  }

  if (msEditName) {
    msEditName.addEventListener("input", () => {
      editModelsetName = msEditName.value || "";
    });
  }

  if (msEditDesc) {
    msEditDesc.addEventListener("input", () => {
      editModelsetDescription = msEditDesc.value || "";
    });
  }

  if (msEditTags) {
    msEditTags.addEventListener("input", () => {
      editModelsetTags = msEditTags.value || "";
    });
  }

  if (msUpdateBtn) {
    msUpdateBtn.addEventListener("click", async () => {
      if (!modelsetSelectId) return;
      const name = (editModelsetName || "").trim();
      const description = (editModelsetDescription || "").trim();
      const tags = (editModelsetTags || "")
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
      if (!name) {
        alert("ModelSet name cannot be empty.");
        return;
      }
      try {
        await updateModelSet(modelsetSelectId, { name, description, tags });
        await refreshModelSets();
        render(lastState);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (msCreateBtn) {
    msCreateBtn.addEventListener("click", async () => {
      try {
        const id = (newModelsetId || "").trim() || null;
        const name = (newModelsetName || "").trim() || null;
        const desc = (newModelsetDescription || "").trim() || "";
        const resp = await createModelSet(id, name, desc);
        const created = resp.modelset?.modelset_id || id || "";
        await refreshModelSets();
        if (created) {
          modelsetSelectId = created;
          modelsetVersionSelectId = "";
        }
        newModelsetId = "";
        newModelsetName = "";
        newModelsetDescription = "";
        render(lastState);
      } catch (error) {
        alert(error.message);
      }
    });
  }

  if (msImportFile) {
    msImportFile.addEventListener("change", () => {
      importFile = msImportFile.files?.[0] || null;
      render(lastState);
    });
  }

  if (msImportBtn) {
    msImportBtn.addEventListener("click", async () => {
      if (!importFile) return;
      try {
        const resp = await importModelSet(importFile);
        importFile = null;
        await refreshModelSets();
        if (resp?.modelset_id) {
          modelsetSelectId = resp.modelset_id;
          modelsetVersionSelectId = resp.version_id || "";
        }
        try {
          rulesConfig = await fetchRules();
        } catch (error) {
          console.error("Failed to fetch rules", error);
          rulesConfig = null;
        }
        rulesEditorOpen = false;
        rulesDraft = rulesConfig ? JSON.stringify(rulesConfig, null, 2) : "";
        rulesDirty = false;
        rulesModified = false;
        rulesErrorMessage = "";
        render(lastState);
        alert("Imported ModelSet.");
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
  return `
    <div class="metrics">
      ${renderMetricBlock("Risk level", metrics.level)}
      ${renderMetricBlock("Department", metrics.dept)}
    </div>
  `;
}

function renderSanityReport(report) {
  if (!report.available) {
    return `<p class="muted">Sanity report unavailable. Train a model and load labeled data.</p>`;
  }
  const accuracyLevel = Number(report.accuracy_level || 0).toFixed(2);
  const accuracyDept = Number(report.accuracy_dept || 0).toFixed(2);
  const mismatches = Array.isArray(report.rows)
    ? report.rows.filter((row) => !row.match_level || !row.match_dept)
    : [];
  return `
    <div class="metrics">
      <div>
        <h4>Accuracy</h4>
        <p>Level: ${accuracyLevel} | Department: ${accuracyDept}</p>
      </div>
      <div>
        <h4>Mismatches</h4>
        <p>${mismatches.length} row(s)</p>
      </div>
    </div>
    ${mismatches.length ? renderSanityRows(mismatches) : `<p class="muted">No mismatches detected.</p>`}
  `;
}

function renderSanityRows(rows) {
  const header = `
    <div class="table-header">
      <span>Row</span>
      <span>Text</span>
      <span>Expected level</span>
      <span>Pred level</span>
      <span>Expected dept</span>
      <span>Pred dept</span>
    </div>`;
  const body = rows
    .map(
      (row) => `
        <div class="table-row">
          <span>${row.row_id}</span>
          <span>${row.risk_text || ""}</span>
          <span>${row.expected_level || ""}</span>
          <span>${row.pred_level || ""}</span>
          <span>${row.expected_dept || ""}</span>
          <span>${row.pred_dept || ""}</span>
        </div>`
    )
    .join("");
  return `<div class="table-placeholder">${header}${body}</div>`;
}

function renderEvaluateReport(report) {
  if (!report.available) {
    const error = report.error ? `<div class="muted">${report.error}</div>` : "";
    return `<p class="muted">Evaluation unavailable. Train a model and load labeled data.</p>${error}`;
  }
  const warnings = Array.isArray(report.warnings) && report.warnings.length
    ? `<ul class="event-log">${report.warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`
    : "";
  const metrics = report.metrics || {};
  return `
    <div class="metrics">
      ${renderMetricBlock("Model", metrics.model)}
      ${renderMetricBlock("Rules", metrics.rules)}
      ${renderMetricBlock("Vector (k=1)", metrics.vector?.k_1)}
      ${renderMetricBlock("Ensemble", metrics.ensemble)}
    </div>
    ${warnings}
  `;
}

function renderMetricBlock(title, metrics) {
  if (!metrics) {
    return `<div><h4>${title}</h4><p class="muted">No data</p></div>`;
  }
  const macroF1 = Number(metrics.macro_f1 || 0).toFixed(2);
  const weightedF1 = Number(metrics.weighted_f1 || 0).toFixed(2);
  const balancedAcc = Number(metrics.balanced_accuracy || 0).toFixed(2);
  const labels = metrics.labels || Object.keys(metrics.per_class_recall || {});
  const perClassPrecision = metrics.per_class_precision || {};
  const perClassRecall = metrics.per_class_recall || {};
  const perClassF1 = metrics.per_class_f1 || {};
  const minRecall = Number(trainingParams?.min_recall_per_class ?? 0.5);
  return `
    <div>
      <h4>${title}</h4>
      <div class="metrics-summary">
        <div><strong>Macro F1</strong> ${macroF1}</div>
        <div><strong>Weighted F1</strong> ${weightedF1}</div>
        <div><strong>Balanced acc</strong> ${balancedAcc}</div>
      </div>
      ${renderPerClassTable(labels, perClassPrecision, perClassRecall, perClassF1, minRecall)}
      ${renderConfusionMatrix(metrics.confusion_matrix, labels)}
    </div>
  `;
}

function renderPerClassTable(labels, precision, recall, f1, minRecall) {
  if (!labels.length) {
    return `<p class="muted">No per-class metrics available.</p>`;
  }
  return `
    <table class="table metrics-table">
      <thead>
        <tr>
          <th>Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        ${labels
          .map((label) => {
            const recallValue = Number(recall?.[label] ?? 0);
            const warnRecall = ["high", "extreme"].includes(label) && recallValue < minRecall;
            const recallClass = warnRecall ? "metric-warning" : "";
            return `
              <tr>
                <td>${safeString(label)}</td>
                <td>${Number(precision?.[label] ?? 0).toFixed(2)}</td>
                <td class="${recallClass}">${recallValue.toFixed(2)}</td>
                <td>${Number(f1?.[label] ?? 0).toFixed(2)}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderConfusionMatrix(matrix, labels) {
  if (!Array.isArray(matrix) || !matrix.length) {
    return `<p class="muted">No confusion matrix available.</p>`;
  }
  const header = labels.map((label) => `<th>${safeString(label)}</th>`).join("");
  const body = matrix
    .map((row, rowIndex) => {
      const cells = row.map((value) => `<td>${value}</td>`).join("");
      const rowLabel = labels[rowIndex] ?? "";
      return `<tr><th>${safeString(rowLabel)}</th>${cells}</tr>`;
    })
    .join("");
  return `
    <div class="matrix">
      <table>
        <thead>
          <tr>
            <th></th>
            ${header}
          </tr>
        </thead>
        <tbody>
          ${body}
        </tbody>
      </table>
    </div>
  `;
}

function renderInsightsPanel({
  available,
  type,
  className,
  topN,
  data,
  loading,
  errorMessage,
}) {
  if (!available) {
    return `
      <div class="card placeholder-card">
        <h3>Model insights</h3>
        <p class="muted">Train and save a ModelSet version to view top TF-IDF terms.</p>
      </div>
    `;
  }

  const insights = data?.insights?.[type] || null;
  const labels = insights?.labels || [];
  const selectedClass = className || labels[0] || "";
  const terms = insights?.top_terms?.[selectedClass] || [];
  const metadata = insights?.metadata || {};
  return `
    <div class="card" id="step-insights">
      <h3>Model insights</h3>
      <p class="muted">Explore the top TF-IDF terms influencing each class prediction.</p>
      <div class="param-grid">
        <label class="field">
          <span>Label type</span>
          <select id="insights-type">
            <option value="level" ${type === "level" ? "selected" : ""}>Risk</option>
            <option value="dept" ${type === "dept" ? "selected" : ""}>Department</option>
          </select>
        </label>
        <label class="field">
          <span>Class</span>
          <select id="insights-class">
            ${labels.map((label) => `<option value="${label}" ${label === selectedClass ? "selected" : ""}>${label}</option>`).join("")}
          </select>
        </label>
        <label class="field">
          <span>Top N</span>
          <input type="number" id="insights-top-n" min="5" max="50" value="${topN}" />
        </label>
      </div>
      <div class="rule-actions">
        <button id="insights-refresh" ${loading ? "disabled" : ""}>${loading ? "Loading..." : "Load insights"}</button>
      </div>
      ${errorMessage ? `<div class="train-error">${safeString(errorMessage)}</div>` : ""}
      ${insights
        ? `
          <p class="muted">Vectorizer: ngram ${safeString(metadata.ngram_range)} | max features ${safeString(metadata.max_features)}</p>
          ${terms.length
            ? `
              <table class="table metrics-table">
                <thead>
                  <tr>
                    <th>Term</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  ${terms.map((term) => `<tr><td>${safeString(term.term)}</td><td>${Number(term.weight).toFixed(3)}</td></tr>`).join("")}
                </tbody>
              </table>
            `
            : `<p class="muted">No positive-weight terms found for this class.</p>`}
        `
        : `<p class="muted">Load a trained ModelSet version to view insights.</p>`}
    </div>
  `;
}

export default {
  async mount(containerEl, ctx = {}) {
    storeRef = ctx.store || null;
    rootEl = document.createElement("section");
    rootEl.className = "pane";
    containerEl.innerHTML = "";
    containerEl.appendChild(rootEl);
    try {
      rulesConfig = await fetchRules();
      rulesDraft = rulesConfig ? JSON.stringify(rulesConfig, null, 2) : "";
      rulesModified = false;
      rulesDirty = false;
      rulesErrorMessage = "";
    } catch (error) {
      console.error("Failed to fetch rules", error);
      rulesConfig = null;
      rulesDraft = "";
    }

    try {
      trainingStatus = await fetchTrainingStatus();
      trainingMetrics = trainingStatus?.metrics || null;
      if (trainingStatus?.status === "running") {
        startTrainingPoll();
        await pollTrainingStatus();
      }
      if (trainingStatus?.stats) {
        datasetHealth = trainingStatus.stats;
      } else {
        await refreshDatasetHealth();
      }
    } catch (error) {
      console.error("Failed to fetch training status", error);
      trainingStatus = null;
      trainingMetrics = null;
    }

    try {
      await refreshModelSets();
    } catch (error) {
      console.error("Failed to fetch modelsets", error);
    }
    await refreshLabelPolicy();

    if (storeRef && typeof storeRef.getState === "function") {
      lastState = storeRef.getState();
    }
    render(lastState || { active_modelset_id: null, active_modelset_version_id: null });
  },
  refresh(state) {
    render(state);
  },
  unmount() {
    stopTrainingPoll();
    if (rootEl && rootEl.parentElement) {
      rootEl.parentElement.removeChild(rootEl);
    }
    rootEl = null;
    summary = null;
    preview = [];
    loading = false;
    trainingFile = null;
    datasetHealth = null;
    testResult = null;
    rulesConfig = null;
    rulesEditorOpen = false;
    rulesDraft = "";
    rulesDirty = false;
    rulesModified = false;
    rulesErrorMessage = "";
    rulesFocusTest = false;
    trainingAdvancedOpen = false;
    trainingStatus = null;
    trainingMetrics = null;
    insightsData = null;
    insightsLoading = false;
    insightsError = "";
    insightsType = "level";
    insightsClass = "";
    insightsTopN = 20;
    vectorBuildStatus = null;
    sanityReport = null;
    sanityLoading = false;
    evaluationReport = null;
    evaluationLoading = false;
    validationErrorMessage = "";
    validationCompleted = false;
    validationFile = null;
    validationSummary = null;
    validationPreview = [];
    validationSanityReport = null;
    validationEvaluationReport = null;
    validationSanityLoading = false;
    validationEvaluationLoading = false;
    validationAccordionOpen = false;
    validationError = "";
    labelPolicy = null;
    labelPolicyError = "";
    labelPolicyLoading = false;
  },
};
