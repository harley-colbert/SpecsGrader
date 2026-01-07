import {
  fetchState,
  fetchPreview,
  loadDataset,
  fetchRules,
  saveRules,
  testRules,
  startTraining,
  fetchTrainingStatus,
  cancelTraining,
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
} from "../api/client.js";

let rootEl = null;
let lastState = null;
let storeRef = null;
let summary = null;
let preview = [];
let loading = false;
let trainingFile = null;
let rulesConfig = null;
let testResult = null;
let trainingStatus = null;
let trainingMetrics = null;
let vectorBuildStatus = null;
let trainingPollTimer = null;

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
let trainingParams = {
  oversample_enabled: false,
  oversample_cap_ratio: 0.3,
  min_recall_per_class: 0.5,
  calibration_method: "sigmoid",
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

function safeString(value) {
  return value === null || value === undefined ? "" : String(value);
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

function renderMiniJson(obj) {
  try {
    return `<pre class="mini-code">${safeString(JSON.stringify(obj, null, 2))}</pre>`;
  } catch (_) {
    return `<pre class="mini-code">(Unable to format)</pre>`;
  }
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
  rootEl.innerHTML = `
    <h2>Train</h2>
    <p>Load a training file, configure rules, train models, and build a vector store.</p>
    <div class="card modelset-card">
      <div class="modelset-header">
        <h3>Model sets (.sgm)</h3>
        <p class="modelset-description">
          Save / load named, versioned ModelSets that bundle trained models, the vector store, and rules.
          Export / import as a single <code>.sgm</code> file.
        </p>
      </div>

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
            <span class="chip">${state.active_modelset_id ? state.active_modelset_id : "(none)"}</span>
            <span class="chip chip-muted">${state.active_modelset_version_id ? state.active_modelset_version_id : "(none)"}</span>
          </div>
        </div>
      </div>

      <div class="modelset-actions">
        <button id="modelset-refresh" ${modelsetsLoading ? "disabled" : ""}>${modelsetsLoading ? "Refreshing..." : "Refresh"}</button>
        <button id="modelset-save" ${modelsetSelectId ? "" : "disabled"}>Save snapshot (new version)</button>
        <button id="modelset-load" ${modelsetSelectId && modelsetVersionSelectId ? "" : "disabled"}>Load selected version</button>
        <button id="modelset-export" ${modelsetSelectId && modelsetVersionSelectId ? "" : "disabled"}>Export .sgm</button>
        <button id="modelset-import" ${importFile ? "" : "disabled"}>Import</button>
        <button id="modelset-delete-version" class="danger" ${modelsetSelectId && modelsetVersionSelectId ? "" : "disabled"}>Delete version</button>
        <button id="modelset-delete" class="danger" ${modelsetSelectId ? "" : "disabled"}>Delete ModelSet</button>
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
        <div class="card card-inset modelset-inset">
          <h4>Edit ModelSet metadata</h4>
          <div class="param-grid modelset-param-grid">
            <label class="field">
              <span>Name</span>
              <input type="text" id="modelset-edit-name" placeholder="ModelSet name" value="${safeString(editModelsetName)}" />
            </label>
            <label class="field">
              <span>Description</span>
              <input type="text" id="modelset-edit-desc" placeholder="optional" value="${safeString(editModelsetDescription)}" />
            </label>
            <label class="field">
              <span>Tags (comma-separated)</span>
              <input type="text" id="modelset-edit-tags" placeholder="e.g. customer, v4" value="${safeString(editModelsetTags)}" />
            </label>
          </div>
          <div class="rule-actions">
            <button id="modelset-update" ${modelsetSelectId ? "" : "disabled"}>Update metadata</button>
          </div>
        </div>
        <div class="card card-inset modelset-inset">
          <h4>Import .sgm</h4>
          <label class="field">
            <span>Choose .sgm file</span>
            <input type="file" id="modelset-import-file" accept=".sgm" />
            ${importFile ? `<small>Selected: ${importFile.name}</small>` : ""}
          </label>
          <small class="muted">Use Import in the action row above.</small>
        </div>
      </div>
    </div>
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
      <div class="train-actions">
        <button id="train-start" ${status === "running" ? "disabled" : ""}>Train</button>
        <button id="train-cancel" ${status === "running" ? "" : "disabled"}>Cancel</button>
        <span class="chip">Status: ${status}</span>
        ${phase ? `<span class="chip chip-muted">Phase: ${phase}</span>` : ""}
        <span class="chip chip-muted">${progressPercent}%</span>
      </div>
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
        try {
          rulesConfig = await fetchRules();
        } catch (error) {
          console.error("Failed to fetch rules", error);
          rulesConfig = null;
        }
        await syncState();
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
  async mount(containerEl, ctx = {}) {
    storeRef = ctx.store || null;
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

    try {
      trainingStatus = await fetchTrainingStatus();
      trainingMetrics = trainingStatus?.metrics || null;
      if (trainingStatus?.status === "running") {
        startTrainingPoll();
        await pollTrainingStatus();
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
    testResult = null;
    trainingStatus = null;
    trainingMetrics = null;
    vectorBuildStatus = null;
  },
};
