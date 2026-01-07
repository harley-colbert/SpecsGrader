const defaultHeaders = {
  "Content-Type": "application/json",
};

async function request(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const mergedHeaders = options.headers
    ? { ...defaultHeaders, ...options.headers }
    : { ...defaultHeaders };
  if (isFormData) {
    delete mergedHeaders["Content-Type"];
  }

  const response = await fetch(path, {
    ...options,
    headers: mergedHeaders,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchState() {
  return request("/api/state");
}

export async function setActivePane(pane) {
  return request("/api/ui/set_active_pane", {
    method: "POST",
    body: JSON.stringify({ pane }),
  });
}

export async function loadDataset(mode, source) {
  if (source instanceof File) {
    const formData = new FormData();
    formData.append("mode", mode);
    formData.append("file", source);
    return request("/api/data/load", {
      method: "POST",
      body: formData,
    });
  }

  return request("/api/data/load", {
    method: "POST",
    body: JSON.stringify({ mode, path: source }),
  });
}

export async function fetchPreview(mode, limit = 20, offset = 0) {
  const params = new URLSearchParams({ mode, limit, offset });
  return request(`/api/data/preview?${params.toString()}`);
}

export async function fetchRules() {
  return request("/api/rules/get");
}

export async function saveRules(rules) {
  return request("/api/rules/set", {
    method: "POST",
    body: JSON.stringify({ rules }),
  });
}

export async function testRules(text, rules) {
  return request("/api/rules/test", {
    method: "POST",
    body: JSON.stringify({ text, rules }),
  });
}

export async function startTraining(params) {
  return request("/api/train/start", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function fetchTrainingStatus() {
  return request("/api/train/status");
}

export async function cancelTraining() {
  return request("/api/train/cancel", { method: "POST" });
}

export async function runSanityCheck() {
  return request("/api/train/sanity", { method: "POST" });
}

export async function runEvaluate(payload = {}) {
  return request("/api/train/evaluate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchTrainingMetrics() {
  const resp = await request("/api/train/metrics");
  return resp && resp.available ? resp.metrics : null;
}

export async function buildVectorStore(k = 5) {
  return request("/api/vector/build", {
    method: "POST",
    body: JSON.stringify({ k }),
  });
}

export async function testVector(text, k = 5) {
  return request("/api/vector/test", {
    method: "POST",
    body: JSON.stringify({ text, k }),
  });
}

export async function testLlm(text, model = "openrouter/auto") {
  return request("/api/llm/test", {
    method: "POST",
    body: JSON.stringify({ text, model }),
  });
}

export async function startClassify(payload = {}) {
  return request("/api/classify/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchClassifyStatus() {
  return request("/api/classify/status");
}

export async function cancelClassify() {
  return request("/api/classify/cancel", { method: "POST" });
}

export async function fetchResults(limit = 20, offset = 0) {
  const params = new URLSearchParams({ limit, offset });
  return request(`/api/results/rows?${params.toString()}`);
}

export async function fetchSettings() {
  return request("/api/settings");
}

export async function saveSettings(never_send_externally) {
  return request("/api/settings", {
    method: "POST",
    body: JSON.stringify({ never_send_externally }),
  });
}

// -----------------
// ModelSets
// -----------------

export async function listModelSets() {
  return request("/api/modelsets");
}

export async function createModelSet(modelset_id, name, description = "") {
  return request("/api/modelsets", {
    method: "POST",
    body: JSON.stringify({ modelset_id, name, description }),
  });
}

export async function updateModelSet(modelset_id, payload = {}) {
  return request(`/api/modelsets/${encodeURIComponent(modelset_id)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function deleteModelSet(modelset_id) {
  return request(`/api/modelsets/${encodeURIComponent(modelset_id)}`, {
    method: "DELETE",
  });
}

export async function saveModelSetVersion(modelset_id, payload = {}) {
  return request(`/api/modelsets/${encodeURIComponent(modelset_id)}/versions`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteModelSetVersion(modelset_id, version_id, force = false) {
  const params = new URLSearchParams();
  if (force) params.set("force", "true");
  const query = params.toString();
  return request(
    `/api/modelsets/${encodeURIComponent(modelset_id)}/versions/${encodeURIComponent(version_id)}${query ? `?${query}` : ""}`,
    {
      method: "DELETE",
    }
  );
}

export async function loadModelSet(modelset_id, version_id = null) {
  return request(`/api/modelsets/${encodeURIComponent(modelset_id)}/load`, {
    method: "POST",
    body: JSON.stringify({ version_id }),
  });
}

export function exportModelSetUrl(modelset_id, version_id = null) {
  const params = new URLSearchParams();
  if (version_id) params.set("version_id", version_id);
  const q = params.toString();
  return `/api/modelsets/${encodeURIComponent(modelset_id)}/export${q ? `?${q}` : ""}`;
}

export async function importModelSet(file) {
  if (!(file instanceof File)) {
    throw new Error("importModelSet expects a File");
  }
  const formData = new FormData();
  formData.append("file", file);
  return request("/api/modelsets/import", {
    method: "POST",
    body: formData,
  });
}
