const headers = {
  "Content-Type": "application/json",
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers,
    ...options,
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

export async function loadDataset(mode, path) {
  return request("/api/data/load", {
    method: "POST",
    body: JSON.stringify({ mode, path }),
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

export async function fetchTrainingMetrics() {
  return request("/api/train/metrics");
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
