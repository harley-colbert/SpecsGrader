import { fetchState, testVector, fetchRules, fetchSettings, saveSettings, testLlm } from "../api/client.js";

let rootEl = null;
let stateCache = null;
let vectorResult = null;
let neverSend = false;
let llmResult = null;

function renderVectorResult(result) {
  const neighbors = (result.neighbors || [])
    .map((n) => `<li>${n.row.risk_text} (${n.row.label_dept}/${n.row.label_level}) sim=${Number(n.similarity).toFixed(2)}</li>`)
    .join("");
  return `
    <div class="metrics">
      <div><strong>Dept:</strong> ${result.dept_pred ?? "abstain"} (${Number(result.dept_conf).toFixed(2)})</div>
      <div><strong>Level:</strong> ${result.level_pred ?? "abstain"} (${Number(result.level_conf).toFixed(2)})</div>
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
    </div>
  `;

  const neverSendBox = rootEl.querySelector("#never-send");
  neverSendBox?.addEventListener("change", async () => {
    try {
      neverSend = !!neverSendBox.checked;
      await saveSettings(neverSend);
      llmResult = null;
      render();
    } catch (error) {
      alert(error.message);
    }
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

export default {
  async mount(containerEl) {
    rootEl = document.createElement("section");
    rootEl.className = "pane";
    containerEl.innerHTML = "";
    containerEl.appendChild(rootEl);

    stateCache = await fetchState();
    const settings = await fetchSettings();
    neverSend = !!settings.never_send_externally;
    render();
  },
  refresh() {
    render();
  },
  unmount() {
    if (rootEl && rootEl.parentElement) {
      rootEl.parentElement.removeChild(rootEl);
    }
    rootEl = null;
    stateCache = null;
    vectorResult = null;
    llmResult = null;
  },
};
