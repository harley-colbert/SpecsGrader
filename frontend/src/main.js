import { fetchState, setActivePane } from "./api/client.js";
import { store } from "./state/store.js";
import { createShell } from "./ui/shell.js";
import classifyPane from "./panes/classifyPane.js";
import resultsPane from "./panes/resultsPane.js";
import trainPane from "./panes/trainPane.js";

const panes = {
  train: trainPane,
  classify: classifyPane,
  results: resultsPane,
};

let shell;
let mountedPane = null;
let mountedPaneKey = null;

function mountPane(paneKey, state) {
  if (paneKey === mountedPaneKey && mountedPane) {
    mountedPane.refresh(state);
    shell.setActiveNav(paneKey);
    return;
  }

  if (mountedPane && typeof mountedPane.unmount === "function") {
    mountedPane.unmount();
  }

  const nextPane = panes[paneKey];
  if (!nextPane) return;

  mountedPane = nextPane;
  mountedPaneKey = paneKey;
  mountedPane.mount(shell.contentEl, { store });
  mountedPane.refresh(state);
  shell.setActiveNav(paneKey);
}

async function syncAndRender(targetPane) {
  if (targetPane) {
    await setActivePane(targetPane);
  }
  const state = await fetchState();
  store.setState(state);
  mountPane(state.active_pane, state);
}

async function bootstrap() {
  const appRoot = document.getElementById("app");
  shell = createShell(appRoot, (pane) => syncAndRender(pane));
  await syncAndRender();
}

bootstrap().catch((error) => {
  console.error("Failed to bootstrap app", error);
  const appRoot = document.getElementById("app");
  if (appRoot) {
    appRoot.innerHTML = `<div class="error">${error.message}</div>`;
  }
});
