const NAV_ITEMS = [
  { pane: "train", label: "Train" },
  { pane: "classify", label: "Classify" },
  { pane: "results", label: "Results" },
];

export function createShell(rootEl, onPaneSelected) {
  rootEl.innerHTML = `
    <div class="app-shell">
      <aside class="nav">
        <div class="logo">SpecsGrader</div>
        <nav>
          ${NAV_ITEMS.map(
            (item) => `
              <button class="nav-item" data-pane-button data-pane="${item.pane}">
                ${item.label}
              </button>
            `
          ).join("")}
        </nav>
      </aside>
      <main class="content" data-content></main>
    </div>
  `;

  const contentEl = rootEl.querySelector("[data-content]");
  const navButtons = Array.from(rootEl.querySelectorAll("[data-pane-button]"));

  navButtons.forEach((button) => {
    button.addEventListener("click", () => {
      onPaneSelected(button.dataset.pane);
    });
  });

  function setActiveNav(pane) {
    navButtons.forEach((button) => {
      if (button.dataset.pane === pane) {
        button.classList.add("active");
      } else {
        button.classList.remove("active");
      }
    });
  }

  return { contentEl, setActiveNav };
}
