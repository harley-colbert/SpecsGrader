const button = document.getElementById("health-button");
const result = document.getElementById("health-result");

async function checkHealth() {
  result.textContent = "Checking...";
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    result.textContent = JSON.stringify(data);
  } catch (error) {
    result.textContent = `Error: ${error.message}`;
  }
}

button?.addEventListener("click", checkHealth);
