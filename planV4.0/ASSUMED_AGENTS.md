# Assumed agents (referenced by the plan)

This plan assumes `agentsV4.0` provides the following agent roles and that Codex can invoke them:

- Agent_RepoAuditor: audits repo structure, enforces conventions, validates paths/ports/run.py requirements
- Agent_BackendEngineer: implements Python backend services and FastAPI routes + PyWebView bridge
- Agent_FrontendEngineer: implements HTML/CSS/ESM-JS panes, table UI, nav shell
- Agent_DataIngest: implements CSV/XLSX parsing rules (row 5, col E/F/G, “Standards Risk Matrix” tab)
- Agent_MLTrainer: implements TF-IDF baseline + class weights + calibration + metrics guardrails
- Agent_VectorStore: implements local embeddings, ANN index, kNN voting, vector classifier option
- Agent_LLMIntegrator: implements OpenRouter LLM calls, schema parsing, env-only API key, never-send guard
- Agent_Aggregator: implements weighted aggregation (vector → llm → rules), thresholds, confidence computation
- Agent_ResultsUX: implements editable results grid, overrides, corrections persistence, add-to-training workflow
- Agent_QA: owns pytest suite, golden fixtures, performance smoke, cancel/progress tests
- Agent_ReleaseManager: packaging, docs, reproducible run instructions, final sanity verification

The plan files will reference these agents as if they already exist.
