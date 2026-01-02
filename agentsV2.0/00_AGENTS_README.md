# SpecGrader v2.0 — Agent Prompt Pack (agentsV2.0)

This zip contains **agent prompt files** referenced by `planV2.0.zip`.
Each file is intended to be used as a standalone “agent instruction sheet” in agent-mode.

## Agents included
- AGENT_00_REPO_AUDITOR
- AGENT_01_DEPENDENCY_MANAGER
- AGENT_02_VECTOR_DB_ENGINEER
- AGENT_03_TRAINING_PIPELINE_ENGINEER
- AGENT_04_INFERENCE_PIPELINE_ENGINEER
- AGENT_05_UI_ENGINEER
- AGENT_06_TEST_ENGINEER
- AGENT_07_RELEASE_MANAGER

## Operating rules (apply to all agents)
- Work **only** inside the provided SpecGrader repo working directory.
- Follow the phase instructions in `planV2.0.zip` exactly; do not skip “Must‑Pass Tests”.
- Prefer **small, reviewable commits** per phase.
- Preserve backward compatibility:
  - app must start on a clean machine (even with no models)
  - vector DB is optional and must soft-fail (warnings, not crashes)
  - legacy model sets (no vector keys) must still load using the pickle fallback
- Keep the existing verbose debug logging style unless the phase explicitly changes it.
- Never remove existing output columns; only add new ones (v2.0 requirement).

## Conventions
- Paths are relative to the repo root (the folder containing `app.py`, `logic.py`, `ui.py`, etc.).
- Use `python -m compileall .` frequently.
- Use `python -m unittest -v` after Phase 09.

