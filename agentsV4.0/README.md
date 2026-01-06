# SpecsGrader — agentsV4.0

Date: 2026-01-06

This archive provides the **agent playbooks** referenced by `planV4.0`.

These agents are intended to be used by ChatGPT Codex (or an agentic runner) to execute
each plan phase deterministically. Each agent file includes:
- scope and responsibilities
- step-by-step procedure
- required outputs
- test expectations
- “definition of done” checklist

## How to use with planV4.0
1) Open `planV4.0/phases/Phase00_...md`
2) For that phase, invoke the listed agents to complete work items.
3) Run the required tests.
4) Only proceed when the success checklist is satisfied.

## Non-negotiable repo constraints (repeat)
- App runs from repo root: `python run.py`
- `requirements.txt` at repo root
- Backend + frontend served on the same local port (preferred)
- PyWebView opens that same local URL

## Shared contracts
See:
- `shared/CONTRACTS.md`
- `shared/ENUMS.md`
- `shared/BUNDLE_SPEC.md`
- `shared/METHOD_PREDICTION_SCHEMA.md`
