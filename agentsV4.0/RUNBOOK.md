# RUNBOOK — Daily Codex execution workflow

1) Create/activate venv
2) Install requirements
3) Run pytest
4) Run app
5) Execute the current phase checklist
6) Re-run tests
7) Only move to next phase when success checklist is complete

## Commands (root)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q
python run.py
```

## When a test fails
- Stop and fix the code until the test passes.
- Add regression coverage if a bug was found.
- Re-run `python -m pytest -q` before proceeding.
