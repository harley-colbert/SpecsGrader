# AGENT_01_DEPENDENCY_MANAGER — Installability + Fresh-Install Robustness

## Role
Make the repo installable and robust on a fresh machine:
- correct `requirements.txt` to reflect the real GUI (PySide6)
- prepare for vector DB (ChromaDB)
- ensure `models/` and `models/model_sets.json` are created safely on first run

## Primary Phases
- Phase 01 (required)
- Supports Phase 00 when baseline install fails

## Inputs
- SpecGrader repo
- Phase instructions in `Phase_01_Hygiene_and_Dependencies.md`

## Outputs
- Updated `requirements.txt`
- Hardened `logic.ensure_model_db()` (creates `models/` before writing JSON)
- Hardened `model_set_manager.py` (creates `models/` before read/write)
- Added placeholder directories:
  - `models/.gitkeep`
  - `artifacts/.gitkeep` (optional but recommended)

## Implementation Steps (follow exactly)
1. Update `requirements.txt`
   - Ensure it includes:
     - `PySide6`
     - `chromadb`
     - `pandas`
     - `numpy`
     - `joblib`
     - `scikit-learn`
     - `sentence-transformers`
     - `openpyxl`
   - If Tkinter guidance exists, move it to comments or remove it (PySide6 UI is used).

2. Update `logic.py`
   - In `ensure_model_db()`:
     - add `os.makedirs("models", exist_ok=True)` before any file operations
     - ensure creation of `models/model_sets.json` still works as `{}` JSON

3. Update `model_set_manager.py`
   - Before reading or writing `models/model_sets.json`:
     - add `os.makedirs("models", exist_ok=True)`

4. Add placeholder folders/files
   - Create `models/.gitkeep`
   - Create `artifacts/.gitkeep`

## Must-Pass Tests (no errors)
Run these from repo root:
- `pip install -r requirements.txt`
- `python -m compileall .`

Fresh-install robustness test:
1) Temporarily move or delete the `models/` directory (if present).
2) Run:
   - `python -c "import logic; logic.ensure_model_db(); print('OK')"`
3) Confirm:
   - `models/` exists
   - `models/model_sets.json` exists and is valid JSON

UI test:
- `python splash.py` (must open without crashing)

## Success Checklist
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `python -m compileall .` succeeds
- [ ] `ensure_model_db()` recreates `models/` and `models/model_sets.json` on a clean checkout
- [ ] UI opens via `python splash.py`

## Guardrails
- Do not introduce any vector DB code in this phase beyond adding `chromadb` dependency.
- Do not change inference logic in Phase 01.
