# Phase 01 — Repo Hygiene, Dependency Fixes, and “models/” Safety

## Agents
- Run: **AGENT_01_DEPENDENCY_MANAGER**
- Optional assist: **AGENT_00_REPO_AUDITOR**

## Goal
Make the repo installable and robust on a fresh machine:
- Fix `requirements.txt` to match actual GUI (PySide6)
- Ensure `models/` directory is created automatically when needed
- Ensure `models/model_sets.json` can be created without crashing

## Files to Modify
- `requirements.txt`
- `logic.py` (function `ensure_model_db()`)
- `model_set_manager.py` (defensive: create `models/` directory)
- Optional: `README.md` (installation section)

## Files to Add
- `models/.gitkeep` (so the folder exists in git)
- `artifacts/.gitkeep` (optional)

## Implementation Steps
1. Update `requirements.txt`
   - Remove the Tkinter guidance (or move to comments) because this UI is **PySide6**.
   - Add required GUI dependency:
     - `PySide6`
   - Prepare for vector DB:
     - add `chromadb` (this will be used in Phase 02)
   - Keep: `pandas`, `numpy`, `joblib`, `scikit-learn`, `sentence-transformers`, `openpyxl`

2. Make `models/` creation robust
   - In `logic.py` → update `ensure_model_db()`:
     - `os.makedirs("models", exist_ok=True)` must run before trying to open/write `models/model_sets.json`

3. Harden `model_set_manager.py`
   - Before reading/writing `models/model_sets.json`, ensure:
     - `os.makedirs("models", exist_ok=True)`

4. Add placeholder directories
   - Add `models/.gitkeep`
   - Add `artifacts/.gitkeep` (optional but recommended)

## Must-Pass Tests (no errors)
- `pip install -r requirements.txt`
- `python -m compileall .`
- Delete any existing local `models/` folder (temporarily) and then run:
  - `python -c "import logic; logic.ensure_model_db(); print('OK')"`
  - It must recreate `models/` and `models/model_sets.json` without crashing.
- `python splash.py` (UI opens)

## Success Checklist
- [ ] `requirements.txt` includes `PySide6` and `chromadb`
- [ ] Repo installs cleanly with `pip install -r requirements.txt`
- [ ] `logic.ensure_model_db()` creates `models/` and `models/model_sets.json` on a clean checkout
- [ ] UI still starts with `python splash.py`
