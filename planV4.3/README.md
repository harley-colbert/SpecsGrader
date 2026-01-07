# SpecsGrader PlanV4.3

This folder contains a phase-by-phase execution plan to upgrade SpecsGrader from **v4.2.2** to **v4.3.x**.

## Core goals for v4.3
1. **Complete Phase 2 UX work**: move the **Model Set (.sgm)** section to the top of the Train pane and make it more compact.
2. **Comprehensive, best-practice ModelSet CRUD**:
   - ModelSet “families” + immutable **versions**
   - Create / Read / Update (metadata + new version) / Delete
   - Clear “active modelset/version” behavior and robust empty states
3. **Robust .sgm import/export**:
   - Safe extraction (zip-slip protection)
   - Integrity (checksums)
   - Better UX for export (save picker when supported; fallback otherwise)
4. **Regression safety**:
   - Loading a modelset version hydrates Training/Vector/Rules
   - Train / Build Vector / Grade flow still works
   - No console error spam (no 404s during normal flows)

## Ground rules for executing phases
- Execute phases in order.
- Each phase contains:
  - **Implementation steps**
  - **Tests that must pass**
  - **Success checklist**
- If tests fail, fix code and re-run until passing.
- Keep diffs scoped to the phase objective.

## Standard commands
From repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

python run.py
```

## Testing conventions
- Backend unit tests: `pytest -q`
- “No errors” means:
  - Server starts without tracebacks
  - Browser console has no repeated errors during normal use
  - API returns 200 for “not available yet” states (no 404 spam)
