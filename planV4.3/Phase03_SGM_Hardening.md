# Phase 3 — .sgm import/export hardening + integrity

## Objective
Harden `.sgm` (SpecGrader ModelSet) packaging to be safe, verifiable, and predictable:
- Safe zip extraction (prevent path traversal)
- Integrity checks (checksums)
- Clear manifest schema and versioning
- Robust import policies (no silent overwrites)

## Scope
Backend packaging + import/export endpoints and frontend UX wiring.

## Primary files expected to change
Backend:
- `backend/app/services/modelset_service.py` (or a dedicated `sgm_io.py`)
- `backend/app/main.py`

Frontend:
- `frontend/src/api/client.js`
- `frontend/src/panes/trainPane.js`

Tests:
- add `tests/test_sgm_io.py`

## Required behavior details
1. **Export**
   - Generates an `.sgm` file (zip container) containing:
     - `manifest.json`
     - `checksums.sha256`
     - `rules.json`
     - `bundle/` and `vector_store/` folders
     - optional telemetry files (training snapshot / metrics)
2. **Import**
   - Validates:
     - manifest format + format_version
     - checksum file if present
     - zip-slip safety rules
   - Extracts to a new version folder without overwriting existing versions.
   - Returns created/registered `modelset_id` + `version_id`.
3. **UX**
   - Export should attempt a save-location prompt using a save picker when supported (frontend), with fallback to a normal download.
   - Import should show a clear success message and refresh list.

## Implementation steps
1. Define `manifest.json` schema:
   - `format`, `format_version`, `modelset_id`, `version_id`, `created_at`, `app_version`
   - optional: `parent_version_id`, `tags`, `notes`, counts
2. Implement checksum write/verify:
   - sha256 of each included file path (relative to zip root)
3. Add zip-slip protection on import.
4. Add import policy:
   - if modelset/version exists, reject with a clear error (or generate a new version_id).
5. Extend UI behavior to:
   - show export/import errors clearly (no silent failures)

## Tests that must pass
### Automated (required)
- `pytest -q`

Specific tests:
- Export creates .sgm with required files.
- Import rejects zip-slip entries.
- Import rejects checksum mismatches.
- Import registers a new version and files exist on disk.

### Manual (required)
- Export a version to .sgm.
- Import it into a clean workspace and confirm it appears and can load.

## Success checklist
- [ ] Export always includes manifest and (optionally) checksums
- [ ] Import blocks unsafe zips and validates integrity
- [ ] Import never silently overwrites an existing version
- [ ] Export UX works (save picker when supported; fallback otherwise)
- [ ] `pytest -q` passes
