# Phase 2 — Comprehensive ModelSet CRUD (families + immutable versions)

## Objective
Implement and/or complete **full CRUD** for ModelSets using best-practice semantics:
- ModelSet **family** (stable ID, name, metadata)
- ModelSet **versions** (immutable snapshots containing models + vector store + rules + manifest/snapshots)

## Scope
Backend + frontend wiring for:
- Create modelset family
- Read/list modelsets and versions
- Update modelset family metadata (rename/notes/tags)
- Update “behavior” by creating a **new version** (Save Snapshot)
- Delete modelset family (and/or delete individual versions with guardrails)

## Primary files expected to change
Backend:
- `backend/app/main.py`
- `backend/app/state.py`
- `backend/app/services/modelset_service.py`

Frontend:
- `frontend/src/api/client.js`
- `frontend/src/panes/trainPane.js`
- (optional) `frontend/src/state/store.js`

Tests:
- `tests/test_modelset_service.py` (extend)
- add `tests/test_modelset_routes.py` (recommended)

## Required behavior details
1. **ModelSet family metadata**
   - Fields: `modelset_id`, `name`, `description`, `tags`, `created_at`, `updated_at`
2. **Version**
   - Fields: `version_id`, `created_at`, `parent_version_id` (optional), `notes` (optional)
   - Must include artifact folders/files:
     - `bundle/` (trained model artifacts)
     - `vector_store/` (vector artifacts)
     - `rules.json`
     - `training_snapshot.json` (if available)
     - `manifest.json` (or `version.json`) capturing counts + config references
3. **Guards**
   - Prevent deleting the **active** version unless user first activates another version (or uses an explicit force parameter).
   - Prevent overwriting an existing version_id during import/save.
4. **Empty states**
   - Listing works even if no modelsets exist.
   - Loading a modelset without vectors/rules/metrics gives “available:false” structured payloads (no 404 spam).

## Implementation steps
1. Backend: ensure a clear storage structure:
   - `workspace/modelsets/<modelset_id>/modelset.json`
   - `workspace/modelsets/<modelset_id>/versions/<version_id>/...`
2. Backend: add/confirm routes:
   - `POST /api/modelsets` (create family)
   - `GET /api/modelsets` (list families + versions)
   - `PATCH /api/modelsets/{modelset_id}` (metadata update)
   - `DELETE /api/modelsets/{modelset_id}` (delete family)
   - `POST /api/modelsets/{modelset_id}/versions` (create version snapshot from current staging artifacts)
   - `DELETE /api/modelsets/{modelset_id}/versions/{version_id}` (delete version)
   - `POST /api/modelsets/{modelset_id}/load` (activate version)
   - Ensure responses include enough data for the UI to update without extra guessing.
3. Frontend: update UI so it can:
   - Create modelset family
   - Rename/update metadata (at minimum: name + notes)
   - Delete modelset (with a confirm UI or explicit “type ID to confirm” pattern)
   - Save Snapshot creates a new version and refreshes list
4. State synchronization:
   - After load, call a single “active state” fetch or ensure `load` response includes active snapshot data.
5. Add/extend tests for:
   - Create/list/update/delete family
   - Save/load/delete versions
   - Guard: cannot delete active version without force

## Tests that must pass
### Automated (required)
- `pytest -q`

Recommended test expectations:
- Creating a modelset returns 200 and the family appears in list.
- Saving a version creates files in the version folder.
- Loading sets `active_modelset_id` and `active_version_id`.
- Deleting active version fails with clear error unless `force=true`.
- Deleting family removes it from list.

### Manual UI (required)
- Create modelset family
- Save snapshot (version appears)
- Load version (active marker updates; panes hydrate)
- Update metadata (name/notes)
- Delete a non-active version (removed)
- Delete the modelset family (removed)

## Success checklist
- [ ] ModelSet families are creatable and listable
- [ ] Metadata update works (PATCH)
- [ ] Versions are immutable snapshots and can be created repeatedly
- [ ] Load activates a specific version and UI reflects active state
- [ ] Delete version and delete family work with guardrails
- [ ] No console error spam; “not available” returns structured 200 responses
- [ ] `pytest -q` passes
