# BackendAgent — Operating procedure

## General
- Keep ModelSet versions immutable. 'Update' is creating a new version.
- Never use 404 for 'not ready' in normal UX flows; return 200 with structured payloads.
- Add guardrails around deletion and overwrites.

## .sgm requirements
- Zip-slip protection on import
- Integrity checks (checksums) on import/export
- Clear manifest schema with `format` and `format_version`
- No silent overwrite of existing `modelset_id/version_id` unless explicitly requested by policy

## Testing expectations
- Add route/service unit tests for CRUD and .sgm IO.
- Use temporary directories/fixtures to avoid touching developer state.
