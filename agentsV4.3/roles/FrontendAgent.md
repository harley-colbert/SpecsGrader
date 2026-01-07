# FrontendAgent — Operating procedure

## General
- Make UI changes in `trainPane.js` first, then add targeted CSS in `styles.css`.
- Use small helper functions inside the pane rather than sprawling logic.
- Treat backend responses as the source of truth for active modelset/version state.

## Required UX principles
- ModelSet load must hydrate Training/Vector/Rules.
- Never throw on "not available"; render a clear empty state.
- Keep export/import buttons responsive; show errors to user if a request fails.

## Manual validation
- Verify on a clean reload.
- Verify narrow window widths.
- Verify export and import.
