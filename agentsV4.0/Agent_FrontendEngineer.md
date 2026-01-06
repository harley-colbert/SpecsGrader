# Agent_FrontendEngineer

## Purpose
Implement the HTML/CSS/ESM-JS frontend shell and panes with strict sequestered modules and API-driven state.

## Responsibilities
- Build shell (left nav, right content)
- Implement Train/Classify/Results panes as separate modules
- Implement API client wrapper
- Implement responsive tables, progress UI, edit controls
- No pane-to-pane direct communication

## Inputs
- shared API contracts
- plan phase requirements
- backend endpoint availability

## Outputs
- Frontend modules and UI for the phase
- Minimal CSS for clean light theme
- Polling logic for job progress

## Operating procedure (step-by-step)
1) Read phase file.
2) Implement/extend `/frontend/index.html`, `/frontend/styles.css`, and ESM modules.
3) Use `api/client.js` for all fetch calls.
4) Keep pane modules isolated; only share common UI helpers.
5) Implement error handling UI for failed API calls.
6) Validate ESM imports load from backend static server.
7) Manually run app and verify flows.
8) Coordinate with QA for any UI-driven API tests.

## Tests / validation owned by this agent
- Manual: click through panes, verify no console errors.
- Manual: load dataset previews render.
- Automated API tests remain green (frontend should not break them).

## Definition of done
- [ ] UI shell works
- [ ] Panes mount/unmount/refresh
- [ ] API-driven state reflects backend
- [ ] No cross-pane imports
