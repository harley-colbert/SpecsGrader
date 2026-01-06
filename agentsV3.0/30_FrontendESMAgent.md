# FrontendESMAgent

## Mission
Build the **HTML/CSS/ESM-JS frontend** for SpecsGrader and keep it aligned with the backend API contract.

The frontend must be:
- No framework required
- No build step required
- Uses native ES modules (ESM) with relative imports
- Served as static files by FastAPI

## Inputs
- Repo working copy (unpacked `SpecsGraderV2.3.zip`)
- The current phase file from `planV3.0/`
- The API routes and schemas produced by the backend agents

## Required outputs
Depending on phase scope, create or update:
- `frontend/index.html`
- `frontend/styles.css`
- `frontend/src/main.js`
- `frontend/src/api.js`
- `frontend/src/state.js`
- `frontend/src/router.js`
- `frontend/src/components/*`
- `frontend/src/pages/*`

## Global constraints
- Do not depend on React/Vue/Svelte or any bundler tooling.
- Keep UI fast for large tables by using paging (never render 8000 rows at once).
- Do not hardcode backend URLs; assume same-origin API at `/api/*`.

## UX requirements (mirrors current desktop workflow)
- Left sidebar stepper:
  - Import
  - Train
  - Classify
  - Review
  - Export
- Top bar:
  - Current model set name
  - Status badges (server connected, job running, etc.)
- Main area: page content
- Review page includes a row inspector panel (can be right-side column or modal)

## Implementation guidance by phase

### Phase 1: UI shell
- Implement basic layout and a simple hash-router.
- Create placeholder pages with clear CTA buttons.

### Phase 5: Classification results rendering
- Implement a paged results table component:
  - Column header row
  - Body rows for current page only
  - Click row to open inspector

### Phase 6: Review workflow
- Implement review queue filters:
  - Default filter: Needs Review
  - Search box
  - Next/Previous navigation
- Inspector must support manual overrides and save.

### Phase 7: Export
- Export page must:
  - present export presets/options
  - call export endpoint
  - show download link

## Required tests
Run the phase tests listed in the plan. For frontend, at minimum:
- No console errors when loading `/`
- Pages load with router navigation
- Key API calls handle errors gracefully (render a user-visible message)

## Acceptance criteria
- The phase success checklist is satisfied.
- The UI can complete the phase’s intended workflow.
- No bundler or build tooling is required.

## Output format (agent response)
### Summary
### Files changed
### Tests run
### Notes
