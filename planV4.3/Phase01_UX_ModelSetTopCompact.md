# Phase 1 — UX: Move Model Set (.sgm) to top + compact layout

## Objective
Move the **Model Set (.sgm)** section to the **top** of the Train pane and make it **more compact** without changing backend behavior.

## Scope
- Frontend layout + styling only.
- No API changes required unless a UI dependency is discovered.

## Primary files expected to change
- `frontend/src/panes/trainPane.js`
- `frontend/styles.css`
- (optional) shared UI components under `frontend/src/ui/*` if the layout primitives enforce spacing

## Implementation steps
1. **Reorder sections**
   - Render the ModelSet card first in the Train pane.
2. **Compact the ModelSet card**
   - Reduce vertical padding/margins specifically for this card.
   - Place ModelSet selector + Version selector on one row (or equivalent compact structure).
   - Group actions (Load / Save Snapshot / Export / Import / Delete) into a single button row that wraps.
3. **Preserve clarity**
   - Keep the “active modelset/version” indication visible.
   - Ensure errors remain visible (don’t hide failure states in compact mode).
4. **Responsive behavior**
   - Ensure the compact layout wraps properly on narrow widths.
   - No overlapping controls, no hidden buttons.

## Tests that must pass
### Automated
- `pytest -q`

### Manual UI
1. Start app: `python run.py`
2. Open Train pane:
   - Expected: Model Set (.sgm) section is **first**.
3. Resize window narrow/wide:
   - Expected: controls wrap; nothing overlaps.
4. Export and Import:
   - Expected: Export still triggers the intended download/save behavior; Import still works and imported versions appear.
5. Load a modelset version:
   - Expected: still hydrates Training/Vector/Rules (Phase 1 in v4.2.2 must not regress).

## Success checklist
- [ ] Model Set (.sgm) section appears at the top of Train pane
- [ ] ModelSet UI is visibly more compact (fewer rows / less whitespace)
- [ ] Controls remain usable at narrow widths (wrap, no overlap)
- [ ] Export/Import/Load still work
- [ ] No new console error spam introduced
- [ ] `pytest -q` passes
