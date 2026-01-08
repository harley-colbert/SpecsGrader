# Phase 7 — Release and Versioning

## Objectives

- Finalize SpecsGrader v4.6 as a releasable package.
- Tag the repository and document the Train pane UX changes.
- Provide guidance for users upgrading from v4.5.

## Required Context

- All prior phases completed and validated.
- No known critical issues remain in the Train pane.

## Tasks

1. **Version bump**

   - Update any version identifiers in:
     - `README.md`
     - Any `__version__` or similar constants.
   - Clearly set version to `4.6`.

2. **Changelog / Release notes**

   - Add or update a `CHANGELOG.md` (or equivalent) entry:
     - Summarize the Train pane changes:
       - Quick Start workflow chooser.
       - Path A (Load & classify) simplifications.
       - Path B (Build/Update) stepper.
       - Validation UI refactor.
       - Readiness strip and improved microcopy.
   - Note that backend logic for training and classification remains consistent with v4.5, with changes only in UI flow and help text.

3. **Tag the release**

   - In git:
     - `git commit` all changes with message like `SpecsGrader v4.6 - Train pane UX upgrade`.
     - `git tag specsgrader_v4.6` (or similar).

4. **Build release artifact**

   - Package the code into `SpecsGraderv4.6.zip` (or following existing packaging conventions).
   - Ensure the packaged archive includes:
     - All updated frontend and backend code.
     - Requirements file(s).
     - Instructions for running the app.

5. **Upgrade guidance from v4.5**

   - Document in README or a short `UPGRADE_v4.5_to_v4.6.md`:
     - That existing ModelSets and `.sgm` bundles are still valid.
     - That classification behavior does not change; only the Train pane UX is improved.
     - Any minor behavior changes in UI messaging or error handling that may be notable.

## Tests

- Verify that:
  - The zipped package can be unpacked into a fresh directory.
  - The app starts and runs correctly from that new directory.
  - Train and Classify panes work as expected.

## Success Checklist

- [ ] Version updated to 4.6 in all relevant files.
- [ ] Changelog / release notes describe the Train pane UX changes.
- [ ] Git tag for v4.6 created.
- [ ] SpecsGraderv4.6.zip (or equivalent) built and verified.
- [ ] Upgrade guidance from v4.5 → v4.6 is written.
