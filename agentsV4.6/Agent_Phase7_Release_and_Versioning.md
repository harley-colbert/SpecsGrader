# Agent — Phase 7: Release and Versioning

You are the Phase 7 agent for the SpecsGrader v4.6 upgrade.

## Goal

- Finalize SpecsGrader v4.6 as a releasable version.
- Update version identifiers and changelog.
- Package a release archive and verify it runs.

## Inputs

- Completed work from Phases 0–6.
- `planV4.6/Phase7_Release_and_Versioning.md`.

## High-Level Steps

1. Update version references to 4.6.
2. Document changes in a changelog and/or README.
3. Tag the release in git (if applicable).
4. Build `SpecsGraderv4.6.zip` and verify it runs.

## Detailed Instructions

1. **Update version identifiers**
   - Locate any version string definitions:
     - `README.md` (mentions of v4.5).
     - Any `__version__` or similar variables in the code.
   - Update them to `4.6`.
   - Ensure that UI or logs that display the version reflect v4.6 where appropriate.

2. **Changelog / release notes**
   - If `CHANGELOG.md` exists:
     - Add a new entry for `4.6` summarizing:
       - Train pane redesign with Quick Start mode selector.
       - Path A: Load ModelSet and classify flow.
       - Path B: Build/Update stepper with gated steps.
       - Validation section refactor (Step 5 + optional Path A accordion).
       - Readiness strip and improved microcopy.
   - If `CHANGELOG.md` does not exist:
     - Create one with at least entries for v4.5 and v4.6.
   - Update `README.md` to:
     - Reference v4.6.
     - Briefly describe the new Train pane UX for end users.

3. **Tag the release (if git repo is present)**
   - If `.git` directory exists:
     - Ensure all changes are committed:
       - `git status` should show a clean working tree.
     - Commit if needed, with a message like:
       - `SpecsGrader v4.6 - Train pane UX upgrade`.
     - Create a local tag:
       - `git tag specsgrader_v4.6`
   - Do not push or expose any remote credentials in this environment.

4. **Build release archive**
   - From the project root, create a zip archive named `SpecsGraderv4.6.zip` containing:
     - Backend and frontend source code.
     - `requirements.txt`.
     - Package.json / lockfiles if needed.
     - Documentation, includ­ing the new changelog and upgrade notes.
   - Verify the archive structure matches prior releases (e.g. v4.5).

5. **Sanity check from a clean directory**
   - Extract `SpecsGraderv4.6.zip` into a **separate** folder.
   - In that folder:
     - Create and activate a new virtual environment.
     - Install dependencies: `pip install -r requirements.txt`.
     - If frontend build is required, run appropriate install/build commands.
     - Start the app: `python run.py`.
     - Confirm:
       - The Train pane shows:
         - Readiness strip.
         - Quick Start mode selector.
         - Path A and Path B flows.
         - Validation features and microcopy as expected.
       - Classification works with an existing ModelSet.

6. **Upgrade guidance**
   - Create a short doc `docs/UPGRADE_v4.5_to_v4.6.md` explaining:
     - That v4.6 is primarily a Train pane UX upgrade.
     - Existing ModelSets and `.sgm` bundles remain compatible.
     - No changes to core model training logic beyond gating and UI organization.
     - Any minor differences in behavior (e.g., stricter gating, new warnings).

7. **Final summary**
   - Write `docs/upgrade_v4.6/phase7_summary.md` containing:
     - Locations of version markers updated.
     - Tag name used.
     - Path to `SpecsGraderv4.6.zip`.
     - Overview of manual verification steps performed.

## Success Checklist

- [ ] Version identifiers updated to 4.6.
- [ ] Changelog and README include v4.6 changes.
- [ ] Git release tag created (if repo present).
- [ ] `SpecsGraderv4.6.zip` built and verified from a clean extraction.
- [ ] `docs/UPGRADE_v4.5_to_v4.6.md` and `phase7_summary.md` created.
