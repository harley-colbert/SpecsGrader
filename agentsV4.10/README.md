# agentsV4.10 (SpecsGrader)

These agent prompts are designed to execute **planV4.10.zip** (folder `planV4.10/`) in order, upgrading:

**SpecsGrader v4.9 → SpecsGrader v4.10**  
(XLSX Column Contract Upgrade: D/E/F/G)

## New XLSX Contract (v4.10)

- **D**: Customer Specification (input text; source-of-truth for classification)
- **E**: Specific Risk description *(only when risk level is Medium+)*  
- **F**: Risk Level classification *(model output)*
- **G**: Department for review *(model output)*

Medium+ means: `medium`, `high`, `extreme`.

## How to use

1. Unzip `planV4.10.zip` and `agentsV4.10.zip` into the same working directory.
2. Start with `Orchestrator_AgentsV4.10.md`.
3. Execute phases in order: Phase00 → Phase06.
4. Each phase prompt:
   - tells you which plan file to open
   - tells you what to build
   - tells you what tests must pass
   - tells you what evidence to record

## Assumptions

- You have the starting codebase available and runnable.
- You can run Python tests (pytest) and any frontend build/test checks if present.
- All workflows must remain local/offline.

## Required outputs after each phase

Create:
- `workspace/reports/PhaseXX_Completion.md`

Include:
- Summary (what changed and why)
- Key files touched
- Tests run (commands + results)
- UI evidence (clickpath + observed behavior)
- Follow-ups/risks

Date: 2026-01-09
