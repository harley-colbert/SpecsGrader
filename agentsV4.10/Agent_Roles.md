# Agent_Roles (v4.10)

## OrchestratorAgent
- Executes phases in order and enforces gates.

## BackendAgent
- Implements XLSX ingestion and export behavior:
  - Parse D/E/F/G
  - Write F/G
  - Write/clear E based on medium+ rules
  - Preserve workbook structure and other columns/sheets

## FrontendAgent
- Updates UI copy/help panels referencing Excel mapping.
- Adds any advanced toggles (overwrite behavior) if implemented.

## MLAgent
- Ensures training extraction uses:
  - X = D
  - y_level = F
  - y_dept = G
- Implements deterministic `generate_specific_risk(D, F, G)`.

## TestAgent
- Creates XLSX fixtures and automated tests:
  - reader contract tests
  - training extraction tests
  - classify writeback tests for F/G
  - medium+ gating tests for E
  - best-effort end-to-end regression test

## QAAgent
- Manual end-to-end checks:
  - import workbook
  - classify
  - download
  - verify D/E/F/G behavior
- Captures screenshots or notes

## ReleaseAgent
- Version bump to 4.10
- Changelog updates
- Build `SpecsGraderv4.10.zip` and verify it runs from clean unzip
