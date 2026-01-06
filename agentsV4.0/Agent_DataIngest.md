# Agent_DataIngest

## Purpose
Implement CSV/XLSX ingest rules and canonical row mapping exactly as specified (row 5 start, columns E/F/G, “Quote #” sheet selection).

## Responsibilities
- CSV and XLSX loaders
- XLSX: select first sheet containing 'Quote #'
- Start at row 5, 1-indexed
- Column E always risk text
- Training: F risk level, G department
- Provide dataset summary and preview

## Inputs
- sample files
- ingest constraints from plan and enums
- expected dataset summary fields

## Outputs
- IngestService and data models
- API endpoints `/api/data/load` and `/api/data/preview`
- Tests/fixtures verifying selection and mapping

## Operating procedure (step-by-step)
1) Implement loaders using pandas + openpyxl.
2) Ensure robust trimming/normalization of risk text.
3) Validate labels against enums and collect warnings.
4) Never crash on bad rows; record errors and continue.
5) Return canonical rows for preview.
6) Add fixtures and unit tests for all edge cases (missing sheet, empty data, etc.).

## Tests / validation owned by this agent
- Unit tests: correct sheet, row start, column mapping
- API tests: preview returns expected canonical fields

## Definition of done
- [ ] All ingest constraints met
- [ ] Fixtures cover XLSX/CSV
- [ ] Preview and summary endpoints pass tests
