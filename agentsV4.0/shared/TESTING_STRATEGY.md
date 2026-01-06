# Testing Strategy

## Test tiers
1) Unit tests (pure Python)
   - services (ingest, rules, trainer, vector, llm parser, aggregator)
   - oversampling logic
   - confidence computation / normalization
2) API tests (FastAPI test client)
   - /api/state, /api/data/load, /api/train/*, /api/classify/*, /api/results/*
3) Integration smoke
   - bundle round-trip: train -> save -> load -> classify -> export
4) Manual performance smoke (documented)
   - 1,000 rows classify w/ vector on, UI responsive, cancel works

## Mandatory metrics checks (training)
- Macro F1 computed
- Balanced accuracy computed
- Per-class recall computed
- Guardrail: min per-class recall compared to configured threshold

## Fixtures
Maintain minimal fixtures in `tests/fixtures/`:
- training_sample.xlsx (has “Standards Risk Matrix” sheet)
- classify_sample.xlsx
- csv equivalents

Fixtures must include:
- row start at 5
- risk text in column E
- training labels in F and G
