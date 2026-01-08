# Phase 12 — Vector Voting Upgrade + DecisionPolicy Integration

## Goal
Strengthen vector-based classification (distance-weighted voting, stronger confidence computation) and ensure DecisionPolicy can use these signals consistently across backends.

## Primary agents
- MLAgent
- BackendAgent
- TestAgent

## Scope
- Backend: update `VectorStore` neighbor aggregation and `VectorService` confidence computations.
- DecisionPolicy thresholds remain meaningful across tfidf/lsa/transformer.

## Implementation steps (do in order)
- MLAgent/BackendAgent:
-   1) In `backend/app/vector/vector_store.py`, update neighbor scoring:
-      - implement distance-weighted voting for labels
-      - compute confidence as normalized vote share for top label
-   2) Ensure `VectorPrediction` exposes:
-      - `top_similarity`, `second_similarity`, `margin` (keep existing)
-      - `vote_conf_level`, `vote_conf_dept` (optional but recommended)
-   3) Update DecisionPolicy evaluation in aggregation to optionally use vote confidence instead of similarity thresholds when present.
-   4) Keep backward compatibility: if vote_conf not present, use similarity+margin as in v4.6.

## Testing work to CREATE/UPDATE in this phase
- Add `tests/test_vector_voting.py`:
-   - Create a tiny vector store with known neighbors
-   - Assert distance-weighted vote yields expected label
-   - Assert confidence increases when nearest neighbor dominates
- Update `tests/test_aggregate_service.py` if aggregation now reads new vector fields.

## Tests that MUST pass (gate)
- `python -m pytest -q`

## Success checklist (must be YES for every item)
- ✅ Vector predictions are more stable and confidence is meaningful.
- ✅ Aggregation policy can use vector confidence signals consistently.
- ✅ Tests cover voting behavior.
