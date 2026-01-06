# Agent_VectorStore

## Purpose
Implement local embeddings, vector store persistence, NN querying, and vector method outputs for similarity classification.

## Responsibilities
- Choose and implement one local embedding backend
- Persist embeddings + metadata in bundle vector_store/
- Implement kNN similarity vote for level+dept
- Optional embedding classifier (weighted + calibrated)
- Provide evidence (top neighbors)

## Inputs
- training rows and labels
- vector_embedder.json schema
- performance target 100–1000 rows

## Outputs
- VectorStore build/reload
- VectorService predict(text)
- Tests for build/query/predict

## Operating procedure (step-by-step)
1) Implement embedder interface.
2) Implement store build (embeddings + rows.jsonl + index).
3) Implement query returning top K neighbors with similarity.
4) Implement vote aggregation and confidence mapping.
5) Keep artifacts under vector_store/.
6) Add tests for deterministic querying on fixtures (or stable synthetic embeddings).

## Tests / validation owned by this agent
- Unit tests: embedder output shape
- Unit tests: store round-trip
- Unit tests: predict returns schema + confidences

## Definition of done
- [ ] Local embeddings default works
- [ ] vector_store persists correctly
- [ ] predict returns dept/level + confidence + evidence
