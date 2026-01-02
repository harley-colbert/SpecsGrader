# SpecGrader Vector DB Upgrade Plan — planV2.0

This plan upgrades the current SpecGrader app (from `SpecsGrader-main.zip`) to **v2.0** by adding a **persistent vector database** for top‑K similarity retrieval and evidence-backed grading, while keeping backward compatibility with the existing:
- rules engine
- classic TF‑IDF model
- semantic embedding Logistic Regression model
- model set registry (`models/model_sets.json`)

## What this plan delivers (v2.0)
- Persistent **Vector DB** (recommended: **ChromaDB**) stored per model set.
- New `vector_db.py` abstraction layer.
- Training pipeline builds a vector index automatically.
- Inference uses **top‑K retrieval** + **weighted voting** (risk + dept).
- UI controls: enable/disable vector similarity, choose K, choose similarity threshold.
- “Needs Review” flag for low confidence / low similarity / disagreement cases.
- Migration script for existing `training_data_embeddings.pkl` model sets.
- Smoke tests and unit tests that must pass without errors.

## Non-goals (explicitly out of scope for v2.0)
- Knowledge graph / ontology system.
- Fine-tuning a large LLM.
- Replacing scikit-learn models with deep models.
- Multi-user server deployment.

## Agents assumed to already exist (will be provided in agentsV2.0.zip)
Each phase below references one or more agents by ID. The phase files are written **as if the agent prompt files already exist**.

- AGENT_00_REPO_AUDITOR
- AGENT_01_DEPENDENCY_MANAGER
- AGENT_02_VECTOR_DB_ENGINEER
- AGENT_03_TRAINING_PIPELINE_ENGINEER
- AGENT_04_INFERENCE_PIPELINE_ENGINEER
- AGENT_05_UI_ENGINEER
- AGENT_06_TEST_ENGINEER
- AGENT_07_RELEASE_MANAGER

## How to run this plan in agent-mode
1. Unzip your working repo (e.g., `SpecsGrader-main/`).
2. Unzip `planV2.0.zip` somewhere convenient.
3. For each phase file (Phase_00 … Phase_09):
   - run the referenced agent(s) in order,
   - follow the exact “Implementation Steps”,
   - run every “Must-Pass Tests” command,
   - do not proceed until the phase success checklist is fully satisfied.

---

**Important behavioral rule for v2.0:**
- The app must **still start** and allow **manual grading** even if no models exist yet (fresh install).
- Vector DB similarity must be **optional** and must fail “softly” (warnings, not crashes) when not available.
