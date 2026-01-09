# Changelog

## 4.10
- XLSX contract update: D=input spec, E=specific risk (medium+ only), F=risk level, G=department.
- Classification writes back F/G predictions and optional E specific-risk notes.
- Training pipeline updated to use spec text (D) and validated labels (F/G).
- Removed the tracked SpecsGraderv4.10.zip binary archive from the repository.

## 4.9
- Dataset health checks with imbalance warnings and blocking errors.
- Per-class validation metrics with confusion matrices.
- Model insights with top TF-IDF terms per class.
- Why/Trace evidence in classify results and decision policy evaluation.
- Decision policy configuration with weighted aggregation.
- Embedding backends (TF-IDF, LSA, optional transformer) and upgraded vector voting.
- Optional deep classifier layer (disabled by default).

## 4.6
- Train pane UX redesign with Quick Start mode selector.
- Path A flow for loading a saved ModelSet and going straight to classify.
- Path B build/update stepper with gated actions.
- Validation refactor with optional Path A accordion.
- Readiness strip and improved Train pane microcopy.

## 4.5
- Baseline release prior to the v4.6 Train pane UX upgrades.
