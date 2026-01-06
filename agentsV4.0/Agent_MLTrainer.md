# Agent_MLTrainer

## Purpose
Implement the default supervised text models for level and department with severe-imbalance best practices: class weights, calibrated probabilities, macro metrics, optional capped oversampling.

## Responsibilities
- TF-IDF + Logistic Regression (class-weighted)
- Calibrated probabilities
- Macro F1, balanced accuracy, per-class recall
- Guardrail: min per-class recall >= configurable threshold
- Optional capped oversampling (no SMOTE)

## Inputs
- TrainingDataset with risk_text + labels
- training params from UI
- enums and metrics requirements

## Outputs
- Serialized sklearn pipelines (joblib)
- Metrics in bundle_meta/workspace
- Unit tests for oversampling + calibration presence

## Operating procedure (step-by-step)
1) Build two pipelines (risk level, department).
2) Use class_weight='balanced' or explicit inverse frequency.
3) Calibrate probabilities (CalibratedClassifierCV).
4) Compute metrics on held-out split (stratify when possible).
5) Implement capped oversampling by duplicating minority samples up to cap.
6) Expose training params in TrainingService.
7) Return metrics object used by UI.
8) Add tests verifying predict_proba and metric keys exist.

## Tests / validation owned by this agent
- Unit tests: oversampling cap behavior
- Unit tests: calibrated predict_proba works
- API tests: /api/train/* lifecycle

## Definition of done
- [ ] Models train deterministically on fixtures
- [ ] predict_proba available
- [ ] Macro metrics computed
- [ ] Guardrail evaluated and surfaced
