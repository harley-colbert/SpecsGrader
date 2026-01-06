# Agent_Aggregator

## Purpose
Implement priority-weighted aggregation across vector, llm, and rules into final dept/level predictions with confidence and thresholds.

## Responsibilities
- Implement weighted label scoring
- Priority order vector -> llm -> rules
- Compute normalized confidence
- Apply user thresholds at classify time
- Produce `methods_used` JSON record

## Inputs
- MethodPrediction objects
- weights and thresholds
- export schema requirements

## Outputs
- AggregatorService
- Unit tests for scoring logic and threshold flags
- methods_used structure

## Operating procedure (step-by-step)
1) Define weight constants (vector=1.0, llm=0.6, rules=0.2).
2) Compute label scores for dept and level separately.
3) Choose argmax label.
4) Compute normalized confidence.
5) Mark below-threshold flags but keep label outputs stable.
6) Build methods_used JSON that includes method outputs + final.
7) Add unit tests with synthetic method outputs to validate behavior.

## Tests / validation owned by this agent
- Unit tests for scoring and tie behavior
- Integration: classification pipeline uses aggregator correctly

## Definition of done
- [ ] Aggregation matches spec
- [ ] Confidence in [0,1]
- [ ] methods_used JSON includes required content
