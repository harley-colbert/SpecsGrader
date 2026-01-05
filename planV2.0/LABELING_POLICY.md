# Labeling Policy

## Approach
Staged labeling: reviewed items are written to a local staging file and counted immediately for UI/guardrails. The original training CSV is **not** modified automatically; staged labels can be merged into training data before a retrain.

## Storage
- File: `models/review_labels.jsonl`
- Format: one JSON object per line
- Fields:
  - `id`: string (unique, based on hash of text + timestamp)
  - `text`: string (spec/risk description)
  - `source_file`: string (classified file name when known)
  - `predicted_risk`: string
  - `predicted_dept`: string
  - `selected_risk`: string
  - `selected_dept`: string
  - `confidence`: float (top similarity or 0 if unavailable)
  - `label_source`: string (Label Source column)
  - `status`: string (`accepted`, `corrected`, `skipped`)
  - `created_at`: ISO-8601 string
  - `updated_at`: ISO-8601 string
  - `user_note`: optional string

## Duplicate handling
- If a new decision has the same `text` and `source_file` as an existing record, the new record overwrites the previous one (latest decision wins).
- Overwrite is implemented by re-writing the JSONL file without the older entry.

## Persistence behavior
- Actions in the Review Queue (Accept / Change Label / Skip) immediately write a record to `review_labels.jsonl`.
- Label counts shown in the Train panel include the base training CSV rows **plus** the number of persisted review labels.

## Audit / edits
- Each overwrite updates `updated_at`.
- No version history is stored beyond the latest record.

## Merge to training
- Before retraining, merge `review_labels.jsonl` into your labeled dataset (or point training to the merged file).
- Minimum required columns for training alignment: `Risk Description` (use `text`), `Risk Level` (`selected_risk`), `Review Department` (`selected_dept`).
