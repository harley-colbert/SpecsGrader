# Phase 03 — Rules engine and config

## Goal
Implement the rules-based classifier option using per-department keyword rules.
Rules must **abstain unless confident**.

Also implement `rules_config.json` creation/editing and bundle integration later.

## Agents
- Agent_BackendEngineer (lead)
- Agent_DataIngest
- Agent_QA
- Agent_FrontendEngineer (config UI)
- Agent_RepoAuditor

## Backend: rules_config.json format
Create a versioned format (example):

```json
{
  "version": "1.0",
  "departments": {
    "mechanical": { "keywords": ["bearing", "pneumatic", "..."], "min_hits": 1 },
    "electrical": { "keywords": ["480v", "panel", "..."], "min_hits": 1 },
    "controls": { "keywords": ["plc", "hmi", "ladder", "..."], "min_hits": 1 },
    "project_management": { "keywords": ["schedule", "lead time", "submittal", "..."], "min_hits": 1 }
  },
  "global": {
    "case_sensitive": false,
    "match_mode": "token_contains",
    "abstain_on_tie": true
  }
}
```

## Backend: RuleService
Create `backend/app/services/rule_service.py`:
- `load_rules(path_or_bundle) -> RulesConfig`
- `predict(risk_text) -> MethodPrediction`
  - returns:
    - `dept_pred` or null
    - `dept_conf` 0..1
    - optionally `level_pred`/`level_conf` (default off; implement as “future extension”)

### Confidence rules
- Compute hits per dept
- If best dept has hits >= min_hits and strictly greater than second-best, predict
- If tie or below min_hits, abstain
- `dept_conf` can be mapped from hit strength (simple monotonic mapping is fine)

## API endpoints
- `GET /api/rules/get`
- `POST /api/rules/set`
- `POST /api/rules/test`
  - body: `{ "text": "...", "rules": optional }`
  - returns prediction + matched keywords

## Frontend
Train pane:
- “Rules” section with:
  - JSON editor (or form editor) for keywords per department
  - “Test rule” input box to see matches
  - Save to workspace config (not bundle yet)

## Testing (must run and pass)
```bash
python -m pytest -q
```

Required tests:
- abstain on tie
- abstain when no dept hits
- predict when one dept clearly wins
- config version parsing works

## Success checklist
- [ ] Rules predict dept only when confident
- [ ] Rules abstain on tie/insufficient hits
- [ ] UI can edit rules and test them
- [ ] Rules output includes evidence (matched keywords) for `methods_used`
