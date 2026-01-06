# Phase 06 — OpenRouter LLM and Never-Send mode

## Goal
Implement the LLM method via OpenRouter with:
- env-only API key
- model choice at classify time
- structured prompt + strict JSON parsing
- reason captured but hidden by default
- hard backend guard for “Never send data externally” mode

## Agents
- Agent_LLMIntegrator (lead)
- Agent_BackendEngineer
- Agent_QA
- Agent_RepoAuditor
- Agent_FrontendEngineer

## Security requirements (mandatory)
- API key is read from environment variable only (e.g., `OPENROUTER_API_KEY`)
- Never store the key in files, local settings, or bundle artifacts
- Never-send mode must:
  - disable LLM toggle in UI
  - AND block requests server-side (return error)

## Prompt artifact
Create `llm_prompt.json` schema:
- version
- system message
- user template
- required JSON schema definition
- parsing/validation rules

LLM must return JSON with:
- `risk_level`
- `department`
- optional `confidence` (if model provides; otherwise compute a proxy)
- optional `reason` (stored but hidden by default)

## Backend: LLMService
Create `backend/app/services/llm_service.py`:
- `predict(risk_text, model_name, never_send_mode) -> MethodPrediction`
- Strict parsing:
  - reject non-JSON
  - validate enums
  - if invalid, abstain (or return error based on a flag)

## API endpoints
- `POST /api/llm/test` (for prompt debugging)
- `POST /api/llm/predict` (used by classifier pipeline only)
- `GET /api/settings` and `POST /api/settings`
  - store never-send mode in AppState (not in bundle)

## UI (Classify pane)
- LLM toggle
- model selector text field (or dropdown prefilled)
- “Never send externally” toggle (global)
- warning banner when LLM is blocked

## Testing (must run and pass)
```bash
python -m pytest -q
```

Required tests:
- never-send mode blocks predict even if endpoint called
- missing env var returns clear error
- parsing rejects invalid labels
- “reason” stored but not required for UI rendering

## Success checklist
- [ ] Env-only API key behavior confirmed
- [ ] Never-send mode enforced server-side
- [ ] LLM predictions parse into enums reliably
- [ ] LLM method can abstain safely on bad outputs
