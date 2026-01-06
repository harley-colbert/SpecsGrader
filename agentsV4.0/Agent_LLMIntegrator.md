# Agent_LLMIntegrator

## Purpose
Implement OpenRouter LLM classification option with env-only API key, structured prompt, strict parsing, and never-send server-side blocking.

## Responsibilities
- Read OPENROUTER_API_KEY from env only
- Implement llm_prompt.json template and schema validation
- Call OpenRouter chat completions
- Strict JSON parse, validate enums
- Never-send mode blocks server-side

## Inputs
- prompt schema requirements
- never-send behavior
- method prediction schema

## Outputs
- LLMService predict()
- settings endpoints
- tests for blocking + parsing

## Operating procedure (step-by-step)
1) Implement settings storage in AppState.
2) Enforce never-send mode in the LLM endpoint and service.
3) Implement request builder from llm_prompt.json.
4) Implement strict JSON parsing; on failure abstain or return structured error.
5) Store reason in evidence but keep UI hidden by default.
6) Add tests: missing env -> error, never-send -> blocked, invalid JSON -> abstain.

## Tests / validation owned by this agent
- Unit tests for parser
- API tests for settings and LLM predict guard

## Definition of done
- [ ] No key stored on disk
- [ ] never-send blocks LLM server-side
- [ ] JSON parsing robust and enum validated
