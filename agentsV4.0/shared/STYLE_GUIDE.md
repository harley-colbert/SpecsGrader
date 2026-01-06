# Style Guide

## Python
- Python 3.11+ recommended
- Prefer dataclasses / pydantic models for request/response payloads
- No UI logic in services
- Avoid global mutable state; use a single `AppState` instance owned by the app
- Threading: use thread-safe primitives for cancel flags and progress updates
- Logging: structured, with clear job IDs

## Frontend (HTML/CSS/ESM JS)
- No framework required
- Prefer small modules; panes are separate files
- One source of truth is `/api/state`
- Keep UI responsive: avoid blocking operations in JS; rely on backend jobs + polling

## UX basics
- Light theme
- Clear error messages
- Progress bar + cancel for long operations
- Table filters for Results
