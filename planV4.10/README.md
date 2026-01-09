# SpecsGrader v4.10 – XLSX Column Contract Upgrade (D/E/F/G)

Date: 2026-01-09

## Goal

Update SpecsGrader to use the new Excel (.xlsx) column contract:

- **Column D**: Customer Specification (input text; source-of-truth for classification)
- **Column E**: Specific Risk description *(only when Risk Level is Medium or higher)*
- **Column F**: Risk Level classification *(model output)*
- **Column G**: Department for review *(model output)*

Behavioral contract:

1. Classification uses **Column D** only.
2. The system writes predicted labels to:
   - **F** (Risk Level)
   - **G** (Department)
3. The system determines **E** only when **F is Medium+**:
   - If **F ∈ {medium, high, extreme}** then:
     - **E = derived risk description** based on **D + F + G**
   - Else:
     - **E is blank** (or cleared), per plan rules.
4. Overwrite rules are explicit and test-covered (see Phase04/Phase05).

## Audience

These phase files are written for **ChatGPT Codex / agent-mode execution** and assume the matching **agents zip** exists with:
- OrchestratorAgent
- BackendAgent
- FrontendAgent
- MLAgent
- TestAgent
- QAAgent
- ReleaseAgent

## Output Deliverable

- `SpecsGraderv4.10.zip` (runnable from a clean unzip)
- Updated tests and fixtures
- Updated UI copy/help where Excel mappings are described

## Phase Order

Follow `PHASE_MAP.md` and execute phases in order without skipping.
