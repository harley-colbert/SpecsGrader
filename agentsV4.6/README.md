# agentsV4.6 — SpecsGrader Train Pane UX Upgrade

This archive contains **agent prompts** for executing the steps defined in `planV4.6.zip` to upgrade
SpecsGrader from **v4.5** → **v4.6**.

## How to use

- Open the corresponding `PhaseX_*.md` file from **planV4.6**.
- Then open the matching `Agent_PhaseX_*.md` from this archive.
- Feed the agent file as the **system/instruction prompt** to ChatGPT / ChatGPT Codex in a workspace
  that already has:
  - the SpecsGraderv4.5 codebase,
  - the planV4.6 files,
  - and any other needed project files.
- Let the agent carry out the steps, run tests, and report results.

Files:
- `Orchestrator_AgentsV4.6.md` — top-level controller for running all phases in order.
- `Agent_Phase0_Baseline_and_Backup.md`
- `Agent_Phase1_QuickStart_ModeSelector.md`
- `Agent_Phase2_PathA_Load_ModelSet_and_Classify.md`
- `Agent_Phase3_PathB_Build_Update_ModelSet_Stepper.md`
- `Agent_Phase4_Validation_Section_Refactor.md`
- `Agent_Phase5_ReadinessStrip_and_Microcopy.md`
- `Agent_Phase6_QA_Usability_and_Regression.md`
- `Agent_Phase7_Release_and_Versioning.md`
