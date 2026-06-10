# Repo Cleanup Plan

This is a staged plan for making the repo easier for another engineer to work in without deleting useful research state by accident.

## Goals

- Keep the working source tree small and understandable.
- Preserve current pipeline behavior and methodology docs.
- Separate generated data from source code.
- Classify old scripts before deleting or moving them.
- Give new contributors a small verified workflow.

## Non-Goals

- Do not delete local artifacts as a first step.
- Do not rewrite the pipeline structure before the active workflows are documented.
- Do not remove legacy scripts solely because they are old.
- Do not commit generated CSV/parquet/log/model artifacts as part of cleanup.

## Phase 1: Stabilize Current Work

1. Resolve the dirty worktree into one or more coherent commits.
2. Keep current source/docs/tests changes separate from generated artifacts.
3. Ensure `AGENTS.md`, `CLAUDE.md`, `nba_pipeline/CLAUDE.md`, and `nba_pipeline/docs/README.md` agree on the current operating rules.
4. Run focused validation for any already-modified code before inviting a collaborator into the branch.

Exit criteria:

- `git status --short` shows only intentional untracked local artifacts or is clean.
- The collaborator can clone the branch and read a single docs index for the current workflow.

## Phase 2: Artifact Triage

1. Generate a current inventory:
   ```bash
   python nba_pipeline/scripts/audit_repo_inventory.py --output nba_pipeline/validation/repo_inventory
   ```
2. Review the biggest local artifact directories.
3. Identify required seed data for common smoke tests.
4. Add regeneration commands to docs for any artifact a contributor is expected to use.
5. Use `git clean -ndX` as a preview before deleting ignored outputs from a local checkout.

Exit criteria:

- `ARTIFACT_MANIFEST.md` names the artifact classes and owner workflows.
- A fresh clone does not need the full local `16G` artifact layer to start source work.

## Phase 3: Script Classification

Classify scripts into four buckets:

- `active`: called by daily jobs, current docs, tests, or current operator workflows.
- `legacy`: still useful for backfills, comparisons, or historical rebuilds, but not part of daily operation.
- `experiment`: research code that can remain isolated.
- `archive candidate`: no docs, no tests, no imports, no known current workflow.

Suggested first-pass labels:

| Area | Likely Classification |
| --- | --- |
| `nba_pipeline/scripts/01_fetch_pbp_data.py`, `02_process_rapm.py`, `03_run_rapm_analysis.py`, `rapm.py` | Active |
| `nba_pipeline/scripts/process_rapm_blocks/` | Active core |
| `nba_pipeline/daily_rapm_update.sh` | Active operations |
| `nba_pipeline/scripts/run_alt3_efg_value_rolling.py` | Active publish/build path |
| `nba_pipeline/scripts/build_clean_lineup_net_rating.py` | Active Databallr WOWY/net-rating path |
| `nba_pipeline/scripts/build_shot_clock_estimates.py` | Active/audit path |
| `wnba_test/clean_decomp_wnba.py`, `wnba_test/process_rapm_wnba.py`, `wnba_test/fetch_wnba_cdn.py` | Active WNBA path |
| `randle_strong_weak/` | Experiment/research |
| Root `attempt.py`, `rapm_experiments.py`, `rapm_timedecay_attempt.py` | Archive candidates unless a current doc references them |
| Root `update_2026_*.py` | Operational legacy/current export helpers; classify individually |

Exit criteria:

- A generated script inventory exists under `nba_pipeline/validation/repo_inventory/`.
- Each archive candidate has been checked for docs references and imports before moving or deleting.

## Phase 4: Light Restructure

Only after classification:

1. Move archive candidates into a clearly named `legacy/` or `experiments/` area, or delete them in a dedicated cleanup commit.
2. Add README notes for retained legacy/experiment areas.
3. Keep command compatibility wrappers for scripts that external jobs may call.
4. Update docs and daily job references in the same commit as any path move.

Exit criteria:

- No active command in docs or shell jobs points to a moved/deleted script.
- Focused compile/tests pass.

## Phase 5: Contributor Handoff

For the friend/collaborator:

1. Share the clean branch.
2. Share required env vars out-of-band; do not commit secrets.
3. Share only the minimum task-specific artifacts.
4. Give them one first issue that touches source/docs/tests, not the full data pipeline.
5. Ask them to include validation output in their handoff.

