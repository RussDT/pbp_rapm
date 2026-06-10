# Contributor Onboarding

This repo is easier to work in when source code and generated basketball data are treated as separate layers. New contributors should work from a clean clone and pull only the data artifacts needed for their assigned task.

## First 30 Minutes

1. Read the repo operating instructions:
   - `AGENTS.md`
   - `CLAUDE.md`
   - `nba_pipeline/CLAUDE.md`
   - `nba_pipeline/docs/README.md`
2. Check the worktree before editing:
   ```bash
   git status --short
   ```
3. Skim the artifact policy:
   - `nba_pipeline/docs/ARTIFACT_MANIFEST.md`
   - `nba_pipeline/docs/GIT_BACKUP_STRATEGY.md`
4. Run a source-only sanity check:
   ```bash
   python -m compileall nba_pipeline/scripts nba_pipeline/tests
   ```
5. Run focused tests for the area being touched. Do not start with a full daily pipeline run unless the task explicitly needs regenerated outputs.

## Repo Map

- `nba_pipeline/scripts/`: primary pipeline, processing, solving, publishing, and audit scripts.
- `nba_pipeline/scripts/process_rapm_blocks/`: shared possession and metric-building logic. Changes here have broad blast radius.
- `nba_pipeline/tests/`: focused regression tests. Add tests here for parser, algebra, and small helper behavior.
- `nba_pipeline/docs/`: canonical FreshDocs surface. Update matching docs when behavior, methodology, output schema, or operator workflow changes.
- `wnba_test/`: WNBA-specific CSV-based processing and clean EFG-value decomposition work.
- `randle_strong_weak/`: local research scripts for opponent-strength experiments.
- Root-level `update_2026_*.py` scripts: downstream export/update helpers. Treat these as operational scripts, not general library code.

## Data And Artifact Boundaries

Generated CSV, parquet, log, model, and validation outputs are intentionally ignored by git. On a fresh clone, expect these directories to be missing or mostly empty:

- `nba_pipeline/raw_data/`
- `nba_pipeline/processed/`
- `nba_pipeline/results/`
- `nba_pipeline/master_results/`
- `nba_pipeline/validation/`
- `nba_pipeline/logs/`
- `nba_pipeline/cache/`
- `archive/`
- `wnba_test/Processed/`
- `wnba_test/Results/`

If a task needs one of those artifacts, document the source, copy or regenerate the smallest needed subset, and keep it out of git unless the user explicitly asks for a curated reference file.

## Safe Work Sequence

1. Start a branch from the current canonical branch.
2. Run `git status --short` and confirm there are no unrelated local changes.
3. Read the docs for the area being changed.
4. Make the smallest source/doc/test change that solves the task.
5. Run focused validation.
6. Update `nba_pipeline/docs/README.md` when adding a canonical doc.
7. Append one short entry to `nba_pipeline/docs/Codex_Scratchpad.md` at session end.

## Cleanup Rules

- Do not delete generated data from someone else's working checkout without a dry-run inventory and explicit approval.
- Do not commit generated CSV/parquet/log/model files unless the file is a small curated reference and the reason is documented.
- Do not collapse experimental scripts into core pipeline code just to reduce file count. First classify them as active, legacy, experiment, or archive candidate.
- Prefer adding focused tests before touching shared parser logic in `process_rapm_blocks/common.py` or `process_rapm_blocks/process_rapm.py`.

## Useful Dry Runs

Inspect ignored files that could be removed from a local checkout:

```bash
git clean -ndX
```

Measure local artifact weight:

```bash
du -sh nba_pipeline/{raw_data,processed,results,master_results,validation,logs,external} 2>/dev/null
```

Build a script/artifact inventory:

```bash
python nba_pipeline/scripts/audit_repo_inventory.py --output nba_pipeline/validation/repo_inventory
```

