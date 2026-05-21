# Git Backup Strategy

This repo should be recoverable from GitHub without trying to version every
pipeline artifact. Git is the source-of-truth for code, docs, tests, and small
curated reference files. Large generated data stays outside git.

## Track In Git

- Root operator docs: `AGENTS.md`, `CLAUDE.md`, `HELP_README.md`, and command notes.
- Pipeline source: `nba_pipeline/scripts/`, `nba_pipeline/tests/`, top-level helper
  scripts, and DRL/Shapley harness code.
- Pipeline docs: everything under `nba_pipeline/docs/`, including this file.
- WNBA source code under `wnba_test/*.py`.
- Local app source under `csv-viewer/`, excluding generated `public/data/`.
- Small curated reference artifacts that are hard to reconstruct and explicitly
  unignored, such as `nba_pipeline/external/bref_ft_pct_1997_2000.csv`.

## Keep Out Of Git

- Raw and processed play-by-play data: `raw_data/`, `processed/`, `archive/`,
  `wnba_test/Processed/`, and `wnba_test/Results/`.
- Solver outputs and publish artifacts: `results/`, `master_results/`,
  `Rubberband_Results/`, and `csv-viewer/public/data/`.
- Caches, local validation sandboxes, logs, model checkpoints, and temporary
  training bundles.
- Secrets and machine-local config: `.env`, `.cursor/`, `.gstack/`, and local
  Claude/Codex settings.

## Rescue Branch Workflow

Use a dated branch when the working tree is large and partially experimental:

```bash
git switch -c codex/repo-rescue-YYYYMMDD
git status --short --untracked-files=all
git add .gitignore AGENTS.md CLAUDE.md HELP_README.md commands.md
git add nba_pipeline wnba_test csv-viewer
git status --short
git commit -m "Back up repo source and docs"
git push -u origin codex/repo-rescue-YYYYMMDD
```

Before committing, scan the staged set:

```bash
git diff --cached --stat
git diff --cached --name-only | rg '\.(csv|parquet|pkl|npz|pt|log|gz)$'
```

The second command should normally return nothing except intentionally
force-added small reference files.

## Artifact Backup Rule

If an artifact is expensive or impossible to regenerate, do not sneak it into
git. Put it in external storage or a separate artifact backup location and add a
small manifest or doc entry that records:

- artifact path
- source command or upstream source
- date generated
- why it is worth preserving
- where the non-git backup lives

This keeps GitHub usable while still making important data recoverable.
