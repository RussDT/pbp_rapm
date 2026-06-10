# Artifact Manifest

This repo contains a small source tree plus a large local artifact layer. The source tree should remain git-friendly; generated data should be reproducible, ignored, and documented.

## Git Policy

Commit by default:

- Source code under `nba_pipeline/scripts/`, `nba_pipeline/tests/`, `wnba_test/`, and small root operational scripts.
- Canonical docs under `nba_pipeline/docs/`.
- Small curated reference files with a documented reason.

Do not commit by default:

- Generated CSVs.
- Generated parquets.
- Logs.
- Model/cache files.
- Validation output directories.
- Browser exports and rendered media.

The current intentionally tracked generated/reference exception is:

- `nba_pipeline/external/bref_ft_pct_1997_2000.csv`

## Canonical Local Artifact Directories

| Path | Contents | Commit? | Notes |
| --- | --- | --- | --- |
| `nba_pipeline/raw_data/` | Raw NBA PBP parquet files such as `NBA26.parquet` and `NBA26_PS.parquet` | No | Regenerate with fetch/build scripts or copy a task-specific subset. |
| `nba_pipeline/processed/` | Processed metric parquets such as `RAPM26.parquet`, `FIRST_CHANCE26.parquet`, and validation-only variants | No | Rebuild from raw PBP and metric processors. |
| `nba_pipeline/results/` | Solver outputs, RAPM result CSVs, DRL/Shapley output parquets, shot-quality artifacts | No | Some outputs are publish inputs, but local copies remain generated. |
| `nba_pipeline/master_results/` | Consolidated weighted-factor CSV/parquet outputs | No | Downstream publish target source, not repo source. |
| `nba_pipeline/validation/` | Audit runs, debug parquets, reports, one-off proof artifacts | No | Keep local until no longer needed; summarize durable findings in docs. |
| `nba_pipeline/logs/` | Daily run and launchd logs | No | Local operational evidence only. |
| `nba_pipeline/external/` | External source checkouts, lookups, and supporting data | Usually no | Small curated references can be force-added with a note. |
| `archive/` | Legacy raw CSV archives | No | Local cache or migration source. |
| `wnba_test/Processed/` | WNBA processed CSV surfaces | No | WNBA downstream processing is currently CSV-based. |
| `wnba_test/Results/` | WNBA clean-decomp result outputs | No | Generated outputs. |

## Pre-Cleanup Inventory Snapshot

Before the 2026-06-08 cleanup in `/Users/russellthomas/Docs/pbp_rapm`, the checkout had:

- `239` tracked files.
- `149` tracked Python files.
- `50,910` total files outside skipped dirs in the repeatable inventory script.
- `50,499` local artifact-like files.

Largest local artifact groups at that snapshot:

| Path | Approx Size |
| --- | ---: |
| `nba_pipeline/validation/` | `4.8G` |
| `nba_pipeline/processed/` | `2.5G` |
| `archive/` | `2.1G` |
| `nba_pipeline/results/` | `1.9G` |
| `nba_pipeline/external/` | `628.1M` |
| `nba_pipeline/raw_data/` | `472M` |
| `wnba_test/Processed/` | `179.0M` |
| `nba_pipeline/master_results/` | `170M` |
| `nba_pipeline/logs/` | `79M` |

Treat this as a local snapshot, not a durable source-of-truth count. Regenerate it with `audit_repo_inventory.py` before cleanup.

## Post-Cleanup Inventory Snapshot

After removing ignored generated artifacts on 2026-06-08, the checkout had:

- `32M` total local checkout size.
- `264` total files outside skipped dirs in the repeatable inventory script.
- `239` tracked files.
- `149` tracked Python files.
- `2` artifact-like files in the repeatable inventory script.
- Ignored local config preserved: `.env`, `.claude/`, `.cursor/`, `.gstack/`, and `nba_pipeline/.claude/`.

Removed local artifact/cache groups included:

- `nba_pipeline/validation/`
- `nba_pipeline/processed/`
- `nba_pipeline/results/`
- `nba_pipeline/master_results/`
- `nba_pipeline/raw_data/`
- `nba_pipeline/logs/`
- `nba_pipeline/cache/`
- `nba_pipeline/external/merged_playbyplay/`
- `archive/`
- `wnba_test/Processed/`
- `wnba_test/Results/`
- root-level ignored CSV/parquet scratch files

## Cleanup Workflow

1. Commit or branch off all source/docs/tests work first.
2. Run:
   ```bash
   python nba_pipeline/scripts/audit_repo_inventory.py --output nba_pipeline/validation/repo_inventory
   ```
3. Review the generated markdown and CSV reports.
4. Use `git clean -ndX` to preview ignored-file cleanup.
5. Remove ignored artifacts only after confirming they are reproducible or backed up elsewhere.
6. Promote durable lessons into docs, not into generated reports.

## Publishing Boundary

Some generated files are copied into downstream repos such as `/Users/russellthomas/Docs/rapms` or `/Users/russellthomas/Docs/csvs`. That does not make the local generated copy source code. If a publish path changes, update the relevant docs and scripts in this repo, then verify the downstream branch and commit behavior separately.
