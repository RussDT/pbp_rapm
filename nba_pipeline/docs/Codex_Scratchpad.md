# Codex Scratchpad

This file is short-term session memory only.
Durable operating rules belong in root `AGENTS.md` and `CLAUDE.md`.

## How To Use

1. Read the latest entries at session start.
2. Append one short entry at session end.
3. When a lesson becomes durable, move it into `AGENTS.md` / `CLAUDE.md`.
4. Keep only the latest 5 entries in this file.

Never store secrets, tokens, or credentials here.

## Session Log Template

```md
### YYYY-MM-DD HH:MM (local)
- Task:
- What went wrong:
- User correction:
- Fix applied:
- Prevention rule added:
- Outcome:
```

## Session Log

### 2026-05-21 12:00 (local)
- Task: Make the repo resilient to local data loss without uploading every generated CSV/parquet artifact.
- What went wrong: The repo had only 16 tracked files but about 47k untracked files, mostly generated CSV/parquet/log outputs; `.git` also held several GB of loose objects from prior oversized git-object writes.
- User correction: User asked for a sustainable git structure because the uncommitted tree was overwhelming and could not all be uploaded.
- Fix applied: Added a source-first `.gitignore`, documented `docs/GIT_BACKUP_STRATEGY.md`, and promoted the code/docs-vs-artifacts backup rule into root and pipeline agent docs.
- Prevention rule added: Track code, docs, tests, and small curated reference files; keep generated data/results/log/model artifacts out of git unless intentionally force-added with a documented reason.
- Outcome: The repo has a repeatable rescue-branch workflow and a staged path for pushing source/docs while excluding bulky generated outputs.

### 2026-05-19 13:35 (local)
- Task: Build regular-season assisted dunk passer-dunker combo exports for the last 30 seasons.
- What went wrong: N/A.
- User correction: N/A.
- Fix applied: Parsed `NBA97.parquet` through `NBA26.parquet` for made dunk descriptions with AST tags, resolved passers against offensive lineups where possible, and wrote aggregate/event/summary CSVs under `nba_pipeline/validation/`.
- Prevention rule added: Keep event-level audit rows alongside aggregate combo counts for historical assist-text extracts, because unresolved/ambiguous passer fragments need traceability.
- Outcome: Exported `209,021` assisted dunk events and `61,354` season-combo rows; top single-season combo was 2018-19 Harden-Capela with `129`.

### 2026-05-19 13:15 (local)
- Task: Check whether 2026 RS raw PBP can produce passer-dunker assisted dunk combo counts and report leaders.
- What went wrong: N/A.
- User correction: N/A.
- Fix applied: Parsed made-shot descriptions in `nba_pipeline/raw_data/NBA26.parquet` for `DUNK` plus `(... AST)` tags; used `player1_name/id` as dunker and parsed passer text from the assist tag.
- Prevention rule added: For event-level assist-pair tables, use raw PBP rows rather than possession-level `RIM_ASSIST`, because passer identity is only in description text.
- Outcome: Found `9,115` assisted dunk events; top pair was Cunningham-Duren with `71`.

### 2026-05-19 12:00 (local)
- Task: Answer Alex Caruso 2026 OKC regular-season on/off net rating.
- What went wrong: One OKC-HOU game (`22500581`) had corrupted lineup IDs that did not overlap OKC event players.
- User correction: N/A.
- Fix applied: Computed from `RAPM26.parquet` scoreboard deltas, classified OKC sides from raw team/player data, and manually inferred Caruso intervals for the corrupted-lineup game from substitution events.
- Prevention rule added: For one-off on/off lookups, validate team-side classification by game and inspect any zero-overlap lineup games before reporting.
- Outcome: Caruso on: `+18.65` net rating; off: `+8.63`; on-off swing `+10.01` across all 82 OKC RS games.

### 2026-05-18 17:55 (local)
- Task: Fix and rebuild the 1997-2013 historical playoff parquets after the RS rebuild skipped PS source files.
- What went wrong: The refreshed Gabriel source validation directory only had RS files, and existing production PS had all RAPM files but only 140/306 expected metric files plus same-player O/D overlap artifacts.
- User correction: User asked whether playoffs were fixed, then asked to fix it.
- Fix applied: Added `repair_existing_historical_lineup_sides.py`, taught `reprocess_metric.py` `--raw-dir` / `--processed-dir`, repaired existing PS raw files in validation, rebuilt all 18 PS metrics, backed up production, and promoted repaired raw/processed PS parquets.
- Prevention rule added: Treat historical PS as its own gate when the Gabriel source directory lacks PS files; validate production PS with full metric count plus O/D overlap checks before weighted-factor solves.
- Outcome: Production now has 306/306 PS processed files, `0` O/D overlap cell hits across PS metric files, RAPM_PS `239,952` rows, and RAPM partial-lineup rows reduced from `5,914` to `2,721`; backup is `validation/historical_ps_lineup_repair_1997_2013/production_backup_20260518_175307`.
