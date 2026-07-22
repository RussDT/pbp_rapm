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

### 2026-07-22 02:00 (local)
- Task: Report Jokić and Shai from the fresh and published 2022-26 ALL RB+SE DECOMP files.
- Outcome: Fresh Jokić is `8.907 / 1.505 / 10.412`; fresh Shai is `7.159 / 1.267 / 8.426` for offense / positive-good defense / net, closely matching publication.

### 2026-07-22 01:42 (local)
- Task: Promote the 2014-2026 lineup repairs, rebuild all processed metrics, and compare a fresh 2022-26 ALL RB+SE DECOMP with the owned `rapms` publication.
- What went wrong: The 2024-25 licensed ShotQuality enrichment inputs were absent, the rebuild registry omitted `INITIAL_EV_BETA` / `EV_RAPM`, the master runner invoked a dependency-less global Python, and the local venv lacked `statsmodels`.
- User correction: User requested a full modern processed rebuild and an empirical five-year publication check.
- Fix applied: Promoted/re-audited the repaired raws, rebuilt the canonical 6.3M-shot repo-owned model, rebuilt 644 locally supported parquets, made targeted rebuilds fail-fast, added the missing registry entries, fixed interpreter inheritance, assembled a fresh 97-column DECOMP, and wrote validation/comparison reports.
- Prevention rule added: Fail bulk rebuilds on partial output, use the orchestrator's interpreter for subprocesses, validate `SHOOTER_OREB` with its shooter-plus-four schema, and never synthesize missing licensed SQ enrichment.
- Outcome: 66,359,329 processed rows have zero invalid saved lineups; FIRST_CHANCE row algebra is exact to `5.69e-14`; fresh vs published 2022-26 net RAPM correlates `0.99875`, with identical top 10. LeBron is `off=2.523`, `def=+0.128` (positive-good display), `net=2.651`.

### 2026-07-21 14:35 (local)
- Task: Diagnose and repair modern 2014+ raw lineup corruption, including the LeBron-affected 2016-17 game.
- What went wrong: GameRotation responses/archive rows can be incomplete, overlapping, or reversed, while event-number-first sorting also moves corrected V3 actions backward in time.
- User correction: User supplied Gabriel's rotation archive as optional reattachment evidence.
- Fix applied: Added `repair_modern_raw_lineups.py`, constrained period/substitution replay with actor/rotation/raw evidence, safe validation/promotion controls, clock-first ordering in shared consumers, focused tests, and operator docs.
- Prevention rule added: Treat GameRotation as positive evidence only and require clock-first PBP ordering plus zero lineup/substitution/actor issues before promotion.
- Outcome: Full 2014-2026 RS+PS validation audited 7,784,893 rows and repaired 10,623 bad rows across 31 games to zero remaining issues; production raw files were not promoted.

### 2026-07-21 13:43 (local)
- Task: Run 2012-2016 all-games RAPM with rubberband and season effects and report LeBron James.
- What went wrong: The local `.venv` is Python 3.9 while `rapm.py` contains Python 3.10 union annotations; multiprocessing workers also cannot inherit an in-memory postponed-annotation wrapper.
- User correction: N/A.
- Fix applied: Ran the unchanged solver through postponed annotation compilation with sequential parquet loading.
- Prevention rule added: Use a Python 3.10+ project environment for normal `rapm.py` CLI execution; sequential loading is a safe compatibility fallback.
- Outcome: Wrote `rapm_12_16_all_rb_se_results.csv`; LeBron is first at `off=7.658`, raw `def=-2.578`, and `net_rapm=10.236` across 66,696 possession appearances.

### 2026-07-21 (local)
- Task: Download and assemble the shared Drive data handoffs, then rebuild the canonical 2014-2016 processed surface.
- What went wrong: The modern folder was initially described as processed data, but its README identifies it as raw NBAXX PBP only; the Gabriel link is a data TAR, not repository source.
- User correction: User supplied the canonical Drive links and asked to download everything and assemble it.
- Fix applied: Verified all 27 split-ZIP parts and the full ZIP hash, extracted modern 2014-2026 RS/PS raw files, extracted the historical 1997-2013 handoff, created an isolated Python environment, and used `reprocess_metric.py` with six workers to rebuild 18 canonical metrics for 2014-2016 RS+PS using the Supabase service-role environment.
- Prevention rule added: Read the handoff README and inspect archive manifests before classifying or extracting shared artifacts.
- Outcome: Local workspace has 60 raw RS/PS parquets for 1997-2026, the historical processed/results handoff, and all 108 expected 2014-2016 processed parquets; the new files are readable, nonempty, and contain 11,154,550 total rows.
