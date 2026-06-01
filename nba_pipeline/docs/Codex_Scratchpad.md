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

### 2026-05-31 18:33 (local)
- Task: Consolidate the repo back onto `main` so the rescue branch is no longer the active branch.
- What went wrong: The repo had one pushed rescue commit ahead of stale `main`, plus a large uncommitted source/docs layer and generated local artifacts.
- User correction: User asked to bring everything back into `main` and just have a main branch.
- Fix applied: Staged source/docs/curated research files, ignored generated downloads/thumbnails/Randle output browsers, validated with compileall and `pytest nba_pipeline/tests`, and prepared the consolidation commit for `main`.
- Prevention rule added: Before branch cleanup, separate source/docs from generated artifacts so `main` can be clean without losing local generated outputs.
- Outcome: Consolidation is ready to commit and push to `main`; generated artifacts remain on disk but out of git.

### 2026-05-30 23:00 (local)
- Task: Grid-search optimal td700 true-prior alphas for the active 22-26 RS Alt3 EFG-value component solves.
- What went wrong: N/A.
- User correction: User clarified that `oEFG/dEFG` should be understood as the first-chance EFG-value display parent, not legacy accounting `ALT_EFG`.
- Fix applied: Ran `rapm.py --cv-alpha` with grid `1500,2000,2500,3000,5000,6500,8000,10000`, `--season-effects`, `--timedecay --half-life 700`, and factor-specific priors from `draft_shared/rapm/russell/results/alt3_efg_value_priors_22_26_rs_se_a2000_2000`, capped at two concurrent jobs / one core each.
- Prevention rule added: For active Decomp RAPM alpha tuning, map public `oEFG/dEFG` to `ALT_EFG_VALUE` / `oALT_EFG_VALUE` / `dALT_EFG_VALUE`; cite the validation summary before treating high-edge alpha wins as durable methodology.
- Outcome: Summary written to `nba_pipeline/validation/alpha_grid_22_26_rs_td700_20260530_215955/alpha_grid_summary.{csv,md}`; high-level winners were EFG `10000/10000`, TOV `6500/8000`, FT `2000/3000`, SC `10000/10000`, and defensive 3P/rim/mid FG all `10000/10000`.

### 2026-05-30 11:50 (local)
- Task: Rewrite the RAPM decomposition paper as a public arXiv-style RAPM DECOMP paper without internal Alt3 EFG naming.
- What went wrong: The older draft still used internal naming and claimed a stale 2024-2026 parent row-algebra check that current local processed files no longer support; the first rewrite also over-mentioned `oDISPLAY_SUM`, making the paper sound like the display sum was the metric rather than an audit helper.
- User correction: User specified this should be the RAPM DECOMP paper, should not refer to the method as Alt3 EFG, should frame the key claim as actual `off` / ORAPM decomposing nearly perfectly into the factors, should use public `oEFG` / `dEFG` labels rather than `oShot` / `dShot`, and wanted player value shapes as separate eight-bar vertical charts rather than dense player tables.
- Fix applied: Replaced `nba_pipeline/docs/papers/additive_rapm_decomposition.tex` with a full public RAPM DECOMP draft, regenerated the HTML companion, refreshed empirical tables from current public exports, removed `DISPLAY_SUM` from the paper, switched table/equation labels to `oEFG` / `dEFG`, added factor-correlation heatmaps, added separate offense-left defense-right value-shape bar charts for Jokic, Shai, and Wembanyama, clarified that public EFG is baseline plus six shot atoms, and updated the docs index.
- Prevention rule added: Public papers/explainers should use RAPM DECOMP language and cite current public residuals as `actual RAPM - factor sum`; avoid foregrounding internal audit columns like `oDISPLAY_SUM`.
- Outcome: The paper source and HTML contain no `alt3` / `EFG-value` / `DISPLAY_SUM` naming and report current near-zero public residuals.

### 2026-05-30 05:11 (local)
- Task: Create a standalone HTML explainer for the active Alt3 EFG-value weighted-factor additive decomposition.
- What went wrong: N/A.
- User correction: N/A.
- Fix applied: Added `nba_pipeline/docs/alt3_efg_weighted_factor_explainer.html` with context, decomposition identities, ridge-matrix explanation, sign conventions, design decisions, and teaching framing; added it to the docs README.
- Prevention rule added: For intern-facing Alt3 explanations, start from row-level possession targets and identities before discussing player coefficients or public columns.
- Outcome: The explainer can be opened directly from the docs folder without a dev server.

### 2026-05-29 18:42 (local)
- Task: Create an elegant HyperFrames explainer video for the active clean Decomp RAPM component structure without using internal Alt3 weighted-factor naming.
- What went wrong: The first multi-worker render failed during FFmpeg encoding because the workstation reported very low free memory, and the animation-map helper could not resolve a bundled HyperFrames version even after the producer package was installed.
- User correction: N/A.
- Fix applied: Built a standalone HyperFrames project at `nba_pipeline/docs/videos/decomp-rapm/`, added local design/prompt files, embedded local font assets, passed lint/validate/inspect, and rendered the MP4 with `--workers 1 --quality draft`.
- Prevention rule added: For HyperFrames renders on this workstation when memory is low, prefer single-worker streaming encode; animation-map may need a pinned-version helper workaround.
- Outcome: Rendered `nba_pipeline/docs/videos/decomp-rapm/renders/decomp-rapm.mp4` as a 40-second 1920x1080 H.264 explainer.
