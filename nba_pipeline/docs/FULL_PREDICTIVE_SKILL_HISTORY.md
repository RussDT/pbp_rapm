# Full predictive stat history from 1996 to 2026

Owner authorized October 21, 2026: build the complete historical predictive-stat library, including new game-log families and learned relationships between stats. This is the upstream skills work in the existing VPM RESEARCH slot. It does not authorize a new impact mapper, selected-model replacement or 2027 outcome evaluation.

**Implementation is built and the corrected historical run was launched. Full training/results are pending until the runtime status and final receipts say complete.** The authoritative generated status is `results/estimated_skills/full_history_v2/run_status.json`. Do not infer completion from this document or from the existence of a few annual model files.

## Scope and data

- Thirty ending seasons, 1997–2026: November 1, 1996 through June 13, 2026, regular season and playoffs.
- 787,558 unique player/game rows, 2,939 player IDs; merged reports supply 356,035 rows from 2014 onward.
- 129 underlying stat concepts, represented by 166 model heads because 37 concepts require separate source-vintage models. These are not 166 independent VPM inputs or 166 new basketball abilities.
- Includes core shot-zone volume/accuracy, FT, assistedness, passing destinations, turnover composition, fouls, rebounds, scoring context, creation/tracking, defended shooting, rebounding opportunities, hustle, screen creation, box-outs and shooting/defender-distance context.
- Makes, points, aggregate shooting, assisted/unassisted splits and other summaries are coherent derived outputs, rather than redundant extra target models.
- Earlier unavailable tracking stays unavailable. A returned column filled with zeros is not sufficient evidence that a historical subtype was measured.

The full source tables were retrieved in about 18 seconds through read-only Cloudflare Catalog/Object GETs, using the existing Wrangler login. No Worker deployments or remote writes are involved. The named Iceberg table snapshots are pinned:

| Table | Snapshot | Rows | Data files |
|---|---:|---:|---:|
| `nba_pbp.player_game` |4512163488754738258|787558|2|
| `nba_pbp.player_game_report` |6265524001155048069|356035|3|

Before a data file is downloaded, its Iceberg season bounds must prove it stays inside 1997–2026. Missing bounds, future years and delete-file snapshots fail closed. Downloads check manifest record counts, byte counts and SHA256. The corrected package reuses these immutable downloaded files through its source symlink; do not refresh that symlink to a newer source.

Source/API discovery: [merged game-log audit](DATABALLR_GAME_LOG_SOURCE_AUDIT.md). Curated definitions: [model concepts and source expressions](../scripts/full_skill_history_specs.py), [versioned fields](../config/full_skill_history_sources.json).

## Source corrections made before the corrected run

The first short fit queue, `full_history_v1`, was stopped and quarantined after checking subtype totals. Its model outputs are not admitted.

1. **Turnover classifications change over time.** Offensive-foul turnovers are zero or nearly absent before 2018 despite thousands of recorded offensive fouls. Bad-pass out-of-bounds coding begins in 2016; lost-ball out-of-bounds and stepping out become usable in 2004. A raw zero in an unsupported era is not a true observation.
2. Model total turnovers and four stable coarse groups: all bad passes, all lost balls, traveling and the remainder. Their observed counts close exactly to total turnovers. Model finer OOB/offensive-foul/other detail only in its admitted source era. Do not substitute offensive foul counts for offensive-foul turnover labels.
3. Group personal, away-from-play and inbound fouls for a consistent coarse non-shooting personal category. Keep unsupported away-from-play observations after 2020 and inbound observations after 2018 out of their separate historical targets. Six other types plus the coarse personal group close to recorded all-foul counts. These are source-admission rules pending any fuller provider explanation, not a claim that the basketball events disappeared.
4. **Different tracking vintages have separate filters.** Explicit `_v1` and `_v2` fields never become one stitched time series. Modern v2 fields begin in 2025. Canonical display aliases use v1 through 2024 and v2 thereafter, without filling a missing v2 value from v1. Both source-specific forecast columns remain available for inspection.
5. Preserve FT reliability flags, per-version passing outages and missing defended-shot measurements. No filled/inferred measurement is used. 341 player-game rows have the documented individual passing-outage pattern; 1,803 unsafe-lineup rows cannot supply possession-rate denominators. Box shooting proportions may still use their independently observed attempts.
6. Two rows share the same player/date without an established tip-time order. Predict the entire same-date batch first, then update from its observed totals. Do not choose an arbitrary game-ID order that leaks one game's results into another's pregame state.

## Independent forecast baseline

Each target uses the existing two-state OU idea: a slowly changing level plus shorter-term form, with exposure-dependent observation uncertainty and a season-boundary variance term. The compiled kernel has been checked against the existing OU implementation.

Fit each annual parameter recipe using only completed earlier ending seasons. The objective is next-game exposure-weighted squared error; internal parameter fitting uses all valid earlier rows after a player's first observation. Annual exported forecasts are generated with the recipe available before their target season; do not backfill earlier forecasts with a later parameter vintage. Source-specific first years and the archive's initial year have explicit bootstrap priors rather than invented earlier training.

All available valid earlier observations feed replay; there is no one-year measurement truncation. The filter learns its level/form memory. Pregame query rows exist even when the target stat has zero opportunities or no measurement. Missing measurements advance time without a fabricated zero update. Same-date batches share the same prior state. Store pregame accumulated exposure and last-observation date separately from postgame values.

These are **context-free forecasts of game production under the observed role/source definitions**. They are not identified neutral causal ability estimates, and they do not silently replace the selected current-neutral library or the lab's previous context/age/debut recipes. Archive history before November 1996 is unavailable; first observed exposure is not automatically rookie/debut exposure.

The baseline fit has a fixed DE budget, seed 20261021, maxiter30/popsize10, frozen through the run. Save convergence and objective metadata per head/year. Do not call every optimizer run converged or interpret raw filter variance as calibrated full uncertainty.

## How relationships between stats are learned

The baseline filters remain separate. A second, regularized cross-stat layer predicts each target's remaining error from the entire vector of **pregame** baseline forecasts and their missing flags. FT shooting can therefore inform corner/above-break shooting, and creation or activity stats can inform other rates, when the historical data support the relationship.

This layer is entered through the existing [next-game lab](../scripts/research_skills_nextgame_lab.py), using `--full-history`; [implementation](../scripts/full_skill_cross_talk.py). It is weighted residual ridge with training-only standardization, a free intercept and penalties0.1/1/10. Older calibration rows decay with a fixed1460-day half-life. Coefficients and preprocessing use only older annual out-of-time forecast rows. They never see the current game's realized FT%, shot volume, minutes or other box counts as predictors.

Before each target season, choose the ridge setting versus the independent null using only older out-of-time per-season losses, with four-year decay and at least three prior scored seasons. Null wins ties. This is selection of a predictive cross-stat model, not proof that every nonzero coefficient contributes independently or that one stat causes another. Individual relationship claims still need ablations.

2026 is filtering/export-only. It contributes no cross-stat target fitting, penalty selection or error scores. All historical validation is retrospective/spent research evidence; no 2027 outcomes are loaded. The new layer has no completed joint-makes or downstream VPM admission claim. Save its independent baseline alongside the research candidate.

## Execution and expected runtime

Observed pilot: early-history volume fit14s; two 2026 fits44s/52s. The older17-stat/11-vintage package recorded2.67 CPU-hours for187 fits. The current first full pass was provisionally budgeted **3–6 hours wall time** on this machine with8 fit workers, including replay, cross-stat comparisons and exports. That estimate is not a completion promise; source checks, fit throughput and final validation can change it.

From the RAPM root:

```sh
# Tested scientific runtime (includes the new fastavro source reader).
uv pip install --python .venv/bin/python -r nba_pipeline/requirements_predictive_skills.txt

# Source acquisition on a fresh package; the running v2 reuses the pinned v1 source.
.venv/bin/python nba_pipeline/scripts/download_databallr_skill_history.py
.venv/bin/python nba_pipeline/scripts/prepare_full_skill_history.py

# Resume the identical immutable fit/replay recipe; verified checkpoints are reused.
.venv/bin/python nba_pipeline/scripts/build_full_skill_history.py --stage all --workers 8

# These stages are automatically run by the active finish supervisor after replay.
.venv/bin/python nba_pipeline/scripts/research_skills_nextgame_lab.py --full-history --workers 6
.venv/bin/python nba_pipeline/scripts/export_full_skill_history.py

.venv/bin/python -m pytest -q nba_pipeline/tests/test_full_skill_history.py
```

The active [finish supervisor](../scripts/finish_full_skill_history.py) watches only this build's fit PID, verifies all annual recipe locks, then runs [whole-panel chronology/source checks](../scripts/audit_full_skill_history.py), cross-stat testing and coherent export. It records `failed` on a missing stage or exception; it does not report success just because a process started. No publication or live daily data-refresh scheduler was installed. After the owner asked work to continue while away, a temporary hourly follow-up in this chat (`finish-predictive-stat-history`) was enabled to inspect failures/completion, verify results and update documentation; pause it after verified completion. This follow-up does not itself refresh sources or fit a daily live model.

Before advancing, the supervisor checks all 30 Parquet footers: a file appearing in the directory is not proof that its writer has finished. It then runs [historical results diagnostics](../scripts/summarize_full_skill_history.py) after export and hashes that receipt in the completion status. This results step does not change any saved model, cross-stat selection or VPM input.

## Results contract fixed before full-run scores

The October 21 finishing pass adds a fixed evaluation of 2020–2025. For FT%, corner/above-break three-point accuracy, turnovers and rebounds, report independent versus selected cross-stat predictions. Also test the coherent derived three-point makes, FT makes, points and assists per 100 possessions. Each comparison uses identical observed rows in both arms, exposure weighting within season and equal weight across the six seasons. Different target units never enter a pooled raw MSE.

Report paired game-cluster bootstrap intervals (1,000 draws, seed20261021), resampling games within each season. These intervals condition on saved annual filters, coefficients and model choices; they do not represent all parameter/selection uncertainty. This remains spent retrospective research evidence, not new confirmation. Derived forecast comparisons are diagnostics, not an automatic joint-makes or VPM admission gate.

The specific FT question has a matched diagnostic: calibrated own-zone pregame accuracy versus that same input plus pregame FT% and its missing flag. Both residual ridge arms use fixed alpha0.1 and a1460-day calibration half-life, train only on earlier annual out-of-time forecasts, and score only through2025. Save annual coefficients and2026 predictions without reading2026 target rows. A positive paired increment supports predictive association in this limited comparison; it neither establishes causation nor isolates FT's contribution within the full166-head model. No reported result is used to retune these arms.

The diagnostics also report fitted versus bootstrap annual slots and actual optimizer convergence separately. A fixed-budget forecast need not be a converged optimum. Head/family comparison CSVs retain the number of scored seasons per head, including short-history source-vintage branches.

## Outputs and checks

Generated package: `nba_pipeline/results/estimated_skills/full_history_v2/` (ignored):

- `prepare_receipt.json`, `coverage.csv`, `registry.json`: admitted sources, measurements, families/eras and missingness.
- `source_validation.json`: positive/unique identities across37,986 games and exact raw point/assist-zone/turnover accounting; `chronology_audit.json` is written only after all annual forecast/date/epoch/bounds gates pass.
- `run_lock.json`, `models/<year>/<head>.json`: immutable recipe and annual fit/date/convergence receipts.
- `baseline/<year>.parquet`: pregame baseline states plus explicitly named `__post`, prior exposure and last-observation fields. `seconds_pbp` here is retrospective target metadata, not a pregame predictor.
- `cross_stat/`: past-only choices, coefficients, historical scores and predictions; no2026 scores.
- On completion, `library/independent/` and `library/cross_stat_research/`: pregame-only stat views and derived accounting, plus `library_receipt.json` and `latest_2026_pregame_values.csv`. Latest values carry each player's actual game cutoff; they are not silently relabeled as today's roster or states.
- `run_status.json`: current stage and completion/failure status. Logs: `fit_replay.log`, `cross_stat_testing.log`, `coherent_exports_and_audits.log` and `supervisor.log`.
- `diagnostics/RESULTS.md`, `research_summary.json`, head/family comparison CSVs and FT-association coefficients: historical results, paired conditional intervals and the fixed FT diagnostic. The completion status hashes this summary; until it exists, these results are pending.

Twelve focused tests cover target/future perturbation invariance; missing versus observed zero; same-day batching; source epoch gates; tracking-vintage separation; agreement with the existing filter; manifest season fences; direct/moment ridge equivalence; chronological selection; an end-to-end synthetic FT-to-three example that is invariant to2026 target changes; and bounded makes/point/rebound accounting.

Five additional finishing tests verify footer readiness, paired game aggregation, equal-season weighting, refusal to materialize2026 target rows and the matched FT diagnostic's chronology/2026 invariance. Combined focused suite:17 tests pass before final results.

Early real-data probe: full166-head replay for ending1997 and2018 covers53,011 player-game rows. Exact source identity, strictly earlier last-observation dates, forecast bounds and unsupported-source epochs pass; coherent point accounting has maximum gap0 in both seasons. This partial probe does not replace the final30-season audit. A probe initially requested still-fitting2019; it stopped on a missing annual recipe, and2018 was checked instead. The main fit queue was unaffected.

Raw independent proportion estimates are preserved when rebound/turnover compositions are normalized for coherent reporting. Head-error summaries refer to forecasts before deterministic products/composition normalization; they do not certify every derived output's accuracy. No balancing residual repairs broken shot/point accounting. Source provenance and shared aliases do not establish a public deployment or a new selected VPM model. Selected clean v61 and separate v65 history remain intact.
