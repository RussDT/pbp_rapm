# Codex Scratchpad

Short-term session memory. Durable rules belong in AGENTS.md / CLAUDE.md.

## 2026-11-23 — Scoring Impact + rZTOV build daily for 2026-27

- `scripts/run_scoring_rztov_daily.py` builds both boards for the season in progress with the fit frozen at 2026 and promotes to `results/scoring_rztov_daily/2027/current` only when the 42-check audit and the frozen-rows check pass. First build `20261123T213052Z-g247`: 389 current-season rows, rows through 2026 identical to the frozen artifacts. Canonical `SCORING_RZTOV_DAILY.md`.
- Still open: unscheduled, nothing published, `--solves run` untested, and a live failing run not yet observed. The Cloudflare job replaces this runner (plan doc and databallr issues #2559-#2562).

## 2026-11-09 — Full-feature VPM now runs nightly

- `scripts/vpm_daily/` builds and publishes the R2-fed VPM Preview unattended (launchd `com.russellthomas.vpm-daily`, 7:30 AM). Nov 3 replay was byte-identical; Nov 9 published with 146 games/641 players. Canonical `VPM_DAILY_RUNNER.md`.
- Still open: first unattended run unverified, read-only R2 token, a no-games night is untested, and this work is uncommitted.

## 2026-11-09 — Nightly TD550 publish verification and rim copy names

- `publish_no_prior_vpm_snapshots.py` now verifies row counts only on the snapshot dates it published. The whole-table check could never pass because each day's rolling as-of date is pruned locally but stays remote; consumers read the latest date, so nothing was deleted. Manual rerun: 35,684 rows across 29 dates verified, latest 2026-11-08.
- `daily_rapm_update.sh` now maps `rim_freq`/`rim_fg_pct`/`midrange_freq` TD700 outputs onto the published `rimfreq`/`rimfgpct`/`midrangefreq` names. Still open: no 2027 ShotQuality source locally (`nba_games` holds only 2025, and `SQ_BASE` is an old-machine path), selected v61/pregame VPM unwired, and the job prints completed despite step failures.
- Nightly downstream pushes are on: launchd sets `NBA_RAPM_PUSH=1`, `NBA_RAPM_CSVS_DIR=data/csvs`, `NBA_RAPM_RAPMS_DIR=nba/research/rapms`. csvs goes through core `publish_csvs.sh` (test-branch, regression guard, checkout untouched); rapms pulls ff-only, then commits and pushes main. A dry run published 0 paths because the guard held back 2,017 stale local files; the first real push is the next 6 AM run.

## 2026-10-26 — Full-feature VPM current board independently audited

- The 453-date four-arm development replay is complete and independently audited (`c6d03cf1`); the separate selection certificate (`4ac7e6f6`) freezes `full_split_raw` under literal equal-season game MSE. Its RMSE 13.427847 versus total-raw 13.431220 has an unresolved paired game interval; no predictive-win claim. Exactly one 2026-10-01 mapper is fitted and independently audited (`e92973c8`); its separate current board passes full independent source/native/prior/Net/SQ/persisted168-column/20-array and fresh Parent/SQ normal checks (`873fc59d`), covering 2,856 players and 37,845 earlier native games. Current preview comes before full history, with O/D/Total and own priors compared against OLD v74 at the identical date.
- Canonical `nba_pipeline/docs/VPM_FULL_FEATURE_HISTORY_REBUILD.md` and readiness evidence govern continuation. Preserve all old successful/failed roots, original resource-failure log/capsule and auditor versions; the same-v2 recovery completed without source/model/producer/scoring/tolerance changes. Runtime reserve remains 15 GiB. Spent four-release fits/diagnostics, annual/native adapters, storage budget, 19 early-release fits and the 6,108-date replacement remain unfinished. Reuse the current2026 mapper: donor queue `1501fc0` / review `ddc817c0` requires 3 development, 4 spent and 1 current audited donor plus separate exact plan approval; fit only 19 releases (2000–2018). No donor-history plan or fit has run. The old 20-fit history queue stays immutable and unauthorized. Existing reader8800 remains OLD v74; selected v61 pointer, servers8798/8799/8800, 2027 and public/live wiring remain unchanged.

## 2026-10-24 — Daily VPM history verified; annual-trained and updated daily

- Canonical VPM_DAILY_1997_HISTORY.md; v74/source-adaptedv73 completes6108dates/9416143rows/2856players with9507828prior-observed featurequeries/37845admittednativegames, zero newfilters/mappers. All30annual endpoints reproduce1.128e-10; all-date source/nativeprefix/prior dot/11head/TD/center/intercept algebra and11fresh literal native+SQ batchcuts pass. Ownpriorsgap0;ratinggap3.553e-15;freshnormal8.915e-15. Numericalaudit169ea2e3/workflow64611d0c/source9a295a78; frozenpendingmarker preserved with separate report/certificate closure, no healthy/completedjob restart.
- Reader8800/coherentv2 fullSQL9416143float64rows exactlysource; finalconsumer39bb4b1e. Measured slow-status/date-picker race fixed in separately versioned consumer only, oldreader/source preserved;11readerchecks pass.742markedbackcastdates=741estimated+empty1996-11-01;401574estimatedbackcastrows. Uniformown-only/unknownbackgroundNaN, held originalannualreport parameter/model releases, observed-only2016+SQ.38of2504upstreamfits converge (2476bootstraps), no optimal-decay/forecast-win/live/public/2027/pointerchange. Retainselectedv61; history is not same-game daily-trained experiment. Temporarycontinuation pauses after final independent report/certificate verification.
