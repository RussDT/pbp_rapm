# Codex Scratchpad

Short-term session memory. Durable rules belong in AGENTS.md / CLAUDE.md.

## 2026-11-25 — Nightly VPM fixed: node refused under ~/Documents under launchd

- Nov 20-25 runs all died at `wrangler auth token`. The captured stderr showed node itself got `EPERM` opening `wrangler.js` below `~/Documents`, while bash and Python could read the checkout. `databallr_r2_read.py` now runs wrangler from a copied pnpm closure in `~/Library/Caches/rapm-databallr-wrangler/`, and any wrangler failure now puts its redacted stderr in the nightly log. A second problem was a stall: `api.cloudflare.com` resolves IPv6-first and this network has no IPv6 egress. R2 GETs now give each address 10 s to connect and alternate families. Canonical `VPM_DAILY_RUNNER.md`; tests `tests/test_databallr_r2_read.py`. Uncommitted.

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

## 2026-10-30 — Net SQ td550 nightly written (not installed)

- Nothing in this repo was ever scheduled: no launchd plist, no caller for `build_pbp_shot_quality.py`, `run_net_sq_decomp_rolling.py` or `publish_td550_rolling_archive.py`. That one gap is why the three surfaces are stale at three different dates (shot quality Jul 22, net-lane targets Sep 6, archive end year 2026) while `raw_data/NBA27.parquet` landed Oct 26. `scripts/nightly_net_sq_td550_refresh.sh` closes it with one mtime gate per stage, so a tick costs a few `stat()` calls and runs nothing unless an upstream moved.
- The Supabase publish defaults OFF and needs `PUBLISH=1`: `nba_no_prior_vpm_td550_rolling` is live and feeds the site. Its stamp is the sha256 of the staged `sources.json`, written only after the publish returns 0, and any failure in A-D skips the publish — a failed C skips D, which leaves the windows unresolved, so validating over a half-built season must not reach the table. Not installed; the launchd step is in the doc.
- Correcting yesterday's framing: the 2027 chain is NOT blocked upstream. `build_pbp_shot_quality.py --years 2027` parses `NBA27.parquet` cleanly (9,298 shots, ~2s); the table is just three months stale. Build it beside the 2026-named artifact — `--output-name` defaults to `pbp_shot_quality_all`, so the old name was explicit and a rerun cannot clobber it. 9,298 of 6,327,218 shots is 0.15%, so measure the EB-prior drift on the overlap rather than assuming it. The single real blocker is `clean_sq.protected_seasons = [2027]`; the nightly gates on config coverage and reports BLOCKED instead of editing a research config past a protected season.
- `NIGHTLY_STUB=1` logs each stage's command instead of running it. Ten gate cases verified that way: quiet tick runs nothing, each upstream touch cascades, a blocked B skips C and D, a failed stage records no stamp and exits 1, and an unchanged archive is not republished.
