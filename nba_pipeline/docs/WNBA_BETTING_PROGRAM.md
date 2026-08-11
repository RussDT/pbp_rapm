# WNBA Predictive Program (Plan of Record)

Adopted August 10, 2026. This formalizes the strategy synthesized on
August 6, 2026 (nine-agent deep-read of this repo plus `unified-rapm/`,
three independent strategist lenses). Until this document is superseded,
WNBA work in this repository executes against this plan.

## Core thesis

The WNBA program is the business; the NBA program is the R&D lab. We hold a
data monopoly — full Second Spectrum/RTSS tracking 2023-2026 (991 games local
with paired 2D and 3D markings), exact five-player chance lineups, and
creation-role labels — in the softest major market. The live WNBA box-prior
rating is already statistically indistinguishable from DARKO NPI on held-out
2026 game margins (13.118 vs 13.037 RMSE; CI spans zero). The measured
headroom is not in rating-shape changes (the NBA meta-finding: on a mature
regularized RAPM almost nothing moves out-of-sample prediction); it is in
evaluation surface, minutes/availability, coverage, calibration, and
market-facing products.

## Standing fences

- **Ratings stay market-blind.** The rating pipeline never sees betting
  lines. The wagering layer alone is market-aware.
- **2026 is the designated confirmation season** for the RTSS component-prior
  candidates. No 2026-informed selection anywhere: not alphas, not features,
  not scales. The one-shot confirmation fires once, at season end.
- **Proprietary comparison ratings (DPM/PI/NPI) remain benchmarks only** —
  never priors, features, targets, or tuning inputs.
- **No credentials or signed URLs in repo artifacts.**
- For bet-facing models (forecast log, minutes, sigma, props): ship on
  development evidence with pre-registered freezes and let live scoring
  adjudicate. The heavyweight one-shot confirmation discipline binds rating
  promotions only.

## Workstreams

### 1. The market is the holdout (Phase 0)

Chronic problem: every season is spent — 2023-2025 RTSS, the 2025 holdout,
partially 2026. The closing line is a fresh, unspendable holdout delivered
daily. Stand up a frozen pre-tip forecast log during the live 2026 season:
every game day, emit margin/win distributions from the current frozen metric
(zero model changes), capture market open/close when an odds source is
available, and score log-loss, calibration (PIT/coverage), and closing-line
value. Implementation: `wnba_test/emit_wnba_pretip_forecasts.py`,
`wnba_test/capture_wnba_odds.py`, `wnba_test/score_wnba_forecasts.py`;
operational doc `docs/WNBA_FORECAST_LOG.md`.

### 2. Minutes, availability, rotation (Phase 1)

The actual alpha engine. Books are slow to reprice second-order consequences
of news — not who is out, but who absorbs the 200 team minutes. Build the
game-day allocator: carry-forward minutes-when-active plus short-half-life
form, with scratch redistribution driven by observed `chance_players` stint
co-occurrence rather than depth-chart guesses. Evaluate against naive
baselines on 2025 and 2026-to-date MAE. Implementation:
`wnba_test/project_wnba_minutes.py`.

### 3. Promote RTSS through the frozen gate (Phase 2)

The candidate list is frozen from cross-season CV (see
`docs/WNBA_RTSS_CONFIRMATION_PREREG.md`, amended August 10, 2026). When the
2026 season completes, run the guarded harness once: the primary endpoint is
the strictly forward fold 2025->2026 (components solved on 2025 around each
prior arm, corrections fit on 2023-2024 only, scored on 2026 possessions);
the in-sample 2026 nowcast is a secondary descriptive diagnostic. Promote
passers into the 2027 live prior as cross-fitted residual corrections on the
box backbone.
Never monolithic RTSS; RIM/THREE finishing stay box-only; `qSP-qSQ` stays
quarantined behind a season-relative transform. Extend the defense alpha grid
past 40000 in the same cycle using development folds only.

### 4. Player props are the primary product (Phase 3)

A net-rating shop cannot price props; a component shop can. `assistOpp`
creation labels, exact on-ball time, passer shot value (R2 0.597 from on-ball
time plus potential assists), per-component frequency priors, and the additive
decomposition map one-to-one onto PTS/REB/AST prop distributions.
Role-conditional distributional projections (count models conditioned on the
minutes simulation) on lineup-change days are the widest per-bet edge.
Develop on 2023-2025 (spent — that is development), confirm on 2026 shadow,
live 2027. Preregistration: `docs/WNBA_PROPS_PREREG.md`.

### 5. Coverage and cold-start (Phase 3)

The live metric's top gap: 20-23 percent of 2026 player-slots unrated
(scored league-average), about 10 percent of possessions fully covered,
versus about 99 percent coverage for DARKO. The gap coincides with the
market's softest windows (openers, debuts, call-ups, 2026-27 expansion) and
creates adverse selection for any edge-thresholded selector. Builds:
hierarchical NBA-to-WNBA partial pooling for box-to-component slopes, and an
NCAA-W-to-WNBA translation lane for cold-start uncertainty. Until shipped:
hard no-bet filter on low-coverage games. Preregistration:
`docs/WNBA_COLD_START_PREREG.md`.

### 6. Integrity and calibration before any real stake (Phase 1)

A backtest is capital. Known failure modes to fix first: interval
over-confidence (nominal 90 percent showed actual 0.74-0.79 coverage in the
unified-rapm audit), as-of leaks, closure test failures, hardcoded paths.
Recalibrate sigma per market on development seasons (2022-2025 only), verify
PIT uniformity on the live log, quarter-Kelly cap, pre-registered bet
triggers, fail-closed gates (stale-data, closure, coverage asserts) in front
of every wager. Implementation: `wnba_test/calibrate_wnba_forecast_sigma.py`.
The calibrated scalar margin sigma is explicitly a **v1 placeholder**: it
makes the win-probability map honest on average but prices every game with
the same distribution width. The conditional-sigma replacement is workstream
7. The RTSS S3 automation still needs a least-privilege AWS profile; the
provider dependency needs graceful box-only degradation.

### 7. Distribution engine (Phase 1.5)

The panel-selected highest-alpha lane most books and modelers skip: stop
selling the mean, start selling the distribution. Books derive totals, team
totals, alt lines, and moneylines from a mean plus a stale variance
template; a game-level simulator with **conditional variance** (pace level,
projected-total level, rotation shortening, blowout dynamics) reprices every
derivative from one distribution and harvests the conversion errors — the
mean edge can be zero while the derivative edge is real. Planned build:
`wnba_test/simulate_wnba_game.py` (possession/segment-level simulator
conditioned on the minutes engine and Kalman rates) and
`wnba_test/price_wnba_derivatives.py`, doc `docs/WNBA_DERIVATIVE_PRICING.md`.
Kill test before any real build-out: two development days of conditional-vs
scalar-sigma CRPS on 2022-2025 finals; if conditional sigma cannot beat the
scalar on development CRPS, the lane dies cheaply. Scoring hook already
exists: the forecast scorer reports CRPS-vs-realized next to close-CLV so
distribution quality is measured in outcome space from day one.

## Sequencing

| Phase | When | What |
| --- | --- | --- |
| 0 | Now | Odds ingestion + frozen pre-tip forecast logger on the existing daily update. Zero model changes. Every remaining 2026 game becomes free evaluation. |
| 1 | Now to +8 weeks | Minutes/allocation engine; sigma/calibration repair on 2022-2025; integrity fixes; paper-trade from the log. |
| 1.5 | After the conditional-sigma kill test passes | Game simulator + derivative repricing (totals, team totals, alt lines, ML conversion) from one conditional distribution. |
| 2 | 2026 season end | One-shot frozen RTSS confirmation on 2026 -> promote passers into the 2027 live prior; extend defense alpha grid (dev folds only). |
| 3 | 2026-27 offseason | Props model (develop 2023-2025, confirm 2026 shadow); NBA-to-WNBA + NCAA-W cold-start priors, live before 2027 debuts. |
| 4 | 2027 season | Full-stack shadow book; staking gate = pre-registered CLV threshold (300+ logged bets, bootstrap CI excluding zero) AND calibration in bounds; then scale in quarter-Kelly. No in-season retuning. |

## Explicit non-goals

- **Chasing NBA sides.** The NBA lane's value here is indirect: transfer
  priors, evaluation templates, translation machinery.
- **Tuning more component alphas for game prediction.** The factor-specific
  alpha candidate won development and lost 2026 confirmation; solve one
  regularized net for the betting carrier and keep the decomposition for
  prior-injection and props role structure.
- **Letting the no-forward-projection convention bind the betting stack.**
  A bet on Tuesday's game is a forecast of Tuesday. Keep as-of causal
  discipline and as-of boards, but bet drift-corrected, minutes-simulated,
  aging-aware numbers.

## Honest risk

WNBA limits are small and winners get restricted; the program can be
genuinely CLV-positive with a modest dollar ceiling. Mitigation is breadth
(props, derivatives, books). The fast-feedback metric is CLV, not short-run
PnL.

## Execution ledger

| Item | Status | Where |
| --- | --- | --- |
| 2026 RTSS full sync (through Aug 6) | Done | `downloads/rtss-data/wnba/` (991 games) |
| Follow-up RTSS syncs (~101 remaining 2026 games) | Blocked on AWS reauth (`aws login`) | `docs/WNBA_RTSS_FEATURE_AUDIT.md` |
| Game-log panel (finals table + player logs) | Built (August 10, 2026) | `wnba_test/build_wnba_game_logs.py` |
| Phase 0 forecast logger + odds capture + scorer | Built (August 10, 2026); operational doc `docs/WNBA_FORECAST_LOG.md` | `wnba_test/emit_wnba_pretip_forecasts.py`, `capture_wnba_odds.py`, `score_wnba_forecasts.py` |
| Phase 0 retrospective 2026 pregame replay | Done — indistinguishable from the DARKO-NPI benchmark on the same games | `wnba_test/calibrate_wnba_forecast_sigma.py replay` |
| Phase 1 minutes engine | Built — beats carry-forward and trailing baselines on 2025 dev and 2026 | `wnba_test/project_wnba_minutes.py` |
| Phase 1 sigma calibration (v1 scalar) | Built — `normal_std` selected on 2022-2025 dev; scalar sigma is the workstream-7 replacement target | `wnba_test/calibrate_wnba_forecast_sigma.py` |
| Predictive box panel (Kalman + DE, DARKO-style) | Built — all 14 stats beat trailing-10 on dev and frozen 2026 holdout; doc `docs/WNBA_KALMAN_BOX.md` | `wnba_test/build_wnba_kalman_box.py` |
| Phase 2 confirmation prereg + guarded harness | Frozen + amended (design review) August 10, 2026; `--check` correctly BLOCKED until season end | `docs/WNBA_RTSS_CONFIRMATION_PREREG.md`, `wnba_test/run_wnba_rtss_confirmation.py` |
| Phase 2 defense-alpha grid extension (dev folds) | Done — 40000 retained; degradation is monotonic through 160000 | `wnba_test/extend_wnba_defense_alpha_grid.py` |
| Phase 3 preregs (props, cold-start) | Frozen August 10, 2026 | `docs/WNBA_PROPS_PREREG.md`, `docs/WNBA_COLD_START_PREREG.md` |
| Phase 1.5 distribution engine | Not started — gated on the conditional-sigma CRPS kill test | workstream 7 |
| Offline lane test suite | Green | `nba_pipeline/tests/test_wnba_gameday_lane.py` |
