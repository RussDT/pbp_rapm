# NBA Estimated Skills (PBP Kalman Panel)

Owner objective correction: estimated activity skills should optimize
**next-game per-100-possession accuracy**. The deflections implementation is
documented in [NBA_DEFLECTIONS_NEXT_GAME.md](./NBA_DEFLECTIONS_NEXT_GAME.md).
Remaining-season tests are secondary. Existing 17-skill historical v3 results
and files are preserved; they have not silently been retuned for this objective.

This doc is the PBP panel's receipt **and** the index of every other NBA
estimated-skills lane (hustle/tracking game logs, rim defense, shooting
priors, comparison tables): see the
[family map](#nba-estimated-skills-family-map-2026-09-14-every-lane-and-what-the-lab-says-about-each)
at the end.

September 12 deflections extension: [estimated deflections](./NBA_ESTIMATED_DEFLECTIONS.md) uses the v3-style OU core plus a separately tuned random-walk control, with 2020–2022 tuning and 2023–2025 historical stress. See that receipt for its independent selection and export status.

September 12 preparation update: the separate
[tracking game-log backfill](./NBA_TRACKING_BACKFILL.md) now audits and repairs
potential-assist and DFGA gaps in Gabriel's downloaded `player_sheets` archive.
It preserves observations, distinguishes DFGA omitted zeros from outages, and
labels every estimate. It does not add these skills to v3 or replace its exports.

Built August 26, 2026. E-Skills/DARKO-style walk-forward predictive player
rates for the NBA, mirroring the WNBA lane
([WNBA_ESTIMATED_SKILLS.md](./WNBA_ESTIMATED_SKILLS.md) /
[WNBA_KALMAN_BOX.md](./WNBA_KALMAN_BOX.md)) on the repo's own raw PBP and
processed possession surfaces, seasons 2018-19 through 2025-26 (RS+PS).
Implementation: `scripts/build_nba_player_game_lines.py`,
`scripts/build_nba_estimated_skills.py`. Outputs live under
`results/estimated_skills/` (generated artifacts, not in git).

## Data: per-game PBP lines

`build_nba_player_game_lines.py` extracts one row per (game, player) from
`raw_data/NBA{YY}[_PS].parquet` — clock-first ordering, 10-char `game_id`
normalization, exact-row dedupe, and the shared `RIM_ACTION_TYPES` + rim
description patterns, all via read-only reuse of
`scripts/process_rapm_blocks/common.py`. Fetched raw v2 PBP has
`event_action_type` all zero and `player2_*`/`player3_*` all null, so
secondary attribution comes from cumulative description counters —
`(Name N AST)`, `(N STL)`, `(N BLK)`, `(Off:X Def:Y)`, `(Pn.Tn)`, `(N PTS)`
— resolved to player ids with per-game display-name anchors plus season
surname maps and initial-prefix elimination (PBP disambiguates same-surname
teammates with variable-length initials, e.g. "Ja. Green"/"Je. Green").
On-court exposure joins the processed `RAPM{YY}[_PS].parquet` lineup melt:
`off_poss`/`def_poss` per player-game.

Validation (`lines_reconciliation.json`), all 16 season-files 2019-2026:

- Team-score identity: summed attributed player points equal the final
  scoreboard for **100.0000%** of team-games in every season, and per-player
  points match the `(N PTS)` running tag at 100.0000%.
- Counter-vs-event agreement for REB/AST/STL/BLK: 100.0000% everywhere
  except one 99.98% BLK cell (2026 RS).
- Possession-join coverage of active rows (minutes >= 0.5 or any counting
  stat): 0.99919-1.00000 per season. The residual zero-exposure rows are
  coach/bench-staff technicals (e.g. person id 873 = Gregg Popovich) and
  similar admin attributions carrying zero stats; the panel's exposure gate
  drops them (237,292 extracted rows -> 217,051 panel rows, 1,279 players).
- Star-total spot checks vs official 2023-24 values are exact: Doncic 2,370
  PTS/70 GP, Haliburton 752 AST, Jokic 708 AST, Sabonis 1,120 REB, Fox 150
  STL, Wembanyama 254 BLK.
- `pf_committed` (max `P` in the `(Pn.Tn)` ledger) intentionally ignores
  technicals, so the pf-vs-foul-rows share sits at ~0.95 by design.

### Offensive fouls drawn: resolved through the CDN live feed (September 14, 2026)

The requested `off_fouls_drawn_rate` (with charges subset) cannot be
attributed from the stats-feed PBP surface, and per the build contract it was
not faked here. Evidence (recorded in `lines_reconciliation.json`):

1. `player2_id`/`player3_id` are null on 100% of foul rows in every raw
   season file 2019-2026.
2. The trailing parenthetical on foul descriptions resolves to exactly three
   distinct names per game (mean 3.00, max 3) matching NBA referee crews
   (S. Foster, Z. Zarba, J. Goble, ...), not defenders/drawers.
3. "Offensive Foul Turnover" rows name only the committing player.

The committed side **is** extracted (`pf_committed`, `off_fouls_committed`,
`charges_committed`) and `pf_committed_rate` ships in this panel.

The drawn side now comes from a different surface: the NBA live CDN
play-by-play (`cdn.nba.com/static/json/liveData/playbyplay/`) names the
fouled player on every non-technical foul from 2019-20 on.
`fetch_fouls_drawn_cdn.py` and `build_fouls_drawn_game_lines.py` build a
box-confirmed per-player-game ledger (8,543 of 8,879 games measured, 2019-20
through 2025-26; unmeasured games are never zero), and
`build_tracking_next_game.py --family fouls_drawn` fits next-game filters for
offensive fouls drawn (charges plus non-charge), charges drawn, and PFD
(family-map lane 20; `NBA_FOULS_DRAWN_NEXT_GAME.md`). That lane is separate
from this 17-skill panel and its data start one season later.

## Skill panel

`build_nba_estimated_skills.py` fits one scalar Kalman filter per player per
skill with the WNBA panel's engine generalized to an exposure column
(prior-variance inflation `q_game + q_day*days_gap + q_offseason*season_cross`,
observation variance `r / exposure`):

| Family | Skills | Exposure / obs variance |
| --- | --- | --- |
| Percentage | `rim_pct`, `mid_pct`, `three_pct`, `ft_pct` | attempts, `r / attempts` |
| Offensive rate | `rim_fga_rate`, `mid_fga_rate`, `three_fga_rate`, `fta_rate`, `pts_rate`, `ast_rate`, `tov_rate`, `oreb_rate`, `usg_pct` | offensive on-court possessions |
| Defensive rate | `dreb_rate`, `stl_rate`, `blk_rate` | defensive on-court possessions |
| Total rate | `pf_committed_rate` | off + def on-court possessions |

`usg_pct = (FGA + 0.44*FTA + TOV) / off_poss`. Zone split: three = "3PT"
text; rim = rim action types/descriptions and not three; mid = the rest.
Each skill runs on its own sub-panel (rows with positive exposure, per-skill
day gaps and season crossings): a game with zero FTA is no `ft_pct`
observation but a legitimate zero for `fta_rate`. Per-skill
`(q_game, q_day, q_offseason, r, v0)` are tuned by differential evolution
(seed `20260826`, `maxiter=30`, `popsize=12`, `tol=1e-7`, no polish) on
one-step-ahead exposure-weighted squared error. Full DE completed in 314.7s
total (~5.3 min), so no compute reduction from the WNBA protocol was needed.
Percentage skills are fit twice — raw and centered on the as-of league mean
— selected on development WMSE only.

## Walk-forward protocol and leakage controls

2019 warms the filters only; development = seasons 2020-2025 with at least
one prior observation of the skill; **2026 is a frozen-parameter holdout**.
Baselines on identical eligible rows and weights: trailing-10-observation
exposure-weighted mean, season-to-date with career fallback, career-to-date.

Leakage controls (each verified in code and, where noted, empirically):

- **Pre-2026 constants only.** Every global scalar touching predictions or
  tuning — per-skill debut mean, base variance, and mean exposure used to
  scale DE bounds — is computed over pre-2026 rows only.
- **As-of centering.** The percentage-skill league center is completed
  prior-season totals pooled with league season-to-date strictly before the
  game date; the only fallback (full-first-season mean, dates before any
  league observation exists) is asserted at runtime to touch 2019 warm-up
  dates only, and warm-up rows never enter the objective or eval.
- **Strictly-prior baselines.** All three baselines are shift-based
  (exclude the current game), on the same eligible rows and exposure weights
  as the filter eval.
- **One look at 2026.** `--quick` (the smoke/debug path) evaluates dev only
  and suppresses the holdout entirely, so bug-fix iterations could not view
  2026; the holdout was computed exactly once, by the full run, after every
  design choice (extraction, skill definitions, exposures, DE settings,
  bounds, variant selection) was frozen.
- **Exposure is noise scaling, not signal.** Realized same-game exposure
  enters only the observation-noise term `r/exposure` and the evaluation
  weights; the evaluated `pred_rate` is formed strictly before the game's
  observation. Deployment pairs predicted rates with separately projected
  minutes/possessions.
- **Real dates.** `days_gap`/`season_cross` come from the schedule-derived
  `game_date` (0 of 10,191 games ambiguous; season windows match the real
  calendar including the 2020 bubble; opening-night/Finals spot checks
  exact), not fetch timestamps.
- **Dev-only selection.** The centered-vs-raw variant choice uses dev WMSE
  only. Reconciliation/sanity aggregates over all seasons are audit-only and
  feed nothing.

## Results (August 26, 2026 build)

Percent improvement in exposure-weighted MSE versus each baseline (positive
= filter better), on identical eligible rows and weights. Dev = 2020-2025;
2026 = frozen holdout, evaluated exactly once.

| Skill | Variant | Dev vs t10 | Dev vs season | Dev vs career | 2026 vs t10 | 2026 vs season | 2026 vs career | n dev | n 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rim_pct | centered | +8.93% | +8.95% | +1.90% | +9.09% | +9.20% | +2.15% | 124,738 | 22,056 |
| mid_pct | centered | +9.41% | +9.66% | +2.09% | +9.83% | +10.22% | +2.54% | 109,855 | 18,842 |
| three_pct | uncentered | +9.67% | +8.96% | +1.88% | +9.73% | +8.78% | +1.93% | 125,207 | 22,731 |
| ft_pct | uncentered | +8.10% | +11.18% | +2.50% | +8.47% | +10.49% | +2.56% | 88,095 | 15,846 |
| rim_fga_rate | uncentered | +6.33% | +6.72% | +6.81% | +6.36% | +5.68% | +9.12% | 159,849 | 28,425 |
| mid_fga_rate | uncentered | +5.58% | +6.89% | +14.15% | +5.46% | +5.04% | +13.63% | 159,849 | 28,425 |
| three_fga_rate | uncentered | +5.46% | +5.86% | +10.63% | +5.25% | +5.66% | +12.43% | 159,849 | 28,425 |
| fta_rate | uncentered | +6.67% | +6.25% | +4.54% | +6.33% | +6.06% | +6.86% | 159,849 | 28,425 |
| pts_rate | uncentered | +6.33% | +5.48% | +6.03% | +5.82% | +5.58% | +7.26% | 159,849 | 28,425 |
| ast_rate | uncentered | +5.62% | +5.09% | +5.95% | +5.78% | +5.08% | +6.87% | 159,849 | 28,425 |
| tov_rate | uncentered | +7.68% | +7.32% | +2.71% | +7.10% | +6.18% | +2.57% | 159,849 | 28,425 |
| oreb_rate | uncentered | +8.27% | +8.47% | +3.09% | +9.43% | +7.60% | +4.15% | 159,849 | 28,425 |
| usg_pct | uncentered | +5.43% | +6.29% | +14.48% | +5.52% | +5.62% | +15.44% | 159,849 | 28,425 |
| dreb_rate | uncentered | +7.01% | +6.53% | +3.17% | +6.83% | +5.43% | +4.03% | 159,890 | 28,433 |
| stl_rate | uncentered | +8.78% | +7.85% | +1.90% | +8.76% | +7.23% | +2.32% | 159,890 | 28,433 |
| blk_rate | uncentered | +7.84% | +7.16% | +2.24% | +7.69% | +5.97% | +2.36% | 159,890 | 28,433 |
| pf_committed_rate | uncentered | +7.57% | +7.69% | +2.74% | +8.27% | +10.47% | +3.06% | 159,989 | 28,444 |

**All 17 skills beat all three baselines on both windows (17/17 holdout
sweep), with holdout improvements tracking development closely** — the
tuned drift/noise structure generalized rather than overfitting the dev
seasons. The mechanism pattern is coherent: career-to-date is the hardest
baseline for slowly-moving percentage and stocks skills (career is already a
strong shrinkage estimator there; the filter's edge is +1.9-2.6%), while
volume/role skills drift with team context, so the filter beats career by
far more (mid_fga_rate +13.6%, usg_pct +15.4%, three_fga_rate +12.4% on the
holdout). rim_pct and mid_pct selected the as-of league-centered variant on
dev (the league's rim/mid environment moves enough to be worth removing);
the other percentages stayed raw. `--quick` selection can differ (mid_pct
was raw under quick DE) — only the full run's dev selection is canonical.

Board sanity at the June 2026 states (582 players with a 2026 last
observation): top predicted usage = Dončić .431, Antetokounmpo .415, with
Brunson/Leonard/Embiid/Gilgeous-Alexander in the top ten; top predicted
rim% = Ayton/Poeltl/Allen/Mobley/Antetokounmpo; top predicted per-100
assists = McConnell/Young/Giddey/Jokić; blocks = Wembanyama 5.22 per 100
def poss, clear of the field; OREB = Mitchell Robinson/Adams/Capela. No
`p_off_fouls_drawn` column exists, per the blocker above.

## EPM E-Skills benchmark: full 2025-26 sweep

Reproduce with
`.venv/bin/python nba_pipeline/scripts/compare_nba_estimated_skills_to_epm.py`.
The command reads the canonical Kalman predictions, player-game outcomes, and
12 cached date-D `go0` snapshots, then writes aggregate-only results under
`results/estimated_skills/epm_benchmark/`; it does not copy raw provider
values into the benchmark artifacts.

Season-long head-to-head against Dunks & Threes E-Skills predicted skills
(`p_*` columns from the cached semi-monthly `epm_all_*_go0.csv` snapshots;
benchmark-only per the standing EPM fence — this comparison changes
nothing about the panel). At each of 12 dates from pre-season October 15,
2025 through April 1, 2026, our states (latest posterior strictly before
D) and EPM's snapshot predict each joined player's pooled rest-of-2026
(RS+PS) actuals from `player_game_lines.parquet`, exposure-weighted (rest
attempts for percentages, rest offensive/defensive possessions for
rates), 399-501 common players per skill-date. Sixteen skills map one-to-one
(`pf_committed_rate` has no EPM counterpart).

**Result: statistical parity wire to wire.** Mean exposure-weighted r
across the 16 skills is within ±0.005 of EPM at every date — including
pre-season (.803 vs .799), where our predictor is a flat offseason
carryover of the June 2025 posterior with no age curve, and ~5 games in
(Nov 1: .803 vs .805). The pre-hoc hypothesis that EPM's machinery wins
early when priors matter most is not supported. Both panels sit far above
the naive references everywhere (season-to-date .561 at Nov 1 vs .803/.805;
career-to-date .762 at pre-season vs .803/.799) — and agree with each
other far more than either agrees with the baselines.

Level-adjusted paired-bootstrap verdicts over all 192 skill-date cells:
154 ties, 13 ours, 25 EPM. Ten of our 13 are `rim_fga_rate` (our-basis
rim-zone definition — discounted, not claimed). EPM's 25 are scattered
small edges (max |Δr| ≈ .015) concentrated mid-season in the
box-counting family — ast/tov/oreb/dreb/pts rates plus ft_pct — exactly
the standard-definition skills where basis bias is minimal. Plausible
mechanisms, not tested: EPM's pre-2019 player history (our filters warm
up from 2019, so veteran career anchoring — sharpest for FT% — is
truncated) and cross-stat pooling (their model is multivariate; ours is
17 independent scalar filters). Both suggest extensions: earlier warm-up
from the deeper raw archive, and any cross-stat layer must predeclare its
gate given the WNBA fusion null.

Caveats: targets use our zone/possession extraction (advantages us on
rim/mid skills; neutral for box-counting skills). The late-season r
decline (Apr 1 mean .696) reflects shrinking rest-of-season target
samples, not degrading panels. Snapshot cache is semi-monthly; a fresh
fetch needs `EPM_API_KEY`, not configured in this environment.

## Outputs

| File | Contents |
| --- | --- |
| `player_game_lines.parquet` | Per player-game shot/box/possession lines, 2019-2026 RS+PS |
| `lines_reconciliation.json` | Extraction audits + the foul-drawn blocker evidence |
| `hyperparams.json` | Per-skill DE params for both centering variants, bounds, selection |
| `predictions.parquet` | One-step-ahead pred rate/var + posterior per row (selected variant, raw space) |
| `epm_benchmark/skill_date_results.csv` | Aggregate date-by-skill correlations, level-adjusted WMSE differences, paired intervals, and verdicts |
| `epm_benchmark/skill_summary.csv` | Twelve-date correlation and verdict summary by skill |
| `epm_benchmark/summary.json` | Protocol, aggregate date summary, verdict counts, fences, and input hashes |
| `current_states.csv` | The board: `p_{skill}` posterior + variance, obs counts, career exposure, per-100 columns for rate skills |
| `eval_report.json` | Per-skill dev/holdout vs all baselines and both variants + leakage-control ledger |

## Fences

- The panel is **descriptive/forecast research only** — never a rating,
  never a prior for RAPM/VPM solves, and it feeds no production surface.
  Only stabilized player-season summaries could ever flow toward rating
  priors, and that needs its own predeclared forward gate.
- No proprietary ratings (EPM/DARKO/DPM/eWINS) appear anywhere as features,
  targets, or tuning inputs; the panel is built entirely from repo-owned
  PBP and possession surfaces.
- Predicted rates are per-attempt (percentages) or per-possession (rates);
  per-game projections require separately projected minutes/possessions.
- 2026 stays out of every objective, bound, initialization, and selection;
  in-season use is frozen-parameter filtering only.
- Generated artifacts stay out of git (`nba_pipeline/results/` is ignored).

## Rebuild

```
.venv/bin/python nba_pipeline/scripts/build_nba_player_game_lines.py --start 2019 --end 2026
.venv/bin/python nba_pipeline/scripts/build_nba_estimated_skills.py           # --quick for dev-only smoke
```

States are as-of each player's last observed game (June 13, 2026 Finals at
this build). DE re-tuning is deterministic and 2026-fenced.

## Future work

- Pre-2019 warm-up from the deeper raw archive once extraction is validated
  there (1997-2013 Gabriel files have different description conventions).
- Foul-drawn rates from a vetted per-game external surface, if one is ever
  staged locally; the committed side is already in place.
- Age-curve drift at season boundaries once a DOB table is wired in.
- A fusion second stage is deliberately not attempted: the WNBA LightGBM
  fusion was a documented null under far less data-starved conditions per
  skill count, and any NBA attempt should predeclare its gate first.

---

# v3 (2026-09-04): OU state + forward objective + cold-start corrections

> **Status (2026-09-13): objective rejected by the owner.** The target for this
> lane is **next-game error**. v3 below is tuned on a forward K=41 aggregate and
> is retained only as a documented research variant; it is not the selected
> model. Selection now uses next-game WMSE (see "Next-game selection" at the end
> of this file). The forward 2026 confirmation below answers a different
> question and does not confirm any next-game candidate.

Confirmed on the frozen 2026 holdout, scored once. **15/17 skills improved,
median +4.81% forward MSE, range -1.22% .. +10.01%**; 52/68 as-of dates won
(p=1.4e-05); sign test 15/17 (p=0.0023). Dev median was +4.91%, so shrinkage
dev -> holdout is +0.10 pp, and the dev-vs-2026 gain correlation across skills
is r=0.844 — development predicted *which* skills would benefit, not merely
that they would.

## What changed, and how much each part was worth

Measured on dev 2020-2025, forward (rest-of-season) aggregate metric:

| change | median gain | skills | note |
|---|---|---|---|
| OU state (v2) | +0.18% *(game metric)* | 17/17, 102/102 seasons | see below |
| + forward objective | +3.23% | 17/17, 344/391 dates | the dominant term |
| + cold-start corrections | +4.91% (incr. +1.63%) | 17/17, 363/391 dates | |

### 1. Objective (largest effect)

v1 tuned hyperparameters against **next-game** WMSE. Every consumer of these
skills wants **rest-of-season**. A rim_pct row averages ~3.5 attempts, so
game-level WMSE is dominated by irreducible binomial noise and barely
discriminates between hyperparameters that differ materially in deployment:
the same state change reads +0.18% on the game metric and +3.2% on the forward
metric. v3 tunes against the player's next-K-game exposure-weighted rate,
precomputed per row by reverse cumsum, so it costs the same to evaluate.

Horizon K=41 (half a season) selected on dev. The gain is monotone and
saturating in K (usg_pct +4.68/+5.60/+6.22/+6.38% at K=10/20/41/rest-of-season)
with no knife-edge. K=41 captures ~97% of the rest-of-season gain while
deliberately NOT matching the rest-of-season evaluation construction, so the
forward evaluation stays an arms-length check rather than a restatement of the
objective. Game-level MSE cost is <=0.61%.

Mechanism: v1 **under-smoothed**, chasing game noise. Forward tuning yields a
smoother filter; game MSE gets slightly worse while forward MSE improves a lot.

### 2. State: Ornstein-Uhlenbeck with a population attractor

All 17 skills showed negative mean reversion under v1's random walk (Test 1),
and DE could not tune it away because only q/r is identified by the gain. Both
components are now OU sharing a common population attractor, stationary-variance
parameterised; v1 is nested at tau_mu -> inf. Fitted timescales:

| component | pct skills | rate skills |
|---|---|---|
| tau_mu (slow talent) | 13k-72k days | 4k-8k days |
| tau_d (fast deviation) | 61-400 days | 23-41 days |

tau_mu running to decades means the talent component is effectively v1's random
walk, so the entire structural gain comes from the *second* component reverting
to zero instead of random-walking — transient hot/cold streaks now decay.
Substantively: **role/volume skills mean-revert on a ~1-month timescale,
shooting percentages on a ~2-13-month one.**

### 3. Cold-start corrections (prior-seasons-only)

Debut prior from biographical covariates (v1 bias -0.076 SD, rmse_z 1.34 vs
0.96 at 100+ games), age level shift at season crossings, early-season
transient on the prediction only. Every table is fit on seasons strictly before
the target season.

These load on the *opposite* skills from the objective fix: they are worth
+2.88% on rim_pct and +2.68% on ft_pct but only +0.94% on mid_fga_rate. The
objective fixes dynamics, which matters where there is real fast signal; the
corrections fix the prior, which dominates where the likelihood is weak — a
noisy shooting percentage leans hard on where it started.

### 4. Variance calibration (independent of the point-accuracy work)

v1's intervals were unusable for props: sd(z) 0.168-0.352 (median 0.240),
68% intervals covering 99-100%. WMSE tuning leaves the absolute variance scale
unidentified because only q/r enters the gain. An NLL objective fixes
calibration (sd(z) 1.000) but costs point accuracy (-0.98% MSE on three_pct).

v3 therefore keeps WMSE-family tuning and adds a **one-parameter post-hoc
rescale fit on prior seasons only**: dev sd(z) 0.997-1.000, and on frozen 2026
median 1.011 (range 0.846-1.118). Calibration at zero cost to accuracy.

## Deliberate omissions

- **No health/injury mean term.** The within-season absence diagnostic is a
  null (-0.003 to -0.004 SD across every gap bucket out to 60 days). Elapsed
  time belongs in the variance, which the OU decay already handles.
- **No fusion stage**, consistent with v1 and with the WNBA fusion null.

## Underperformers — do not paper over these

| skill | 2026 | CI | dates | dev |
|---|---|---|---|---|
| three_pct | -0.43% | [-3.52,+2.47] | 0/4 | +1.12% |
| stl_rate | -1.22% | [-4.59,+1.95] | 0/4 | +2.19% |

Both CIs cross zero, so neither is a resolved regression, but both lost every
as-of date on 2026 and both are low-signal skills (2026 wcorr 0.417 and 0.726).
Per-skill 2026 power is genuinely low — 4 as-of dates and ~1,700 cells versus
23 dates on dev — so only 8/17 individual CIs exclude zero even though the
pooled result is strong. Treat per-skill 2026 effect sizes as noisy; the
pooled and cross-skill-correlation evidence is what carries.

## Files

- `scripts/build_nba_estimated_skills_v2.py` — OU state, stage A (v1 re-tuned
  at an identical DE budget for a fair comparison; v1's own published params
  were unconverged, 0/21 DE runs converged, 8/85 params on bounds)
- `scripts/build_nba_estimated_skills_v2_stageb.py` — corrections
- `scripts/build_nba_estimated_skills_v3.py` — selected candidate
- `scripts/estimated_skills_corrections.py` — correction fitters
- `scripts/build_estimated_skills_age_lookup.py` — age/bio lookup builder
- `scripts/eval_estimated_skills_forward.py` — forward metric; `forward_cells`
  defaults to DEV_SEASONS so no dev tooling can read the holdout by accident
- `scripts/confirm_estimated_skills_2026.py` — the one-shot confirmation
- `results/estimated_skills/{v2_stage_a,v3_nocorr,v3,confirmation_2026}.json`

## Spend ledger

The forward-aggregate 2026 comparison was scored once, on the forward-tuned
candidate, with no tuning. No next-game comparison of any candidate has been
observed on 2026, but per-skill 2026 forward gains and v3 `sd(z)` on 2026 game
rows were seen, so 2026 is partially exposed for next-game selection.

Caveat on the comparison: league centering uses the contemporaneous season's
mean, which is v1's own pre-existing convention and is applied identically to
both arms, so it cancels in the paired comparison but is not a strictly
causal-in-time construction.

---

# Next-game selection (2026-09-14): v4 = OU + corrections + pre-tip next-game context

**Target: one-step-ahead exposure-weighted game MSE on dev 2020-2025** (owner
decision). Every number in this section is next-game WMSE on dev with 2026
dropped at load; no v4 candidate has been scored on 2026.

**Fairness rule (owner, 2026-09-14):** every feature must be constructible
before tip from information that exists in live use. The first v4 build read
tonight's realised PBP roster to define teammate absence. It was removed and
its numbers withdrawn (usg_pct had shown +2.46%; almost all of that absence
gain came from knowing the realised roster).

Script: `scripts/build_nba_estimated_skills_v4.py` (context terms in
`scripts/estimated_skills_context.py`, game table from
`scripts/build_estimated_skills_game_context.py`). Outputs
`results/estimated_skills/{v2_stage_b,v4_nextgame,v4_nextgame_nocorr,v4_nextgame_noabsence}.json`.

## Result

| skill | v4 vs A (OU) | 95% CI (game bootstrap) | seasons | vs B | seasons | vs v1 | sd(z) |
|---|---|---|---|---|---|---|---|
| rim_pct | +0.454% | [0.362, 0.545] | 6/6 | +0.296% | 6/6 | +0.500% | 0.995 |
| mid_pct | +0.137% | [0.068, 0.206] | 6/6 | +0.065% | 6/6 | +0.161% | 0.999 |
| three_pct | +0.184% | [0.128, 0.246] | 6/6 | +0.060% | 6/6 | +0.194% | 1.005 |
| ft_pct | +0.096% | [0.009, 0.188] | 5/6 | +0.018% | 4/6 | +0.151% | 0.992 |
| rim_fga_rate | +0.777% | [0.681, 0.875] | 6/6 | +0.746% | 6/6 | +1.158% | 0.999 |
| mid_fga_rate | +1.637% | [1.483, 1.797] | 6/6 | +1.516% | 6/6 | +2.229% | 0.959 |
| three_fga_rate | +1.058% | [0.935, 1.166] | 6/6 | +1.081% | 6/6 | +1.336% | 1.003 |
| fta_rate | +0.556% | [0.473, 0.631] | 6/6 | +0.507% | 6/6 | +0.802% | 0.981 |
| pts_rate | +0.606% | [0.523, 0.688] | 6/6 | +0.452% | 6/6 | +0.830% | 1.000 |
| ast_rate | +0.789% | [0.692, 0.889] | 6/6 | +0.651% | 6/6 | +1.255% | 1.001 |
| tov_rate | +0.732% | [0.643, 0.821] | 6/6 | +0.681% | 6/6 | +0.838% | 0.991 |
| oreb_rate | +0.261% | [0.202, 0.318] | 6/6 | +0.109% | 6/6 | +0.338% | 1.024 |
| usg_pct | +0.716% | [0.626, 0.807] | 6/6 | +0.648% | 6/6 | +1.343% | 0.992 |
| dreb_rate | +0.452% | [0.383, 0.523] | 6/6 | +0.301% | 6/6 | +0.675% | 0.973 |
| stl_rate | +0.343% | [0.285, 0.402] | 6/6 | +0.319% | 6/6 | +0.409% | 0.999 |
| blk_rate | +0.635% | [0.551, 0.715] | 6/6 | +0.428% | 6/6 | +0.817% | 0.993 |
| pf_committed_rate | +0.478% | [0.399, 0.561] | 6/6 | +0.322% | 6/6 | +0.551% | 0.977 |

v4 beats stage A in 17/17 skills and 101/102 seasons (median +0.56%), and beats
stage B in 100/102 seasons (median +0.43%). Its median gain over re-tuned v1 is
+0.80%. Every CI against A excludes zero. `sd(z)` is the median of per-season
values from an expanding-window variance scale.

A = OU state (stage A). B = A plus debut/age/early-season corrections re-tuned
on next-game WMSE (stage B: 17/17 skills non-negative except three_fga_rate
-0.023%, median +0.12% vs A).

## What each context term is worth (dev)

- **Opponent** (shrunk season-to-date mean of other players' one-step
  residuals against tonight's opponent, strictly earlier dates, plus last
  season's final opponent value): +0.02% to +1.39% on the base OU filter,
  6/6 seasons in 15/17 skills; largest for mid/three/rim attempt rates, tov,
  blk. **Placebo** — a random opponent drawn independently per game — gives
  -0.004%..+0.014%, so the effect is not a leakage or group-construction
  artifact.
- **Own team** (same construction on tonight's team): small except usage
  (+0.19%), ast, tov, pts.
- **Schedule** (home, back-to-back, 3+ rest days): at most +0.10%; kept
  because it is cheap and mostly positive.
- **Teammate absence, pre-tip** (rate skills only). A candidate is a teammate
  averaging 10+ minutes who missed the team's most recent completed game;
  `streak` is how many straight team games they have missed (1-20). Each one
  is weighted by the estimated chance they sit tonight,
  `1 - P(plays | streak bucket 1/2/3/4-5/6-10/11-20, 20+ minute tier)`, fit
  on earlier seasons only. The terms are expected-out minutes share and
  skill excess, split fresh (streak <= 5) versus long, plus share x focal
  excess. P(plays) is calibrated (prior-season estimate vs realised: 0.43 vs
  0.44 at streak 1, 0.04 vs 0.04 at 11+; Brier 0.155 vs 0.182 for a constant).
  Increment over the other terms (v4 minus `v4_nextgame_noabsence`, both
  measured against the same A rows, separately DE-tuned, no paired CI):
  usg_pct +0.30 pp, pts +0.15, ast +0.15, mid_fga +0.12, fta +0.07, tov
  +0.05, rim_fga/three_fga +0.04/+0.03; oreb/dreb/stl/blk/pf within about
  ±0.02 pp, i.e. DE noise. At a tiny tuning budget, the chance-weighted
  version beat an unweighted streak-1-vs-2+ split on usg_pct (+0.37% vs
  +0.14% over A). Not used for pct skills, where absence was null.
- **Corrections under context** (v4 vs `v4_nextgame_nocorr`): positive in
  16/17 skills, largest for blk/dreb/oreb (+0.15..+0.19 pp) and pct skills
  (+0.07..+0.12 pp); three_fga_rate -0.04 pp.

Mechanics: context coefficients are expanding-window (seasons < S). The
context enters as `transient = early_season + lam * ctx`, so it is added to
the prediction and removed from the innovation. DE tunes the six OU params
plus `lam` in [0, 2]; the selected `lam` is 0.50-1.40. Without absence,
usg_pct `lam` goes to 1.99 (at the bound). The shrinkage `c` is picked on dev
with the base filter.

## Caveats — read before using for props

- The absence proxy is roster history only. A live pipeline could add the
  official injury/inactive report, but only as a pre-tip snapshot with its
  timestamp logged. That version is untested, and the report must never be
  backfilled from post-game rosters.
- Hyperparameters, `lam` and `c` are tuned on the same dev seasons they are
  scored on, which is the convention in this lane. Only the context,
  correction and P(plays) coefficients are out-of-sample in time. Per-season
  wins and game-clustered CIs are paired against A/B/v1 on identical rows.
- **2026 status:** 2026 is partially exposed by the v3 forward confirmation
  (per-skill forward gains and v3 `sd(z)` were seen). No next-game number for
  any candidate has been observed on 2026, and the context terms did not
  exist when that exposure happened. Whether 2026 may still serve as the
  one-shot next-game confirmation for v4 is an owner decision. Running it
  requires `build_estimated_skills_game_context.py --include-holdout` plus
  `--allow-holdout` on v4.

# Next-game lab (2026-09-14): what moves next-game error beyond v4

Owner ask: review the methodology, find the biggest next-game improvements,
test them, and make the test surface reusable. Everything below is dev
2020-2025 (2019 warm-up), 2026 dropped at load, 2027 untouched. All numbers
are **% reduction in one-step-ahead exposure-weighted game MSE** with a
paired, game-clustered 95% bootstrap CI (1000 reps) and per-season wins /6.
Percent gains are small by construction (game rows are mostly binomial noise;
v4 itself was a median `+0.56%` over OU), so read CIs and season wins first.

## Protocol

- `scripts/research_skills_nextgame_lab.py` (details under "The harness").
  Each variant re-tunes the six OU parameters plus `ctx_lambda` (plus any
  option-specific parameter) with the v4 differential-evolution recipe
  (seed 20260826, maxiter 60, popsize 15) followed by a log-space
  Nelder-Mead polish that is kept only if it improves. `base` is v4 under the
  same optimiser, so "variant vs base" isolates the model change.
- Closure: v4's stored parameters pushed through the lab engine reproduce
  the v4 objective to `0.00e+00` in 17/17 skills; base re-tune vs v4 is
  `+0.000%` everywhere (the polish finds nothing, the DE optimum is stable).
- Every feature is pre-tip constructible: the playoff flag is the schedule;
  the minutes trend uses completed games only; usage form uses the
  focal player's own prior-game residuals joined strictly on earlier dates
  (`merge_asof`, `allow_exact_matches=False`); team change compares the
  team id of completed games. No realised roster, no same-game information.

## Round 1: eight single ideas, every skill

| option | what it is |
|---|---|
| `fixr` | `r` fixed at the natural per-exposure noise scale; tune 5 params (tests the flat scale direction) |
| `noise` | heteroskedastic observation noise: binomial `p(1-p)/n` for pct skills, Poisson `mu/n` for rates; `r` becomes dimensionless |
| `ctxplus` | + playoff flag, playoff x (stage-A level minus league centre), minutes trend `(mean prior 3 - mean prior 30)/48` |
| `ctxmult` | rate skills: context betas fit on relative residuals (weights `expo * level^2`) and applied as `level * (1 + lam * ctx)` instead of additively |
| `center` | rate skills: as-of league centering of the observation (pct skills already are) |
| `clip` | innovation clipped at `k * sqrt(S)`, `k` tuned in [1.5, 8] |
| `xusg` | cross-skill form: shrunk trailing-5 usage residual of the focal player (`c = 5` games), strictly prior games |

Variant vs base (`% MSE gain [95% CI] seasons won`):

| skill | fixr | noise | ctxplus | ctxmult | center | clip | xusg |
|---|---|---|---|---|---|---|---|
| rim_pct | +0.000 [-0.000,+0.000] 4/6 | +0.002 [-0.001,+0.005] 5/6 | +0.036 [+0.020,+0.051] 5/6 | – | – | -0.000 [-0.000,+0.000] 3/6 | +0.008 [-0.005,+0.019] 5/6 |
| mid_pct | +0.000 [-0.000,+0.000] 3/6 | +0.000 [-0.001,+0.001] 4/6 | +0.006 [-0.006,+0.019] 5/6 | – | – | +0.000 [-0.000,+0.000] 2/6 | -0.001 [-0.003,+0.001] 4/6 |
| three_pct | -0.000 [-0.000,+0.000] 3/6 | +0.002 [+0.000,+0.003] 5/6 | +0.016 [+0.003,+0.028] 5/6 | – | – | -0.000 [-0.000,+0.000] 4/6 | -0.003 [-0.009,+0.003] 3/6 |
| ft_pct | -0.000 [-0.000,+0.000] 2/6 | -0.076 [-0.093,-0.057] 0/6 | +0.011 [+0.001,+0.023] 5/6 | – | – | +0.000 [-0.000,+0.000] 3/6 | -0.000 [-0.006,+0.005] 3/6 |
| rim_fga_rate | -0.000 [-0.000,+0.000] 2/6 | -0.141 [-0.188,-0.094] 0/6 | +0.034 [+0.015,+0.055] 6/6 | +0.077 [+0.032,+0.123] 6/6 | -0.004 [-0.013,+0.005] 3/6 | +0.000 [-0.000,+0.000] 4/6 | +0.009 [+0.001,+0.017] 5/6 |
| mid_fga_rate | +0.000 [-0.000,+0.000] 2/6 | -0.748 [-0.857,-0.649] 0/6 | +0.023 [+0.007,+0.038] 6/6 | +0.500 [+0.397,+0.605] 6/6 | +0.015 [-0.002,+0.031] 5/6 | +0.001 [-0.003,+0.006] 4/6 | +0.021 [+0.005,+0.039] 6/6 |
| three_fga_rate | +0.000 [-0.000,+0.000] 4/6 | -0.001 [-0.051,+0.050] 3/6 | +0.057 [+0.035,+0.080] 6/6 | +0.143 [+0.095,+0.188] 6/6 | +0.010 [-0.009,+0.028] 3/6 | +0.006 [-0.003,+0.015] 4/6 | +0.000 [-0.002,+0.003] 4/6 |
| fta_rate | +0.000 [-0.000,+0.000] 3/6 | -0.315 [-0.370,-0.258] 0/6 | +0.009 [-0.006,+0.024] 4/6 | +0.119 [+0.073,+0.164] 6/6 | -0.030 [-0.047,-0.013] 2/6 | +0.030 [+0.012,+0.048] 5/6 | +0.098 [+0.059,+0.134] 6/6 |
| pts_rate | -0.000 [-0.000,+0.000] 3/6 | -0.074 [-0.100,-0.046] 0/6 | +0.141 [+0.107,+0.174] 6/6 | +0.077 [+0.053,+0.102] 6/6 | -0.035 [-0.045,-0.026] 1/6 | +0.000 [-0.000,+0.000] 5/6 | +0.086 [+0.060,+0.115] 6/6 |
| ast_rate | +0.000 [-0.000,+0.000] 3/6 | -0.421 [-0.482,-0.356] 0/6 | +0.237 [+0.188,+0.288] 6/6 | +0.149 [+0.101,+0.198] 5/6 | +0.004 [-0.013,+0.021] 3/6 | +0.000 [-0.000,+0.000] 5/6 | +0.013 [-0.015,+0.039] 4/6 |
| tov_rate | -0.000 [-0.000,+0.000] 3/6 | -0.049 [-0.071,-0.026] 0/6 | -0.002 [-0.007,+0.002] 2/6 | +0.007 [-0.025,+0.038] 3/6 | -0.020 [-0.032,-0.008] 1/6 | -0.000 [-0.000,+0.000] 4/6 | +0.059 [+0.029,+0.091] 6/6 |
| oreb_rate | -0.000 [-0.000,+0.000] 3/6 | -0.283 [-0.345,-0.224] 0/6 | -0.002 [-0.012,+0.007] 3/6 | +0.062 [+0.028,+0.096] 5/6 | -0.003 [-0.016,+0.011] 2/6 | +0.001 [-0.003,+0.003] 4/6 | +0.003 [-0.001,+0.008] 6/6 |
| usg_pct | -0.000 [-0.000,+0.000] 4/6 | -0.137 [-0.169,-0.108] 0/6 | +0.098 [+0.075,+0.122] 6/6 | -0.005 [-0.035,+0.024] 3/6 | -0.003 [-0.005,-0.001] 2/6 | +0.019 [+0.004,+0.035] 4/6 | – |
| dreb_rate | -0.000 [-0.000,+0.000] 3/6 | -0.132 [-0.165,-0.101] 0/6 | +0.007 [-0.012,+0.024] 4/6 | +0.054 [+0.030,+0.080] 6/6 | -0.008 [-0.020,+0.003] 1/6 | +0.001 [-0.000,+0.003] 2/6 | -0.003 [-0.011,+0.006] 4/6 |
| stl_rate | +0.000 [-0.000,+0.000] 3/6 | -0.054 [-0.072,-0.037] 0/6 | +0.015 [-0.002,+0.032] 5/6 | +0.025 [+0.005,+0.044] 6/6 | -0.018 [-0.032,-0.004] 1/6 | +0.000 [-0.000,+0.000] 3/6 | -0.000 [-0.002,+0.002] 4/6 |
| blk_rate | -0.000 [-0.000,+0.000] 4/6 | -0.333 [-0.402,-0.266] 0/6 | +0.016 [-0.007,+0.040] 5/6 | +0.367 [+0.279,+0.455] 6/6 | -0.019 [-0.032,-0.007] 2/6 | +0.001 [-0.000,+0.001] 5/6 | -0.001 [-0.003,+0.001] 2/6 |
| pf_committed_rate | -0.000 [-0.000,+0.000] 3/6 | -0.067 [-0.084,-0.048] 0/6 | +0.205 [+0.160,+0.249] 6/6 | +0.025 [+0.010,+0.040] 5/6 | -0.035 [-0.057,-0.012] 2/6 | +0.000 [-0.001,+0.001] 1/6 | +0.001 [-0.008,+0.011] 3/6 |

Reading:

- **Multiplicative context (`ctxmult`) is the single biggest improvement.**
  Additive context under-corrects high-volume players and over-corrects
  low-volume ones; the same eleven context columns applied as a
  proportional shift win in 11/13 rate skills with CIs above zero, led by
  mid_fga `+0.50`, blk `+0.37`, ast `+0.15`, three_fga `+0.14`, fta `+0.12`.
  tov and usg are null (their context betas are already small).
- **Playoff + minutes trend (`ctxplus`)** is the second: ast `+0.24`,
  pf `+0.21`, pts `+0.14`, usg `+0.10`, and it is the only idea that helps
  pct skills (rim `+0.04`, three `+0.02`, ft `+0.01`, all CI above zero).
  This is the first documented next-game playoff adjustment in the lane;
  nothing in the literature scan (DARKO, E-Skills, LEBRON) describes one.
- **Cross-skill usage form (`xusg`)** is a real but narrow third: fta
  `+0.10`, pts `+0.09`, tov `+0.06`, mid_fga `+0.02`. A hot usage streak
  predicts next-game scoring volume beyond the skill's own fast component.
  This is the DARKO-style cross-stat coupling, restricted to one source.
- **Innovation clipping** is worth `+0.03` on fta and `+0.02` on usg and
  nothing elsewhere: the filter already handles outliers through `r`.
- **`fixr` is `±0.000` everywhere.** `(s_mu, s_d, q_off, r)` are jointly
  scale-invariant for predictions (proved in the engine test), so the
  "not converged" flag on every v4 tune is cosmetic; DE walks a flat
  direction and the predictions do not move.
- **Heteroskedastic noise loses, badly, on rate skills** (mid_fga `-0.75`,
  ast `-0.42`, blk `-0.33`, fta `-0.32`, oreb `-0.28`; 0/6 seasons in all 13
  rates) and is null on rim/mid/three pct (`+0.002`, `0`, `+0.002`) and
  negative on ft_pct (`-0.08`). The constant-`r`/exposure model already
  carries the exposure scaling; per-row `mu/n` variance over-trusts
  low-rate games and under-trusts high-rate games because within-player
  rate variation is not Poisson. Do not retry this as a straight swap.
- **As-of league centering of rate skills is null-to-negative** (pts,
  pf, fta, tov, blk, stl all with CIs below zero). The tuned `q_off` and the
  season-to-date own-team residual already absorb league drift.

## Round 2: stacking the winners, plus four literature ideas

Variants (letters used in the pairs table):

| letter | option set | new idea |
|---|---|---|
| A | `ctxplus+xusg` | additive control for the multiplicative increment |
| B | `ctxplus+ctxmult+xusg` | the round-1 winners together |
| C | B `+clip` | robust update on top |
| D | B `+jump` | **team-change variance inflation**: extra slow-state process noise `q_jump` (tuned, [0, 2 s_mu]) on the first game after the player's completed-game team id changes (DARKO's learning-rate bump) |
| E | `ctxplus+ctxmult+xall` | **cross-skill form from all 16 other skills** (each a shrunk trailing-5 base-filter residual, strictly prior dates) instead of usage only |
| – | `ctxps` | playoff flag + playoff x excess only (ablation of `ctxplus`) |
| – | `ctxmin` | minutes trend only (ablation of `ctxplus`) |

Variant vs base:

| skill | ctxplus+xusg | ctxplus+ctxmult+xusg | ctxplus+ctxmult+xusg+clip | ctxplus+ctxmult+xusg+jump | ctxplus+ctxmult+xall | ctxps | ctxmin |
|---|---|---|---|---|---|---|---|
| rim_pct | +0.044 [+0.025,+0.066] 5/6 | +0.044 [+0.025,+0.066] 5/6 | +0.044 [+0.024,+0.063] 5/6 | +0.051 [+0.028,+0.073] 5/6 | +0.038 [+0.008,+0.076] 5/6 | +0.038 [+0.022,+0.053] 6/6 | -0.002 [-0.007,+0.003] 3/6 |
| mid_pct | +0.005 [-0.007,+0.017] 5/6 | +0.005 [-0.009,+0.017] 5/6 | +0.005 [-0.007,+0.017] 5/6 | +0.005 [-0.007,+0.018] 5/6 | -0.015 [-0.035,+0.005] 3/6 | +0.008 [-0.005,+0.020] 5/6 | -0.002 [-0.005,+0.002] 3/6 |
| three_pct | +0.013 [-0.000,+0.027] 5/6 | +0.013 [-0.002,+0.028] 5/6 | +0.013 [-0.001,+0.026] 5/6 | +0.013 [-0.000,+0.026] 5/6 | +0.014 [-0.009,+0.039] 5/6 | +0.013 [+0.003,+0.024] 4/6 | +0.002 [-0.005,+0.008] 4/6 |
| ft_pct | +0.011 [-0.000,+0.022] 5/6 | +0.011 [-0.000,+0.022] 5/6 | +0.011 [-0.001,+0.022] 5/6 | +0.021 [+0.005,+0.039] 4/6 | -0.001 [-0.016,+0.015] 4/6 | +0.006 [-0.000,+0.013] 6/6 | +0.005 [-0.003,+0.012] 5/6 |
| rim_fga_rate | +0.043 [+0.023,+0.066] 6/6 | +0.114 [+0.066,+0.162] 6/6 | +0.114 [+0.067,+0.163] 6/6 | +0.157 [+0.103,+0.215] 6/6 | +0.243 [+0.178,+0.313] 6/6 | +0.034 [+0.018,+0.052] 6/6 | +0.001 [-0.008,+0.009] 4/6 |
| mid_fga_rate | +0.043 [+0.022,+0.064] 6/6 | +0.516 [+0.414,+0.627] 6/6 | +0.517 [+0.415,+0.617] 6/6 | +0.567 [+0.464,+0.671] 6/6 | +0.655 [+0.537,+0.797] 6/6 | +0.028 [+0.014,+0.044] 6/6 | -0.005 [-0.010,+0.000] 2/6 |
| three_fga_rate | +0.058 [+0.035,+0.081] 6/6 | +0.219 [+0.165,+0.273] 6/6 | +0.225 [+0.166,+0.283] 6/6 | +0.255 [+0.194,+0.311] 6/6 | +0.266 [+0.192,+0.338] 6/6 | +0.041 [+0.018,+0.060] 6/6 | +0.016 [+0.007,+0.026] 5/6 |
| fta_rate | +0.107 [+0.067,+0.147] 6/6 | +0.217 [+0.155,+0.284] 6/6 | +0.245 [+0.175,+0.314] 6/6 | +0.261 [+0.194,+0.328] 6/6 | +0.258 [+0.179,+0.335] 6/6 | -0.007 [-0.018,+0.005] 3/6 | +0.017 [+0.006,+0.028] 5/6 |
| pts_rate | +0.223 [+0.177,+0.266] 6/6 | +0.273 [+0.222,+0.322] 6/6 | +0.273 [+0.223,+0.321] 6/6 | +0.340 [+0.288,+0.396] 6/6 | +0.446 [+0.365,+0.520] 6/6 | +0.129 [+0.101,+0.159] 6/6 | +0.017 [-0.000,+0.035] 5/6 |
| ast_rate | +0.243 [+0.185,+0.297] 6/6 | +0.373 [+0.294,+0.452] 6/6 | +0.373 [+0.300,+0.450] 6/6 | +0.439 [+0.357,+0.520] 6/6 | +0.351 [+0.274,+0.431] 6/6 | +0.240 [+0.190,+0.291] 6/6 | -0.002 [-0.010,+0.006] 2/6 |
| tov_rate | +0.056 [+0.025,+0.085] 6/6 | +0.057 [+0.009,+0.102] 5/6 | +0.057 [+0.014,+0.106] 6/6 | +0.087 [+0.040,+0.136] 6/6 | +0.294 [+0.214,+0.368] 6/6 | -0.001 [-0.006,+0.003] 2/6 | -0.001 [-0.003,+0.001] 2/6 |
| oreb_rate | +0.001 [-0.011,+0.012] 3/6 | +0.062 [+0.023,+0.097] 5/6 | +0.062 [+0.023,+0.099] 5/6 | +0.075 [+0.036,+0.117] 5/6 | +0.098 [+0.049,+0.146] 4/6 | +0.000 [-0.010,+0.009] 4/6 | -0.003 [-0.008,+0.002] 3/6 |
| usg_pct | +0.098 [+0.073,+0.120] 6/6 | +0.090 [+0.054,+0.125] 6/6 | +0.107 [+0.070,+0.148] 6/6 | +0.207 [+0.156,+0.257] 6/6 | +0.161 [+0.098,+0.227] 6/6 | +0.071 [+0.052,+0.090] 6/6 | +0.027 [+0.012,+0.042] 5/6 |
| dreb_rate | +0.005 [-0.016,+0.022] 4/6 | +0.048 [+0.017,+0.081] 4/6 | +0.049 [+0.016,+0.083] 4/6 | +0.060 [+0.025,+0.095] 4/6 | +0.031 [-0.008,+0.071] 4/6 | +0.007 [-0.005,+0.021] 4/6 | -0.001 [-0.013,+0.011] 3/6 |
| stl_rate | +0.014 [-0.005,+0.032] 5/6 | +0.039 [+0.013,+0.065] 6/6 | +0.039 [+0.010,+0.067] 6/6 | +0.041 [+0.013,+0.068] 6/6 | +0.014 [-0.017,+0.046] 5/6 | +0.010 [-0.005,+0.024] 5/6 | +0.005 [-0.005,+0.016] 5/6 |
| blk_rate | +0.015 [-0.008,+0.038] 5/6 | +0.364 [+0.275,+0.456] 6/6 | +0.365 [+0.273,+0.464] 6/6 | +0.364 [+0.272,+0.454] 6/6 | +0.277 [+0.179,+0.372] 6/6 | +0.017 [-0.005,+0.038] 5/6 | -0.000 [-0.008,+0.007] 3/6 |
| pf_committed_rate | +0.204 [+0.159,+0.252] 6/6 | +0.231 [+0.182,+0.278] 6/6 | +0.232 [+0.182,+0.279] 6/6 | +0.253 [+0.202,+0.304] 6/6 | +0.271 [+0.219,+0.324] 6/6 | +0.195 [+0.153,+0.234] 6/6 | +0.010 [-0.009,+0.026] 5/6 |

Pairwise increments, each with its own paired CI (`pairs` in `round2.json`):

| skill | ctxmult increment (B − A) | jump increment (D − B) | xall vs xusg (E − B) | clip increment (C − B) |
|---|---|---|---|---|
| rim_pct | +0.000 [+0.000,+0.000] 0/6 | +0.007 [-0.003,+0.017] 5/6 | -0.005 [-0.035,+0.024] 5/6 | +0.000 [-0.000,+0.000] 3/6 |
| mid_pct | +0.000 [+0.000,+0.000] 0/6 | -0.000 [-0.000,+0.000] 2/6 | -0.020 [-0.038,-0.002] 1/6 | -0.000 [-0.000,+0.000] 2/6 |
| three_pct | +0.000 [+0.000,+0.000] 0/6 | +0.000 [-0.000,+0.000] 4/6 | +0.001 [-0.021,+0.022] 4/6 | +0.000 [-0.000,+0.000] 3/6 |
| ft_pct | +0.000 [+0.000,+0.000] 0/6 | +0.011 [-0.003,+0.024] 4/6 | -0.011 [-0.027,+0.005] 2/6 | +0.000 [-0.000,+0.000] 4/6 |
| rim_fga_rate | +0.071 [+0.028,+0.117] 6/6 | +0.044 [+0.022,+0.065] 6/6 | +0.129 [+0.078,+0.184] 5/6 | +0.000 [+0.000,+0.001] 5/6 |
| mid_fga_rate | +0.473 [+0.372,+0.572] 6/6 | +0.051 [+0.023,+0.075] 6/6 | +0.140 [+0.049,+0.215] 5/6 | +0.001 [-0.003,+0.005] 4/6 |
| three_fga_rate | +0.162 [+0.115,+0.214] 6/6 | +0.035 [+0.013,+0.057] 4/6 | +0.047 [-0.000,+0.098] 4/6 | +0.006 [-0.004,+0.017] 3/6 |
| fta_rate | +0.109 [+0.061,+0.162] 6/6 | +0.045 [+0.022,+0.068] 6/6 | +0.041 [-0.001,+0.089] 4/6 | +0.029 [+0.009,+0.048] 5/6 |
| pts_rate | +0.050 [+0.024,+0.077] 6/6 | +0.067 [+0.041,+0.094] 6/6 | +0.174 [+0.117,+0.231] 6/6 | +0.000 [-0.000,+0.000] 5/6 |
| ast_rate | +0.130 [+0.080,+0.184] 5/6 | +0.067 [+0.033,+0.103] 5/6 | -0.021 [-0.054,+0.017] 4/6 | +0.000 [-0.000,+0.000] 3/6 |
| tov_rate | +0.001 [-0.027,+0.034] 3/6 | +0.031 [+0.014,+0.048] 6/6 | +0.238 [+0.175,+0.298] 6/6 | +0.001 [-0.001,+0.002] 3/6 |
| oreb_rate | +0.061 [+0.028,+0.095] 5/6 | +0.013 [-0.002,+0.028] 5/6 | +0.037 [-0.005,+0.075] 4/6 | +0.000 [-0.003,+0.003] 4/6 |
| usg_pct | -0.008 [-0.038,+0.021] 3/6 | +0.117 [+0.081,+0.151] 6/6 | +0.071 [+0.010,+0.132] 4/6 | +0.017 [+0.006,+0.029] 5/6 |
| dreb_rate | +0.043 [+0.019,+0.067] 6/6 | +0.012 [-0.000,+0.025] 3/6 | -0.017 [-0.043,+0.010] 2/6 | +0.001 [-0.001,+0.003] 5/6 |
| stl_rate | +0.024 [+0.006,+0.044] 6/6 | +0.003 [-0.002,+0.008] 3/6 | -0.024 [-0.047,+0.001] 2/6 | +0.000 [-0.000,+0.000] 4/6 |
| blk_rate | +0.349 [+0.254,+0.439] 6/6 | -0.000 [-0.000,+0.000] 3/6 | -0.088 [-0.134,-0.041] 1/6 | +0.001 [-0.000,+0.002] 5/6 |
| pf_committed_rate | +0.028 [+0.011,+0.043] 6/6 | +0.021 [+0.006,+0.037] 6/6 | +0.040 [+0.014,+0.068] 5/6 | +0.000 [-0.001,+0.001] 2/6 |

Reading:

- **The winners stack.** B beats re-tuned v4 with the CI above zero in all
  13 rate skills and rim_pct (mid_fga `+0.52`, ast `+0.37`, blk `+0.36`,
  pts `+0.27`, pf `+0.23`, three_fga `+0.22`, fta `+0.22`, rim_fga `+0.11`,
  usg `+0.09`), 6/6 seasons in eleven of them. The multiplicative increment
  survives on top of `ctxplus+xusg` (B − A: mid_fga `+0.47`, blk `+0.35`,
  three_fga `+0.16`, ast `+0.13`, fta `+0.11`), so round 1 was not double
  counting.
- **Team-change variance inflation is a fourth real gain** (D − B): usg
  `+0.12`, pts `+0.07`, ast `+0.07`, mid_fga `+0.05`, fta `+0.05`, rim_fga
  `+0.04`, three_fga `+0.04`, tov `+0.03`, pf `+0.02`, all CIs above zero.
  The tuned `q_jump` is 5-25% of the stationary slow-state variance
  (usg 25%, pts 19%, ast 16%); blk tunes it to zero. It is null on the
  defensive/event skills and on pct skills. This is the DARKO idea the
  literature scan flagged as missing, confirmed on the owner's target.
- **All-skill form is a split, not a win** (E − B): large for tov `+0.24`,
  pts `+0.17`, mid_fga `+0.14`, rim_fga `+0.13`, usg `+0.07`, but it *hurts*
  blk `-0.09` (CI below zero) and mid_pct `-0.02`, and is slightly negative
  on ast, stl, dreb. Sixteen extra expanding-window betas add estimation
  noise where the cross-skill signal is weak. The right version is a
  per-skill choice or a ridge-shrunk form block; round 3 tests E's form
  block together with `jump` so the per-skill best can be read off one file.
- **Clipping adds nothing on top** (C − B: fta `+0.03`, usg `+0.02`). Drop it.
- **Inside `ctxplus`, the playoff term is the finding.** `ctxps` alone
  carries ast `+0.24` of `+0.24`, pf `+0.20` of `+0.21`, pts `+0.13` of
  `+0.14`, usg `+0.07` of `+0.10`, rim_pct `+0.04`. The minutes trend alone
  is worth usg `+0.03`, fta `+0.02`, pts `+0.02`, three_fga `+0.02` and is
  null elsewhere; keep it for the volume skills, but it is not the story.

## Round 3: the union, and the per-skill recipe

Variants: D = `ctxplus+ctxmult+xusg+jump` (round-2 best general recipe),
F = `ctxplus+ctxmult+xall+jump` (all-skill form instead of usage form),
G = `ctxps+ctxmult+xusg+jump` (D without the minutes trend).

Variant vs base:

| skill | ctxplus+ctxmult+xusg+jump | ctxplus+ctxmult+xall+jump | ctxps+ctxmult+xusg+jump |
|---|---|---|---|
| rim_pct | +0.051 [+0.028,+0.075] 5/6 | +0.045 [+0.011,+0.081] 5/6 | +0.053 [+0.034,+0.075] 6/6 |
| mid_pct | +0.005 [-0.007,+0.017] 5/6 | -0.015 [-0.035,+0.005] 3/6 | +0.007 [-0.005,+0.019] 5/6 |
| three_pct | +0.013 [-0.000,+0.027] 5/6 | +0.015 [-0.008,+0.038] 5/6 | +0.010 [-0.002,+0.022] 4/6 |
| ft_pct | +0.021 [+0.003,+0.040] 4/6 | +0.010 [-0.012,+0.030] 4/6 | +0.016 [+0.001,+0.031] 4/6 |
| rim_fga_rate | +0.157 [+0.103,+0.210] 6/6 | +0.285 [+0.215,+0.357] 6/6 | +0.153 [+0.100,+0.209] 6/6 |
| mid_fga_rate | +0.567 [+0.457,+0.683] 6/6 | +0.705 [+0.567,+0.831] 6/6 | +0.573 [+0.468,+0.683] 6/6 |
| three_fga_rate | +0.255 [+0.190,+0.314] 6/6 | +0.301 [+0.227,+0.375] 6/6 | +0.225 [+0.165,+0.280] 6/6 |
| fta_rate | +0.261 [+0.196,+0.328] 6/6 | +0.299 [+0.223,+0.379] 6/6 | +0.237 [+0.167,+0.300] 6/6 |
| pts_rate | +0.340 [+0.279,+0.398] 6/6 | +0.504 [+0.426,+0.582] 6/6 | +0.334 [+0.282,+0.386] 6/6 |
| ast_rate | +0.439 [+0.358,+0.519] 6/6 | +0.417 [+0.330,+0.502] 6/6 | +0.438 [+0.348,+0.525] 6/6 |
| tov_rate | +0.087 [+0.041,+0.130] 6/6 | +0.321 [+0.235,+0.400] 6/6 | +0.089 [+0.043,+0.140] 6/6 |
| oreb_rate | +0.075 [+0.035,+0.115] 5/6 | +0.111 [+0.062,+0.160] 4/6 | +0.080 [+0.041,+0.120] 5/6 |
| usg_pct | +0.207 [+0.158,+0.254] 6/6 | +0.270 [+0.195,+0.343] 6/6 | +0.183 [+0.133,+0.234] 6/6 |
| dreb_rate | +0.060 [+0.024,+0.095] 4/6 | +0.043 [+0.001,+0.082] 5/6 | +0.063 [+0.029,+0.099] 5/6 |
| stl_rate | +0.041 [+0.014,+0.066] 6/6 | +0.017 [-0.017,+0.049] 5/6 | +0.033 [+0.008,+0.059] 6/6 |
| blk_rate | +0.364 [+0.277,+0.460] 6/6 | +0.277 [+0.193,+0.363] 6/6 | +0.359 [+0.267,+0.455] 6/6 |
| pf_committed_rate | +0.253 [+0.201,+0.306] 6/6 | +0.291 [+0.231,+0.346] 6/6 | +0.237 [+0.187,+0.285] 6/6 |

Pairwise increments:

| skill | all-skill form vs usage form (F − D) | minutes trend increment (D − G) |
|---|---|---|
| rim_pct | -0.005 [-0.033,+0.025] 5/6 | -0.002 [-0.008,+0.002] 2/6 |
| mid_pct | -0.020 [-0.037,-0.002] 1/6 | -0.002 [-0.004,+0.001] 3/6 |
| three_pct | +0.001 [-0.019,+0.023] 4/6 | +0.003 [-0.006,+0.011] 5/6 |
| ft_pct | -0.011 [-0.028,+0.005] 2/6 | +0.005 [-0.003,+0.014] 5/6 |
| rim_fga_rate | +0.127 [+0.079,+0.178] 5/6 | +0.004 [-0.007,+0.016] 4/6 |
| mid_fga_rate | +0.139 [+0.055,+0.231] 5/6 | -0.006 [-0.014,+0.000] 3/6 |
| three_fga_rate | +0.047 [-0.001,+0.095] 5/6 | +0.029 [+0.013,+0.044] 5/6 |
| fta_rate | +0.038 [-0.004,+0.083] 4/6 | +0.024 [+0.009,+0.038] 5/6 |
| pts_rate | +0.165 [+0.112,+0.218] 6/6 | +0.006 [-0.015,+0.027] 3/6 |
| ast_rate | -0.022 [-0.061,+0.016] 4/6 | +0.002 [-0.011,+0.013] 3/6 |
| tov_rate | +0.234 [+0.174,+0.290] 6/6 | -0.002 [-0.004,+0.001] 1/6 |
| oreb_rate | +0.036 [-0.005,+0.078] 4/6 | -0.005 [-0.014,+0.004] 3/6 |
| usg_pct | +0.063 [+0.006,+0.121] 4/6 | +0.024 [+0.011,+0.036] 5/6 |
| dreb_rate | -0.017 [-0.043,+0.009] 2/6 | -0.003 [-0.015,+0.010] 2/6 |
| stl_rate | -0.024 [-0.045,-0.001] 2/6 | +0.008 [-0.004,+0.020] 5/6 |
| blk_rate | -0.088 [-0.132,-0.042] 1/6 | +0.005 [-0.007,+0.018] 4/6 |
| pf_committed_rate | +0.038 [+0.010,+0.066] 5/6 | +0.016 [-0.008,+0.039] 5/6 |

Reading:

- **The all-skill-form split is stable under `jump`** (F − D): pts `+0.17`,
  tov `+0.23`, mid_fga `+0.14`, rim_fga `+0.13`, usg `+0.06`, pf `+0.04`
  with CIs above zero; blk `-0.09`, stl `-0.02`, mid_pct `-0.02` with CIs
  below zero; ast, dreb, rim_pct, ft_pct slightly negative and crossing.
  Choose the form block per skill.
- **The minutes trend is worth keeping only for volume skills** (D − G):
  three_fga `+0.03`, fta `+0.02`, usg `+0.02`, everything else null.
- **Nothing here moves mid_pct or three_pct.** Their best variants are
  `+0.007` and `+0.015` with CIs crossing zero; shooting percentages are
  already at the OU + corrections ceiling under this feature set. rim_pct
  gains `+0.05` (playoff term), ft_pct `+0.02` (jump).

### Selected recipe (dev, vs re-tuned v4)

| skill | recipe | gain [95% CI] wins |
|---|---|---|
| mid_fga_rate | F: playoff + minutes trend, multiplicative context, all-skill form, team-change jump | +0.705 [+0.567,+0.831] 6/6 |
| pts_rate | F | +0.504 [+0.426,+0.582] 6/6 |
| ast_rate | D: same with usage form only | +0.439 [+0.358,+0.519] 6/6 |
| blk_rate | D | +0.364 [+0.277,+0.460] 6/6 |
| tov_rate | F | +0.321 [+0.235,+0.400] 6/6 |
| three_fga_rate | F | +0.301 [+0.227,+0.375] 6/6 |
| fta_rate | F | +0.299 [+0.223,+0.379] 6/6 |
| pf_committed_rate | F | +0.291 [+0.231,+0.346] 6/6 |
| rim_fga_rate | F | +0.285 [+0.215,+0.357] 6/6 |
| usg_pct | F | +0.270 [+0.195,+0.343] 6/6 |
| oreb_rate | F | +0.111 [+0.062,+0.160] 4/6 |
| dreb_rate | G: playoff only, usage form, jump | +0.063 [+0.029,+0.099] 5/6 |
| rim_pct | G | +0.053 [+0.034,+0.075] 6/6 |
| stl_rate | D | +0.041 [+0.014,+0.066] 6/6 |
| ft_pct | D | +0.021 [+0.003,+0.040] 4/6 |
| three_pct | – (null) | +0.015 [-0.008,+0.038] 5/6 |
| mid_pct | – (null) | +0.007 [-0.005,+0.019] 5/6 |

Fifteen of seventeen skills beat re-tuned v4 with the CI above zero, twelve
in 6/6 seasons; the median gain across all 17 is `+0.29%`, roughly half of
v4's own step over plain OU (`+0.56%`). Ranked by what each idea is worth:

1. **Multiplicative context for rate skills** (`ctxmult`): up to `+0.47`
   (mid_fga) and `+0.35` (blk) as an increment over the same additive
   context.
2. **Cross-skill form**: usage form alone `+0.06..+0.10` on fta/pts/tov;
   the full 16-skill block a further `+0.13..+0.23` on rim_fga/mid_fga/
   pts/tov but harmful on blk/stl/mid_pct.
3. **Playoff regime** (`ctxps`): `+0.24` ast, `+0.20` pf, `+0.13` pts,
   `+0.07` usg, `+0.04` rim_pct; the only idea that helps a pct skill.
4. **Team-change variance inflation** (`jump`): `+0.12` usg, `+0.07`
   pts/ast, `+0.04..+0.05` on the attempt rates.
5. **Minutes trend**: `+0.02..+0.03` on three_fga/fta/usg only.

Per-skill recipe selection on dev is itself a mild selection step (three
candidates per skill); the differences between D, F and G are within the CI
for oreb, dreb, stl, ft_pct and rim_pct, so a builder may use one recipe per
skill *kind* (F for attempt/scoring/usage/tov/pf rates, D for ast/blk/stl,
G for rebounds and pct) without losing anything measurable.

## Literature scan: what was tested here and what was not

Sources read: DARKO (Medvedovsky) method notes, Dunks & Threes E-Skills
notes, Ötting et al. OU hot-hand model, Glickman-Stern offseason variance,
LEBRON archetype priors, EPAA compositional shot diet, EPM predicted-minutes
write-ups, DraftKings log-odds shooting priors.

| idea from the literature | status here |
|---|---|
| DARKO cross-stat coupling (one stat's residual informs another) | tested as `xusg` (usage → others) and `xall` (all 16 others → each skill, round 2) |
| DARKO team-change learning-rate bump | tested as `jump` (extra slow-state process noise on a team-id change, rounds 2-3): confirmed, `q_jump` 5-25% of `s_mu` |
| E-Skills per-stat clocks (days vs possessions) | already in the lane: `dt = max(days_gap, 1)` with `tau` per skill; exposure enters via `r/exposure` |
| within-season league trend | tested as `center` (as-of league centering) — loses on rates; pct skills already centred |
| logit / binomial / Poisson observation noise | tested as `noise` — loses on rates, null on pct |
| innovation clipping / robust update | tested as `clip` — `+0.03` fta, `+0.02` usg, otherwise null |
| playoff regime | tested as `ctxps` inside `ctxplus` — first documented next-game playoff term, positive |
| predicted minutes / rotation trend | tested as `ctxmin` (minutes trend) inside `ctxplus` |
| padding constants (3P% ≈ 242 attempts) | implicit in the tuned `s_mu`, `r`; not separately tested |
| role / archetype priors (LEBRON) | **not tested** — needs a position/archetype table; candidate for the debut prior |
| compositional shot diet (multinomial rim/mid/three) | **not tested** — would replace three independent `fga_rate` filters with one simplex state |
| Glickman-Stern offseason variance by age | partly covered by `q_off` + the age level shift; an age-dependent `q_off` is untested |
| GBDT combiner on top of the filter | **not tested** in NBA; the WNBA fusion lane was a null on far less data, so the expected value is low |
| official inactive list at tip (EPM 12.14→12.10 RMSE) | **not tested** — would need a timestamped pre-tip snapshot pipeline; roster-history absence is the fair stand-in |

## The harness (two meanings)

**1. Next-game lab harness** — `scripts/research_skills_nextgame_lab.py`.
One command tests a modelling theory against v4 on the owner's target with a
symmetric optimiser and paired, game-clustered uncertainty:

```bash
.venv/bin/python nba_pipeline/scripts/research_skills_nextgame_lab.py \
  --variants base,ctxplus+ctxmult+xusg,ctxplus+xusg,ctxps,ctxmin \
  --maxiter 60 --popsize 15 --polish 600 --boot 1000 --workers 17 --tag round2
```

- Loads the panel with 2026 dropped before anything else runs; 2027 is untouched.
- `prepare()` rebuilds v4 exactly per skill (stage-A residuals, the v4-selected
  shrinkage `c`, expanding-window context betas, stage-B corrections) and the
  worker asserts closure: v4's stored params pushed through the lab engine
  reproduce `obj_v4` to machine precision (17/17 skills, gap 0).
- `assemble(prep, opts)` turns a set of option names into `(run, bounds, names)`.
  Options compose with `+` (`ctxplus+ctxmult+xusg`); `--combo a,b,c` is the
  legacy spelling. Options: `fixr noise ctxplus ctxps ctxmin ctxmult center
  clip xusg xall jump`.
- `tune()` = the v4 DE (same seed 20260826, maxiter 60, popsize 15) followed
  by a log-space Nelder-Mead polish that is kept only if it improves. Base and
  every variant get the identical optimiser, so "base vs v4" measures the polish
  and "variant vs base" measures the model change.
- `evaluate()` = pooled % gain, game-clustered paired bootstrap 95% CI (1000
  reps), per-season wins /6. Every variant is scored vs base and vs v4, and
  every pair of run variants is scored against each other under `pairs`, so
  an increment (A minus B) carries its own CI. Output JSON per tag under
  `results/estimated_skills/nextgame_lab/` (ignored artifacts); `round1.json`,
  `round2.json` and `round3.json` are the receipts for this section.
- Parallel across skills with a spawn pool and per-worker `load_all` initializer
  (fork crashes on macOS inside numpy/pandas once the parent has touched BLAS;
  the panel loads in 0.5 s so re-loading per worker is free). `--serial` runs
  in-process for tracebacks. Full round of eight variants: ~20 min wall on
  17 cores.
- Engine test: `tests/test_estimated_skills_nextgame_lab.py` (lab engine ==
  v2 engine with options off; variance params jointly scale-invariant; options
  change predictions and stay finite).

**2. VPM integration seam** — the sibling
`../vpm/vpm/harness/engine.py::build_prior_feature_panel` originally dispatched
`completed_season_box`, `epm_predictive_snapshot`, `epm_box_tracking`. The own-skills
contract requires (a) a dated exporter from the filter: one row per
`(nba_id, estimate_date)` carrying the **neutral** posterior level
(`post` from the filter, i.e. mu + delta with corrections, *without* the
next-game context transient) plus `post_var`, exposure, version/hash, and
(b) a fourth `prior_feature_source` branch that reads that table. Next-game
context (opponent/team/rest/absence/playoff/form) must stay in the game
forecast layer, never in the portable prior. September 14: both pieces now exist in the separate
`export_vpm_own_skills.py` / `own_estimated_skills` v15 integration lane.
The initial 582-player O/D parent and development replay are documented in
[Own-Skill Parent Integration](../../../vpm/vpm/docs/OWN_SKILL_PARENT_INTEGRATION.md).
This admits the data only to an explicitly labeled integration experiment;
no production metric or next-game recipe was promoted. Fixed 2020-2025
parameters do not make historical replay untouched evidence.

## Status

- v4 remains the selected candidate on disk (`v4.json`). The per-skill
  recipe above is a documented dev result with tuned parameters and
  last-window betas in `round3.json`; promoting it means wiring the
  option set into a `v5` builder that writes the same output contract as
  v4 (the lab's `assemble()` is the specification), then the owner's 2026
  decision. **2026 remains unscored by every variant here.**
- Nothing here touches ratings, priors, or VPM inputs.

# NBA estimated-skills family map (2026-09-14): every lane, and what the lab says about each

The estimated-skills program is no longer one script. It is one shared filter
engine (`build_nba_estimated_skills.py::run_filter` and
`build_nba_estimated_skills_v2.py::run_filter_v2`) plus a growing set of
lanes on different data sources, each with its own doc, receipts and holdout
state. This section is the inventory. Read it before answering "how are our
estimated skills doing?" — the answer has to cover all of these, not only the
17-skill PBP panel.

Holdout convention shared by every lane: 2019 is warm-up, 2026 is at most
export-only or already-spent limited evidence, and 2027 is untouched. Nothing
in any lane feeds a rating prior, a solve prior, or a VPM input.

## Lane inventory

| # | Lane | Script(s) / doc | Source | Estimates | Objective and split | Status and evidence |
|---|---|---|---|---|---|---|
| 1 | PBP Kalman panel (17 skills) | `build_nba_estimated_skills{,_v2,_v2_stageb,_v3,_v4}.py`; `estimated_skills_context.py`, `estimated_skills_corrections.py`; this doc | resolved PBP player-game panel | rim/mid/three/FT pct; rim/mid/three FGA, FTA, pts, ast, tov, oreb, dreb, stl, blk, pf per 100; usg | next-game exposure-weighted WMSE; dev 2020-2025 | v4 selected (17/17 skills, 101/102 seasons vs OU, median +0.56%). Lab rounds 1-3 add a per-skill recipe worth a further median +0.29% (15/17 CI>0). No v5 builder yet. 2026 partially exposed by the v3 forward confirmation; never next-game scored. |
| 2 | Next-game deflections OU | `build_deflections_next_game.py`; `NBA_DEFLECTIONS_NEXT_GAME.md` | Gabriel `player_sheets` hustle game logs joined to PBP defensive possessions | deflections per 100 defensive possessions | next-game WMSE; warm-up 2019, tune 2020-2022, stress 2023-2025 (spent), 2026 export-only | `next_ou` beats the longer-horizon OU by 0.328% [0.258, 0.400] and the next-game random walk by 0.129% [0.057, 0.202] on stress; naive baselines lose 4.7-10.6%. 581 current 2026 states. Plain OU: no corrections, no context terms. |
| 3 | Longer-horizon estimated deflections | `build_estimated_deflections.py`, `audit_estimated_deflections.py`; `NBA_ESTIMATED_DEFLECTIONS.md` | same as 2 | same as 2, forward rate from 12 cutoffs | remaining-season rate (rejected as owner objective); tune 2020-2022, stress 2023-2025 | OU beats RW by 1.656% [0.682, 2.650] on stress, correlation 0.810 with future rate. Preserved only as the comparator for lane 2. |
| 4 | Deflections playoff offset | `analyze_deflections_playoff_context.py`, `analyze_deflections_playoff_elite.py`; `NBA_DEFLECTIONS_PLAYOFF_CONTEXT.md` | lane 2 panel | additive prior-season playoff observation offset (-0.18 to -0.39 per 100) | next-game WMSE on playoff rows | dev +0.0288% [-0.0154, +0.0749], stress -0.0231%. Not admitted; lane 2 unchanged. |
| 5 | Deflections-to-steals increment | `test_deflections_steals_increment.py`, `compare_enhanced_steals_epm.py`; `DEFLECTIONS_STEALS_INCREMENT.md` | lane 1 steals posteriors + lane 2 pregame deflections | ridge residual correction to next-game steals per 100 | one-SE on 2021-2022, stress 2023-2025 (spent); same-day 2026 EPM comparison on 10 dates | dev +0.2925% [0.1736, 0.4122]; stress +0.4103% [0.3220, 0.4983] vs steals-only calibration; coefficient 0.23-0.28 extra steals per deflection in every expanding fit. 2026 same-day vs EPM +0.2854% [-0.5788, +1.1420], inconclusive. Not exported to any table. |
| 6 | Tracking game-log backfill | `build_tracking_backfill.py`; `NBA_TRACKING_BACKFILL.md` | Gabriel `player_sheets/game_report/year_files` 2014-2026 (355,995 rows) | repaired potential assists (1,586 rows) and DFGA (746 outage rows, 11,498 inferred zeros, 1,492 unresolved) | postgame imputation only; 2015-2025 out-of-year RMSE (PA 1.843 vs 2.440 prior history) | preparation layer for future tracking skills. Never feed `*_filled` values as independent observations; not a pregame forecast. |
| 7 | Tracking playoff quartiles | `analyze_tracking_playoff_quartiles.py`; `TRACKING_PLAYOFF_QUARTILES.md` | lane 6 measured-only panel | RS-to-PS rate change by RS quartile for PA, drives, possession minutes, contested DREB, contested shots, DFGA, deflections | descriptive, 2020-2025 | top-quartile PA -1.68 and deflections -0.48 per 100; regression to the mean not removed. Deflections has lane 2; the other six now have next-game filters in lane 19. |
| 8 | Rim contest savings + matchup burden | `build_rim_defense_matchup_skills.py`; `RIM_DEFENSE_MATCHUP_SKILLS.md` | `leaguedashptdefend` <6 ft game logs; BAM matchup rows | `2*(FGA*normal_pct - FGM)` saved points; partial-possession scoring/passing burden from prior-season own posteriors | measurement layer only | 196,922 valid player-games (96.04% of defensive exposure); 359 qualified current rows. Provider normal-percentage baseline is retrospective, so this is not a model. |
| 9 | Rolling predictive rim points saved | `build_predictive_rim_saved.py`, `compare_rim_saved_blocks.py`; `PREDICTIVE_RIM_SAVED.md` | lane 8 counts, 2023+ only (unprefixed 2020-2022 fields were season totals and were rejected) | dated contest rate and finishing suppression, combined to saved points per 100 | next-game WMSE; dev 2023 only, stress 2024-2025, 2026 export | 730-day / K=100. Saved points vs zero suppression +1.220% [0.986, 1.450]; allowed rim points +0.323% [0.139, 0.486]; vs blocks-only +0.091% [-0.092, 0.252] (not resolved); blocks + rim vs blocks-only +0.249% [0.163, 0.330]. 577 current dated states. Exponential-decay estimator, not the OU engine. |
| 10 | Rim blocked-shot prior | `research_rim_blocked_prior.py`; `RIM_BLOCKED_PRIOR.md` | box BLKA / FGA, strictly prior | extra rim_pct feature | one-feature increment on lane 1 rim_pct | fails dev (pct MSE -0.00323%, joint rim FGM/100 -0.00744%) and stress. No table change. |
| 11 | Three-point volume prior | `research_three_pct_volume_prior.py`, `export_three_pct_volume_prior.py`; `THREE_PCT_VOLUME_PRIOR.md` | lane 1 three_pct with cumulative prior 3PA/100 | population prior curve in log volume, blended at strength 0.5 | conditional 3P% squared error; dev selects 0.5, stress 2023-2025 | +0.0607% [0.0343, 0.0878] overall; +3.02% [0.64, 5.71] for <1 3PA/100 and +1.23% [0.66, 1.80] for <3. Owner-authorized: the current-player and EPM tables use it for 3P% only, exported through 2026 observations without scoring 2026. |
| 12 | Shooting fusion | `research_shooting_fusion.py`; `SHOOTING_FUSION.md`, `SHOOTING_FUSION_RESULTS.md` | lane 1 posteriors as chronological cross-skill features | ridge / boosted 3P% corrections | conditional log loss plus a joint 3PM/100 makes gate | linear ridge alpha 1000 wins conditional log loss but fails the makes gate; selected volume+FT prior has stress gains of only 0.0094% / 0.0473% / 0.0554% and no confirmed all-player makes increment over volume alone. |
| 13 | Physical features | `estimated_skills_physical.py`; `ESTIMATED_SKILLS_PHYSICAL_FEATURES.md` | Databallr `measurements_master` (1,835 rows, undated) | no-shoes height, wingspan, difference, missing flags | retrospective candidate features for lane 12 | 509/582 current players covered. Fusion physical linear 1000 fails the joint gate (-0.0919%). Undated, so historical use is guarded. |
| 14 | Team-game and league seasonality | `research_team_game_seasonality.py`, `analyze_league_seasonality.py`; `TEAM_GAME_SEASONALITY.md`, `LEAGUE_SEASONALITY.md` | lane 1 innovations by team game number; league shooting by calendar | game-number cubic; descriptive within-season drift | one-SE on 2021-2022 | cubic selects zero. Simple context helps 2023-2025 only (pct +0.0538%, makes +0.0766%), not early dev. EPM caches are `game_optimized=0`. No promotion. |
| 15 | Attempt rates vs EPM | `compare_attempt_rates_next_game_epm.py`; `ATTEMPT_RATES_NEXT_GAME_EPM.md` | lane 1 **v1** posteriors vs 12 cached EPM dates, Oct 15, 2025 - Apr 1, 2026 | rim/mid/three FGA per 100 | same-day and next-appearance paired MSE, 2026 | same-day inconclusive (rim +0.69% [-0.98, +2.26]); next appearance within 7 days rim +1.53% [+0.49, +2.51]. This is spent 2026 evidence for v1, not v4. |
| 16 | EPM E-Skills benchmark | `compare_nba_estimated_skills_to_epm.py`; results `epm_benchmark/` | lane 1 posteriors vs the same 12 EPM snapshot dates | pooled remaining-season rate correlation and level-adjusted MSE | 2026 | aggregate-only receipts; provider values never copied. Spent 2026 evidence. |
| 17 | Current skill comparison table | `build_current_skill_comparison.py`; `CURRENT_SKILL_COMPARISON.md`; results `current_player_table/` | lane 1 v1 (3P% from lane 11), EPM June 15, 2026 snapshot, DARKO upload (516/530 matched) | reader table with pairwise differences | display only | still v1 for every PBP skill except 3P%; v4 and the lab recipe are not on the table. |
| 18 | 2026 forward confirmation (v3) | `confirm_estimated_skills_2026.py`; results `confirmation_2026.json` | lane 1, v1 vs v3 frozen | rest-of-season forward aggregate | one-shot 2026 | 15/17, median +4.81% on a rejected objective. This is why 2026 is "partially exposed" for lane 1. |
| 19 | Next-game tracking filters (six stats) | `build_tracking_next_game.py`; `NBA_TRACKING_NEXT_GAME.md`; results `tracking_next_game/` | lane 6 measured-only panel joined exactly to PBP possessions (217,163 rows; `*_filled` never observations) | potential assists, drives, possession minutes per 100 offensive possessions; contested shots, DFGA, contested DREB per 100 defensive possessions | next-game WMSE; warm-up 2019, tune 2020-2022 (OU vs RW, one SE), stress 2023-2025 (spent), 2026 export-only | all six select `next_ou`. Stress vs re-tuned RW: possession minutes +2.480% [2.088, 2.871], drives +1.144% [0.952, 1.341], PA +0.858% [0.706, 1.015], DFGA +0.269% [0.184, 0.359], contested DREB +0.127% [0.062, 0.189], all 3/3; contested shots reverses to -0.098% [-0.189, -0.010], 0/3 (frozen, not switched). Naive baselines lose 2.8-38.8%. 578-582 current 2026 states. Plain OU: no context terms. |
| 20 | Next-game fouls-drawn filters (three stats) | `fetch_fouls_drawn_cdn.py`, `build_fouls_drawn_game_lines.py`, `build_tracking_next_game.py --family fouls_drawn`; `NBA_FOULS_DRAWN_NEXT_GAME.md`; results `fouls_drawn/`, `fouls_drawn_next_game/` | NBA live CDN play-by-play (`foulDrawnPersonId`, 2019-20 to 2025-26), reconciled against repo PBP committed fouls and the official box archive: 8,543 of 8,879 games measured; 266 CDN objects missing in 2019-20, 62 stat-correction games and 8 identity failures unmeasured, never zero | offensive fouls drawn (charges plus non-charge) and charges drawn per 100 defensive possessions; PFD per 100 offensive possessions | next-game WMSE; warm-up 2020, tune 2021-2022 (OU vs RW, one SE), stress 2023-2025 (spent), 2026 export-only | all three select `next_ou`: PFD clearly (dev +0.287% [0.171, 0.400]; stress +0.348% [0.244, 0.446], 3/3), offensive fouls and charges narrowly (dev +0.062% and +0.032%, intervals crossing zero) yet holding on stress (+0.116% [0.038, 0.194] and +0.116% [0.040, 0.202], 3/3). Naive baselines lose 1.8-8.5%. 582 current 2026 states. Plain OU: no context terms. |

Helpers used across lanes: `build_estimated_skills_age_lookup.py`,
`build_estimated_skills_game_context.py`, `eval_estimated_skills_forward.py`
(defaults to DEV_SEASONS), and the lab `research_skills_nextgame_lab.py`.

The curated stat-level inventory of this family — every stat on the values table, its table model, dev-selected model, receipt-backed evidence and readable state notes — is `config/estimated_skills_registry.json`, built into a status/values/evidence snapshot by `build_estimated_skills_registry.py` and explained in `NBA_PREDICTIVE_BOX_STATS_REGISTRY.md`.

## What the lab findings mean for each lane

The lab established, on the PBP panel, which ideas move next-game error:
multiplicative context on rate skills (ctxmult), the playoff term (ctxps,
which carries most of ctxplus), team-change variance inflation (jump),
cross-skill form (xusg / xall, split by skill), and nothing for pct skills
except the playoff flag. Applied to the other lanes:

- **Next-game deflections (lane 2) is the direct port.** It runs
  `run_filter_v2` with no corrections and no context. Every lab winner is a
  candidate: shrunk opponent and own-team deflection residuals entering
  multiplicatively, a playoff flag plus playoff-times-excess term through the
  context regression (the additive offset of lane 4 failed; the lab form did
  not), team-change jump, and cross-skill form from stl / pf / usage
  posteriors. The catch is the split: lane 2 spent 2023-2025 as stress, so a
  deflections lab has to declare 2020-2025 as development (re-spending those
  seasons) and lean on 2027 for confirmation. That is a documented choice,
  not a silent one.
- **Deflections-to-steals (lane 5) is the cross-family version of xall.**
  The lab's cross-skill form was slightly negative for stl on PBP skills
  alone (F-D stl -0.024), while a hustle feature adds +0.41% on stress. In a
  v5 builder, deflections should enter stl as a context feature inside the
  filter rather than as a post-hoc ridge, and the same test should be run for
  pf and blk.
- **Rim saved (lane 9) should move onto the shared engine before any
  context work.** It is an exponential-decay estimator with a one-season dev
  window. Re-expressing contest rate and suppression as OU states (with the
  lab's DE + polish protocol) is the prerequisite; opponent rim-attempt
  context and team-change jump come after. Blocks-only is still not clearly
  beaten, so report that pair on every rerun.
- **Tracking stats now have the filter (lane 19); the playoff term is
  next.** The quartile declines of lane 7 are not context effects until
  ctxps is tested the lab way on top of the fitted OU, with regression to
  the mean handled by the filter itself. Lane 19 spent 2023-2025 as stress
  exactly as lane 2 did, so a tracking lab must declare 2020-2025 as
  development and lean on 2027; contested shots, whose OU gain reversed on
  stress, is the stat to re-examine first.
- **Fouls drawn (lane 20) has the same shape as lane 19 and the same
  spent stress.** Charges are rare (0.13 per 100 in the 2020 warm-up) and
  offensive fouls drawn are only three times more common, so their OU
  gains over the random walk barely cleared the one-SE gate; the two
  filters are near-equivalent and both beat the naive baselines by 7-8%.
  The multiplicative opponent term is the first lab candidate, because
  the fouling team's tendency is known before tip, and the playoff term
  second. A fouls-drawn lab must declare 2021-2025 as development and
  lean on 2027.
- **The pct-skill lanes (10, 11, 12, 13) are consistent with the lab:**
  extra features do not move rim/mid/three/FT pct. The one positive pct
  result in the family, the volume prior (lane 11), is an attractor idea,
  not a feature idea. The lab never tested a volume-dependent attractor for
  three_pct; that is the one untested pct candidate worth a lab option.
- **Seasonality (lane 14) is low priority.** The lab's minutes trend was
  small and its league-trend centering lost; the calendar cubic selected
  zero. Do not spend runs here.
- **The 2026 comparisons (lanes 15, 16, 5's EPM check) are all v1 or v1+.**
  Rerunning them with v4 or a v5 recipe would spend more of 2026. That is
  the owner's decision, and it should be a single pre-registered pass rather
  than a series of peeks.
- **The reader table (lane 17) is v1.** Promoting v4 or the lab recipe to
  the table is a separate owner decision from the 2026 scoring decision.
