# WNBA Estimated Skills (Zone Kalman Panel + Fusion Gate)

Built August 26, 2026. E-Skills-style predictive shooting/volume skill panel
for the WNBA, extending the box panel
([WNBA_KALMAN_BOX.md](./WNBA_KALMAN_BOX.md)) to zone-level targets.
Implementation: `wnba_test/build_wnba_zone_game_lines.py`,
`wnba_test/build_wnba_estimated_skills.py`,
`wnba_test/build_wnba_skill_fusion.py`. Outputs live under
`wnba_test/Results/gameday/estimated_skills/` (generated artifacts, not in
git).

## Data: per-game zone shot lines

`build_wnba_zone_game_lines.py` parses rim/mid/three attempts and makes per
player-game from the modern normalized PBP
(`wnba_test/GabrielNormalizedClean/WNBA{YY}[_PS].csv`, 2022-2026), reusing
`build_wnba_pbp_shot_quality.build_shot_rows` verbatim (rim = dunk/layup/tip
shot family; mid = short_mid + long_mid + deep_two; three = shot value 3).
Reconciliation against `player_game_logs.csv` is exact: FGA, FGM, FG3A, and
FG3M agree with the box log for 100.0000% of joined player-games in every
season 2022-2026 (both surfaces are CDN-derived). Two log games lack PBP
coverage (`1022300075`, `1022500163`); 746 PBP rows sit outside the log and
all belong to whole games the gameday lane deliberately excludes — there are
zero player-identity mismatches. Zone mix moves as expected: rim share of FGA
0.334 → 0.363 and mid share 0.339 → 0.268 from 2022 to 2026.

Classic per-game files (2010-2021) carry no game dates, so pre-2022 warm-up
is out of scope until a date source is staged.

## Skill panel

`build_wnba_estimated_skills.py` fits one scalar Kalman filter per player per
skill with the box panel's engine generalized to an exposure column:

| Family | Skills | Exposure / obs variance | Source |
| --- | --- | --- | --- |
| Percentage | `rim_pct`, `mid_pct`, `three_pct`, `ft_pct` | attempts, `r / attempts` | zone lines (rim/mid), box log (3P/FT) |
| Volume | `rim_freq`, `mid_freq`, `three_freq`, `fta_rate` | minutes, `r / minutes` | zone lines (coverage-gated), box log |

Each skill runs on its own sub-panel (games with positive exposure, per-skill
day gaps and season crossings), so a game with zero rim attempts is no rim%
observation but a legitimate zero for rim frequency. Per-skill
`(q_game, q_day, q_offseason, r, v0)` are tuned by differential evolution
(seed `20260826`, `maxiter=30`, `popsize=12`, bounds scaled from pre-2026
variance only), exactly the box-panel protocol: one-step-ahead
exposure-weighted squared error in raw rate space, eligible = 2023-2025 with
at least one prior observation, 2022 warms the filters, **2026 is a
frozen-parameter holdout**.

Each percentage skill is fit twice — raw, and centered on the as-of league
mean (prior-season totals pooled with league season-to-date strictly before
the game date) — with the variant selected on development WMSE only.
**Uncentered won all four selections, but the result is a near-tie, not a
loss of signal**: centered is only +0.006% to +0.038% worse on dev and
microscopically better on 2026 for rim/mid/three (e.g. three_pct 0.062140 vs
0.062150). Mechanism: 2022-2026 league drift is small relative to per-game
binomial noise and the tuned `q_offseason` drift already re-anchors states at
season boundaries, so explicit centering is redundant on this window.

## Results (August 26, 2026 build)

All 8 skills beat all three exposure-weighted baselines (trailing-10,
season-to-date with career fallback, career-to-date) on development **and**
on the untouched 2026 holdout. Improvement vs trailing-10 / season-to-date:

| Skill | Dev vs t10 | Dev vs season | 2026 vs t10 | 2026 vs season |
| --- | --- | --- | --- | --- |
| rim_pct | +9.79% | +12.39% | +10.16% | +14.56% |
| mid_pct | +10.76% | +14.46% | +9.80% | +16.43% |
| three_pct | +9.53% | +12.28% | +12.28% | +16.28% |
| ft_pct | +9.47% | +15.29% | +11.80% | +17.57% |
| rim_freq | +6.32% | +7.52% | +6.42% | +8.98% |
| mid_freq | +5.77% | +6.77% | +5.13% | +7.94% |
| three_freq | +4.79% | +6.28% | +7.20% | +9.18% |
| fta_rate | +6.79% | +7.67% | +7.19% | +9.49% |

Panel: 24,932 played rows, 358 players. Pre-2026 exposure-weighted league
means: rim 57.7%, mid 39.2%, three 34.2%, FT 79.2%.

## LightGBM fusion layer: documented null

`build_wnba_skill_fusion.py` tests a DARKO-style second stage: per-skill
LightGBM over strictly-prior state features (all 14 box-stat posteriors plus
all 8 skill posteriors and variances via non-exact-match asof joins, schedule
context, as-of league mean, listed position; current-game exposure excluded),
season-expanding walk-forward (each dev season predicted by a model trained
strictly earlier; one frozen ≤2025 model predicts 2026), gated per skill on
dev only.

- **Direct-rate variant** (600 trees, lr 0.03): lost to plain Kalman on every
  skill, dev −7.9% to −12.8%, with losses shrinking monotonically as training
  seasons accumulated (three_pct −15.0% in 2023 → −5.8% by 2026) and gain
  spread nearly uniformly across ~40 features (top feature 6%) — diffuse
  noise-fitting under training-data starvation.
- **Residual variant** (fits `obs − kalman_pred`, 200 trees, heavy
  regularization): collapses the losses to −0.07% to −1.26% on dev but still
  passes 0/8 dev gates. The corrector finds essentially no cross-stat signal
  beyond the per-skill Kalman states on four WNBA seasons of data.

The DE-tuned Kalman panel therefore stands as the production reading; the
fusion lane stays in the repo as the recorded negative result (compare the v5
netcont-feat lesson: plausible feature stacks do not guarantee solve-level
gains). Because the direct-rate variant's 2026 numbers were viewed before the
residual revision, the fusion layer's 2026 column is report-only evidence —
weaker than the Kalman panel's untouched holdout, which was never part of any
fusion decision.

## Outputs

| File | Contents |
| --- | --- |
| `player_game_zone_lines.csv` | Per player-game rim/mid/three attempts+makes |
| `zone_reconciliation.json` | Box-log seam audit (exact-match shares, coverage) |
| `hyperparams.json` | Per-skill DE params for both centering variants, bounds, selection |
| `predictions.parquet` | One-step-ahead pred rate/var + posterior per eligible row (selected variant, raw space) |
| `current_states.csv` | The board: `p_{skill}` posterior + variance, career exposure, obs counts, per-40 volume columns |
| `eval_report.json` | Per-skill dev/holdout vs all baselines and both variants |
| `fusion_report.json`, `fusion_predictions.parquet` | Fusion gate evidence (null) |

## Fences

- The panel is a **props/forecast-lane input**, never a rating: nothing here
  enters the box-prior ridge, component solves, or any public RAPM/VPM
  surface. Only stabilized player-season summaries could ever flow toward
  rating priors, and that needs its own predeclared forward gate.
- Predicted rates are per-attempt (percentages) or per-minute (volumes);
  per-game projections require the minutes engine
  (`project_wnba_minutes.py`).
- 2026 stays out of every objective, bound, initialization, and variant
  selection until the season completes; in-season use is frozen-parameter
  filtering only.
- The `p_rim_freq`/`p_three_freq` states are natural candidates for the
  playtype-difficulty lane's "projected next-season role" regressor; that use
  is descriptive-forecast plumbing, not a rating change.

## Rebuild

```
.venv/bin/python wnba_test/build_wnba_game_logs.py          # refresh log first if stale
.venv/bin/python wnba_test/build_wnba_zone_game_lines.py
.venv/bin/python wnba_test/build_wnba_estimated_skills.py   # --quick for smoke
.venv/bin/python wnba_test/build_wnba_skill_fusion.py       # optional; gate evidence
```

States are as-of the game log's latest date (August 25, 2026 at the last
refresh); rerun the chain above to advance them. DE re-tuning is
deterministic and 2026-fenced, so a rebuild is a pure state refresh — the
August 26 log refresh (44 new finals) reproduced identical dev numbers and
kept all 8 holdout wins (2026 vs trailing-10 ranging +5.1% to +12.3%).

## Future work

- Pre-2022 percentage warm-up once a classic-era game-date source exists.
- Age-curve drift at season boundaries once a DOB table is staged.
- True on-court denominators (USG%, OREB%/DREB%, AST%) from the clean lineup
  surface — the remaining gap to a full E-Skills set.
- NBA→WNBA hierarchical transfer for the residual layer per the standing
  partial-pooling rule, as the answer to fusion's training-data starvation.
