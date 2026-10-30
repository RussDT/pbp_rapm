# PBP-Only Playtype Difficulty and Future oTS

## Question

Can ordinary NBA play-by-play descriptions recover enough of a shooter's playtype difficulty to predict future offensive true shooting (`oTS`) as well as the existing Synergy-derived `playtype_diff` feature?

This is a player shooting-context experiment. It is not the existing `PLAYTYPE_PROXY_PTS` lineup RAPM, which asks how a player's minutes change the team's shot diet.

## Leakage-safe PBP proxy

Each regular-season FGA is assigned to one of six broad, outcome-invariant categories:

- `CUT_2`
- `SELF_CREATED_3`
- `OTHER_3`
- `SELF_CREATED_2`
- `OTHER_RIM_2`
- `OTHER_2`

The classifier uses shot value and neutral description tokens such as `cutting`, `alley oop`, `pull-up`, `step-back`, `driving`, `floating`, `hook`, `dunk`, and `layup`. It does not use make/miss, actual points, score, or made-only assist tags.

For feature season `t`, the six categories are priced from completed regular season `t-1` only. Category points per FGA are shrunk toward the prior-season league mean with 100 FGA, and the feature-season player mix is then priced at those fixed values:

```text
PBP difficulty (TS percentage points)
  = 50 × (prior-season league points/FGA
          − prior-priced player shot-mix points/FGA)
```

A positive value means the player took a harder-than-average mix. Because the current player's makes and misses never enter the category assignment or price, the feature adjusts observed TS without directly reusing the player's shooting result.

## Evaluation design

The source seasons are 2022-2025 regular-season PBP. They create feature seasons 2023-2025 and annual future-oTS targets for 2024-2026.

The primary evaluation is strictly chronological:

- Predict 2025 after fitting the feature-to-future-oTS mapping only on 2023 → 2024.
- Predict 2026 after fitting only on 2023 → 2024 and 2024 → 2025.

Every candidate is scored on the exact same 735 player-seasons and 440 players. Eligibility requires at least 100 feature-season FGA, 250 minutes, and a resolved future-oTS target. Ridge alpha is selected on the floating scoring baseline training rows and then held equal across all candidates in that fold. Model-specific alpha results are exported only as sensitivity checks.

The primary specification is the decompressed or "floating three" scoring
model. Raw personal points per 100 possessions (`pts100`), role-difficulty
points per 100, and true-shot attempts per 100 (`TSA100`) enter separately.
This lets the regression learn its scoring break-even instead of imposing a
fixed add baseline or using the legacy `TS_added_100` feature. Every model also
contains the same three feature-season creation controls:

- on-ball time percentage
- potential assists per 100 possessions
- rim assists per 100 possessions

These controls are standardized and ridge-shrunk with the other regressors.
They prevent an own-shot difficulty feature from receiving easy credit for the
same creator burden that independently predicts teammate shooting and future
team `oTS`. Rim assists remain a noisy made-shot-dependent control; potential
assists are the cleaner opportunity measure, but the preregistered primary
comparison includes all three identically. The playtype candidates enter as
`role_added_100 = 2 * TSA100 * difficulty / 100`, so the regression can estimate
their coefficients rather than imposing the production zTS coefficient of
`1.0`:

- `box_scoring_baseline`: `pts100 + TSA100` plus the three creation controls.
- `pbp_role_added`: baseline plus PBP role-added value.
- `synergy_role_added`: baseline plus Synergy role-added value.
- `both_roles_added`: baseline plus both role-added values.

Uncertainty is estimated by paired bootstrap resampling clustered by player.

## Primary creation-controlled chronological results

| Model | Future-oTS RMSE | Change vs box | RMSE reduction |
|---|---:|---:|---:|
| Creation-controlled floating baseline | 1.4940 | — | — |
| **PBP role added** | **1.4744** | **-0.0197** | **1.32%** |
| Synergy role added | 1.4860 | -0.0080 | 0.54% |
| Both roles added | 1.4856 | -0.0084 | 0.56% |

The creation controls explain much of the original apparent playtype gain. The
PBP increment over the baseline is small but its paired interval clears zero
when role value is estimated separately from raw scoring. Synergy's increment
and PBP's head-to-head edge over Synergy do not clear zero.

| Paired comparison | RMSE change | Player-clustered 95% interval | P(improves) |
|---|---:|---:|---:|
| PBP vs box | -0.0197 | [-0.0383, -0.0024] | 98.90% |
| Synergy vs box | -0.0080 | [-0.0269, +0.0104] | 80.25% |
| Synergy after PBP | +0.0113 | [-0.0003, +0.0233] | 2.88% |
| PBP after Synergy | -0.0004 | [-0.0058, +0.0048] | 55.00% |
| Synergy vs PBP | +0.0116 | [-0.0004, +0.0238] | 3.23% |

The PBP model still wins the point estimate in both scored target seasons, but
the 2025 difference is small:

| Future season | Box | PBP | Synergy |
|---|---:|---:|---:|
| 2025 | 1.5015 | **1.4851** | 1.5003 |
| 2026 | 1.4865 | **1.4635** | 1.4714 |

### Control ladder

The same common 735-row scored sample shows where the original advantage went:

| Controls | Box | PBP | Synergy | PBP minus Synergy |
|---|---:|---:|---:|---:|
| Shooting only | 1.5245 | 1.4942 | 1.5155 | -0.0213 |
| + on-ball percentage | 1.5163 | 1.4977 | 1.5137 | -0.0160 |
| + potential assists/100 | 1.4971 | 1.4768 | 1.4885 | -0.0117 |
| + rim assists/100 | 1.4940 | 1.4744 | 1.4860 | -0.0116 |

On-ball role absorbs part of the playtype signal, and potential assists absorb
another large part. Rim assists add little once the broader opportunity measure
is present. This confirms the omitted-variable concern: the uncontrolled PBP
feature was partly standing in for playmaking, not only personal shot burden.

### Estimated role coefficient

Because raw scoring and role-added value are now separate, each chronological
fold produces a physical coefficient for both. Dividing the role coefficient
by the raw `pts100` coefficient gives the intended floating-three role
multiplier relative to ordinary personal scoring points:

| Scored target | PBP role coefficient | Raw-points coefficient | PBP multiplier | Synergy multiplier |
|---|---:|---:|---:|---:|
| 2025 | 0.222 | 0.289 | **0.767** | 1.562 |
| 2026 | 0.260 | 0.334 | **0.779** | 1.141 |

The two controlled PBP folds are unusually stable near `0.77-0.78`, close to
the separate descriptive scale audit near `0.81` and below the production zTS
coefficient of `1.0`. The earlier `0.45-0.55` result was an invalid denominator
comparison against `TS_added_100`, not the intended floating-three audit. The
selected ridge alphas are `1000` and `316.2`; this remains a predictive
future-oTS coefficient, not a reader-facing zTS calibration receipt. With only
two scored folds, freeze it as a diagnostic rather than installing `0.78` in
production.

## How close is it to Synergy?

Across 1,097 common feature-season rows, PBP difficulty and Synergy `playtype_diff` have Pearson correlation `0.908` and Spearman correlation `0.910`. Season-level Pearson correlations are `0.913`, `0.913`, and `0.899` for 2023, 2024, and 2025.

So the PBP proxy recovers most of Synergy's broad shooter-difficulty ordering.
After creator-role controls, it beats Synergy by `0.0116` RMSE at the point
estimate and improves the floating scoring baseline by `0.0197`. Only the PBP
increment versus baseline clears zero; the PBP-versus-Synergy interval is
`[-0.0004,+0.0238]` when expressed as Synergy minus PBP. The evidence supports
PBP as the leading development candidate, not as a proven Synergy replacement
or an admitted production feature.

This does not mean six PBP buckets recover every Synergy playtype. PBP descriptions cannot fully identify pick-and-roll ball-handler possessions, handoffs, off-screen actions, screening, matchup pressure, or turnover-bearing playtype possessions. The conclusion is narrower: they recover the part of broad shot-role difficulty that mattered for this future-oTS test.

## Decision

Keep the PBP feature as the leading development challenger to the Synergy
playtype adjustment, but keep the claim narrow: its uncontrolled increment was
partly omitted playmaking, while its separately estimated creation-controlled
increment is small and clears the paired baseline interval. Its head-to-head
edge over Synergy remains inconclusive. Do not promote it from two spent folds.
Freeze the taxonomy, pricing rule, three creation controls, sample rule, and
model specification; require an untouched 2026-27 future-season test. If that
passes, the next gate is incremental prediction of summed future possession
outcomes in the canonical VPM harness, not another in-sample player-rate
comparison.

## Reproduction and artifacts

Run from `nba_pipeline/`:

```bash
python scripts/analyze_pbp_playtype_difficulty_ots.py
```

The default outputs are under `results/pbp_playtype_difficulty/2023_2026/`:

- `manifest.json`: frozen inputs, feature definition, sample rule, and temporal split.
- `pbp_playtype_player_season.csv`: player-season proxy values and category shares.
- `pbp_playtype_category_prices.csv`: prior-season category prices.
- `pbp_vs_synergy_overlap.csv`: Pearson and Spearman overlap.
- `chronological_metrics.csv`: primary creation-controlled common-alpha model metrics.
- `chronological_per_target_metrics.csv`: 2025 and 2026 results separately.
- `chronological_role_coefficients.csv`: fold-level physical and scoring-relative role coefficients.
- `creation_control_ladder_metrics.csv`: nested creation-control prediction audit.
- `creation_control_ladder_role_coefficients.csv`: coefficient movement across the same control ladder.
- `incremental_comparisons.csv`: paired player-cluster bootstrap comparisons.
- `legacy_loto_metrics.csv`: compatibility comparison using the older leave-one-transition-out protocol.

Focused tests live in `nba_pipeline/tests/test_pbp_playtype_difficulty_ots.py`.
