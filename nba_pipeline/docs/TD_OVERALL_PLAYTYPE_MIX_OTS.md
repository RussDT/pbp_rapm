# Time-Decay Overall Playtype-Mix RAPM

## Decision

The overall playtype-mix parent is the right mathematical answer to the usage
problem, but it does not clear the future-oTS admission gate. Keep it as a
research diagnostic and do not add it to production zTS.

On the common 735 player-season chronological sample, adding the parent after
the PBP own-shot role feature improves RMSE from `1.4744` to `1.4703`. The
paired change is `-0.0040 [-0.0233,+0.0136]`, so the interval includes both
meaningful improvement and harm. The parent by itself is slightly worse than
the creation-controlled floating box baseline.

## Question

A teammate-only playtype effect is mechanically affected by usage. If a player
takes 35% of his lineup's shots, he necessarily removes many difficult shots
from the teammate sample. That does not answer whether his presence causes the
team's *overall* shot diet to become harder or easier.

The parent model therefore uses one row per team FGA and asks:

> Controlling for all ten players, does the lineup take harder or easier broad
> playtypes when this player is on offense?

Positive offensive values mean the overall lineup shot mix becomes harder.
This is an adjusted association, not proof that the player created the shot or
made it difficult “for no reason.”

## Event target

Every FGA is assigned to one of seven outcome-invariant buckets:

1. transition
2. cut two
3. self-created three
4. other three
5. self-created two
6. other rim two
7. other two

The transition label takes precedence over the six description families. Each
feature season uses the immediately preceding regular season's actual points
per FGA for pricing, shrunk by 100 attempts toward that season's league mean:

```text
difficulty points = 50 × (prior league points/FGA − prior bucket points/FGA)
```

No current-shot make/miss result enters the feature-season target. Historical
transition assignment has an important limitation: pre-2025 shots are scored
retrospectively by the classifier frozen from the 2025 labeled era. The labels
are outcome-invariant, but this is not pristine historical as-of feature
extraction.

## Parent and role split

The solve uses a 700-day half-life, offense/defense ridge penalties of
`3000/5000`, and a shooter-interaction penalty of `10000`. For every offensive
player on an attempt, the role term is:

```text
is_shooter − player-season TD-weighted shot share
```

This centering eliminates the 35%-usage mechanical distortion. It also gives
the exact player-level identity:

```text
overall parent
  = shot share × conditional own difficulty
  + (1 − shot share) × conditional teammate difficulty
```

Across the 2023-2026 annual solves, the largest weighted interaction-centering
gap is below `3.1e-13`, and the largest parent-decomposition gap is below
`1.2e-16`.

## Future-oTS test

Every model uses the same 735 rows and the same feature-season creation
controls: on-ball percentage, potential assists per 100, and rim assists per
100. The scoring baseline is the floating specification with raw points per
100, TSA per 100, and season separately represented. The inspected target
seasons are 2025 and 2026 and are now spent development evidence.

| Model | RMSE |
| --- | ---: |
| Own PBP role + overall mix | 1.4703 |
| Own PBP role | 1.4744 |
| Synergy role | 1.4860 |
| Synergy role + overall mix | 1.4887 |
| Floating box baseline | 1.4940 |
| Overall mix only | 1.4966 |

The important paired comparisons are:

| Challenger versus baseline | RMSE change | 95% interval |
| --- | ---: | ---: |
| Own PBP role versus box | -0.0197 | [-0.0383, -0.0024] |
| Overall mix versus box | +0.0026 | [-0.0066, +0.0115] |
| Overall mix after own PBP role | -0.0040 | [-0.0233, +0.0136] |
| Own PBP role after overall mix | -0.0263 | [-0.0515, -0.0030] |
| Own + mix versus Synergy | -0.0157 | [-0.0328, +0.0013] |

The overall-mix coefficient is directionally sensible and stable: after own
role difficulty, one point of harder overall mix carries physical coefficients
of `-1.50` for the 2025 target and `-1.23` for 2026. The combined model also
beats own role by point estimate in both folds. The pooled uncertainty remains
too wide to claim incremental predictive value.

## Interpretation

The exercise separates two ideas that should not be merged:

- **Taking difficult shots:** the own-shot PBP role feature, which retains a
  clear incremental future-oTS result on this development window.
- **Changing the lineup's overall shot diet:** the TD parent, which correctly
  removes the usage arithmetic but adds no clear future-oTS signal.

The parent can describe whether tougher playtypes occur around a player. It
cannot, by itself, distinguish productive creation from bailout possessions,
late-clock grenades, coaching choices, or teammate limitations. That causal
language would require event-sequence creation attribution, not a mix RAPM.

## Reproduction

Run:

```bash
.venv/bin/python nba_pipeline/scripts/analyze_td_overall_playtype_mix_ots.py
```

Use `--rebuild-attempts` only when the raw PBP or frozen transition archive has
changed. Generated attempt panels, boards, predictions, coefficient audits,
and manifests live under `nba_pipeline/results/td_overall_playtype_mix/` and
remain outside git.


## Annual reader board, 1997-2026

`scripts/build_td_playtype_mix_board.py` turns the research solver into an
annual reader board for the scoring-impact page. The reader spec is set by the
owner (2026-09-26) and differs from the TD700 research run above:

- **Window and decay.** Each feature season fits a two-season window, the
  feature season plus the prior season, with a 550-day half-life. The prior
  season carries about `0.63` weight. The event target, lag pricing, alphas
  (offense/defense/role `3000/5000/10000`), and exact own/teammate identity
  are unchanged.
- **Every season.** Feature seasons run 1997-2026. Each season's playtypes are
  priced at the previous season's league value. 1997 has no prior season, so
  it is priced on itself and fit on one season (`history_seasons = 1`); every
  later season has 2.
- **Descriptor era.** Before 2009, play-by-play rarely tags self-created
  threes, so early diets mostly separate rim, other twos, threes, and
  transition.
- **Rows.** Only player-seasons with on-court FGA in the feature season are
  written: 14,565 rows, 428-605 per season. The window solve also returns
  2,770 inactive players, which are dropped. Names come from the observed and
  scoring ledgers, then the curated exact-`nba_id` `autocomplete_map.csv`.
  `ID_<n>` placeholders stay blank, and raw-PBP surnames are never used. Nine
  rows are left without a name.
- **Value.** `*_value100 = -contribution / 50 * league FGA per 100` prices
  each part in expected points per 100 offensive possessions, with positive
  meaning an easier diet. League FGA per 100 is measured per season (88-92)
  on the complete-lineup sample. Every identity closes within `7e-16`.

The scoring board carries only the full parent as `playtype_mix_rapm`
(`= mix_value100`). The own/teammate split stays here for audit.

```bash
cd nba_pipeline/scripts
../../.venv/bin/python build_td_playtype_mix_board.py        # ~20 s from the cached panel; --rebuild-attempts after raw PBP changes (~7 min)
../../.venv/bin/python audit_playtype_mix_stability.py       # disjoint-window stability receipt
```

Outputs, all gitignored, go to `nba_pipeline/results/td_playtype_mix_board/`:
the board CSV, the attempt panel, lag category prices, lineup coverage, league
FGA rates, `disjoint_window_stability.{csv,json}`, and `manifest.json`. The
`*_2015_2026` files there are superseded TD700 outputs.

The stability receipt asks whether the diet belongs to the player or to the
roster. It solves disjoint two-season windows three years apart at the same
spec (players above 1,500 weighted FGA of exposure in both). `mix_rapm`
correlates `0.62`, `0.47`, `0.48`, and `0.59` for 2003-04/2006-07,
2010-11/2013-14, 2017-18/2020-21, and 2022-23/2025-26 (187-236 players). It
holds for players whose frequent teammates almost entirely changed (`0.62`,
`0.45`, `0.48`, `0.53`).

This spec has not been through the future-oTS gate, and nothing here changes
the decision above. The board is descriptive and is not a zTS, VPM, or prior
input. See the
[historical scoring board](./HISTORICAL_SCORING_SECURITY.md#playtype-mix-rapm-research-column)
for how the column is displayed.
