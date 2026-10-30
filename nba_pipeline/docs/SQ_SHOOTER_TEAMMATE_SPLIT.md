# Shooter and teammate ShotQuality split

The public reader now defaults to the owner-requested five-year [Russell v2 six-part version](V2_SIX_SHOOTING_SPLIT.md). This page documents the preserved initial three-year analysis.

Owner-requested export for Daniel Myers, September 18, 2026. The final scope is
2023–24 through 2025–26, regular season only, with shooter talent retained in SQ.
This is descriptive FGA-only research, not production DECOMP or VPM.

Build with:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python \
  nba_pipeline/scripts/build_sq_shooter_teammate_split.py
```

The generated, gitignored destination is
`nba_pipeline/results/sq_shooter_teammate_2024_2026_rs/`. It contains a portable
`shot_quality_split.html`, pooled and annual CSVs for both sources,
`all_results.csv`, coverage, role CV, and a manifest. The HTML defaults to the
improved hybrid, pooled window, and at least 3,000 on-court team FGA.

## Sources and scope

- Canonical regular-season FGA rows, actual points, shooter and lineups come
  from `results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet`.
- `../vpm/vpm/data/sq_shots_keyed.parquet` supplies provider `initial_ev`.
  This archived source is held fixed for a consistent three-year comparison;
  it is not the refreshed two-season ShotQuality store. Openness coverage is
  reported as explicit closest-defender distance availability.
- `results/sq_walk_forward_challenger/walk_forward_predictions.parquet`
  supplies the saved `context_epm_def_geometry_identity` expected points for
  matching 2025/2026 mids/threes. This is the complete season-forward
  challenger documented in `SQ_WALK_FORWARD_CHALLENGER.md`.
- The improved hybrid uses those predictions where available and provider EV
  otherwise: all 2024 shots, rim shots, and unmatched rows retain provider EV.
  It is **not** a uniform three-year improved-model replay.

Both sources include shooter talent. SQ means expected scoring value, not
shooter-neutral shot difficulty. Make means actual points minus that expectation.
Do not neutralize shooter skill for this owner-selected split.

Regular-season membership comes from canonical RS shots, never the SQ archive's
`playoffs` flag: that flag mislabeled earlier postseason rows. Exact duplicate
source rows collapse; all conflicting keys are removed. Missing EV, mismatched
shooters, invalid EV ranges, and incomplete or repeated-player lineups are
excluded. Improved predictions also require agreement on shooter and make/miss.
There is no EV imputation. FT, turnovers, and continuation are outside scope.

## Regression and identities

For each SQ source, fit SQ and residual targets separately on the same sample:

```
y = season mean + five offensive effects + five defensive effects
    + sum((is_shooter - player_season_shot_share) * player_role)
```

Use zero-prior generalized ridge, offense 3000, defense 5000, and a role penalty
selected from 300 / 3000 / 30000 / 1e12 in three grouped-game folds. Choose the
largest role penalty within one paired fold-difference SE of the minimum.
Validation targets use training-fold season means. Player-season shot shares
are observed full-sample design quantities; this is retrospective role
estimation, not a date-safe future prediction evaluation. Annual fits reuse
pooled selected role penalties. There is no time decay or rubberband control.

At a player's observed sample shot share u:

```
conditional own      = parent + (1-u) * role
conditional teammate = parent - u * role
own contribution     = u * conditional own
teammate contribution = (1-u) * conditional teammate
parent = own contribution + teammate contribution
```

The parent offensive coefficients are exposure-weight centered. Contributions
are points per 100 on-court team FGA; conditional fields are points per 100
corresponding own/teammate FGA opportunities. The four contributions sum exactly
to SQ parent plus residual parent. Because role penalties may differ, this
combined estimate need not equal a separately regularized actual-FGA regression.
Conditional rates do not sum without the usage weights.

This retains the high-usage mechanism discussed by the owner: concentrating
hard attempts on the focal player can increase the teammate SQ signal. Teammate
coefficients are not separately identified passing, screening, or gravity
credit. No external gravity table has been joined.

## September 18 receipt

| Season | Canonical FGA | Retained FGA | Covered games | Geometry FGA | Improved FGA |
|---|---:|---:|---:|---:|---:|
| 2023–24 | 218,702 | 216,860 | 1,230 | 206,252 | 0 |
| 2024–25 | 219,527 | 212,467 | 1,226 | 191,158 | 136,337 |
| 2025–26 | 219,159 | 216,249 | 1,221 | 92,008 | 138,048 |

Total 645,576 retained FGA and 802 pooled players. Provider role penalties:
SQ 300 / residual 300. Improved-hybrid penalties: SQ 300 / residual 3000.
These are role penalties, not borrowed VPM parent/component hyperparameters.

Validation covers conflicting-key exclusion, train-only baseline centering,
equal-penalty linear solve closure, centered shooter interactions, and exact
player contribution closure. The export asserts both SQ and residual closure
below 1e-10 points/100; reader filters, sorting and responsive containment are
checked separately. Small-sample rankings remain uncertain. No predictive
promotion or external publication was performed.

## Public reader and CSV downloads

Owner-requested Sites publication, September 18, 2026:
https://own-teammate-shot-quality.russdt.chatgpt.site

The public reader includes filtered-view CSV download, full pooled CSVs for both
models, and an all-seasons/both-models CSV. All exports retain the four
`*_conditional` fields. A worked Luka example explains usage weighting, with
explicit field-goal-only scope and limits on gravity/passing attribution.

`nba_pipeline/scripts/publish_sq_split_site_assets.py` rebuilds the static reader
and curated downloadable player tables under the generated result directory's
`site/dist/`. That nested Site has its own Git repository and
`.openai/hosting.json`; reuse project `appgprj_6aadb72bafb8819186176c5fe1afc94b`
for updates. Its small curated aggregate CSVs are intentionally committed for
the authorized downloadable release. No shot-level provider records are hosted.
Native deployment succeeded; CSV schema, local links, JavaScript syntax and
filtered download contents were checked before publishing.
