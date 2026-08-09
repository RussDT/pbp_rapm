# Five-Man Fit Lab Recovery and Historical Refit

## Status

The public deployment at `https://five-man-fit-epm.vercel.app/` preserved the
complete published lineup table even though the original local training project
was lost. On 2026-07-24 the deployment was recovered into:

- `five-man-fit-lab/data/lineups.json`
- `five-man-fit-lab/recovery/original-deployment.html`
- the rebuilt application under `five-man-fit-lab/`

The recovery contains 459 exact five-man offensive lineups with at least 200
possessions, all construction inputs, all feature contributions, the
EPM-relative baseline, the shrunk exact-lineup residual, and the early/late
reliability labels.

This remains an output and product recovery: the original model-training code
was not embedded in the deployed page. A separate 2014-2026 historical model
has now been refit from source data. It is a new model, not a claim that the
lost coefficients or original sample construction were recovered.

The rebuilt private production app is published at
`https://five-man-fit-lab.russdt.chatgpt.site`. The original Vercel deployment
remains untouched.

## Published Accounting Contract

The recovered model uses:

```text
total lineup fit = mechanism fit + shrunk lineup-specific fit
modeled offense vs average = EPM baseline + total lineup fit
```

The mechanism layer closes through four domains:

```text
mechanism fit = usage + on-ball + playmaking + pressure
```

Each domain also closes to its published feature contributions:

- usage: total usage, overload, shortfall, top-two usage, usage concentration,
  and primary-creator usage;
- on-ball: top-two on-ball share and on-ball overlap;
- playmaking: total potential assists, top-two passing, passing overlap, and
  passer-finisher fit;
- pressure: rim/FT pressure, turnover burden, and usage-weighted TS%.

Automated tests assert all three layers of row algebra to `1e-9` for every
recovered lineup.

## Reliability Contract

The deployed report used 1,000 possessions of shrinkage for the exact-lineup
residual, selected by forward-only validation. Its published summary is:

- 459 qualifying lineups;
- 88.6% EPM feature coverage;
- 0.032% mean forward MSE gain;
- positive gain in 3 of 3 forward folds;
- 48 temporally repeatable lineups.

The 48 repeatable groups are 17 positive, 30 negative, and one small. Another
29 are unstable and 382 do not have a sufficient temporal split.

Early/late values are noisy, unshrunk diagnostic residuals. They are not the
published lineup-fit estimate. “Repeatable” means early and late samples agree
in direction after the same controls; it is not a causal claim.

## Rebuilt Product

The new app keeps the model contract visible and adds:

- a 2014-2026 exact-five history as the default product view, with the recovered
  EPM model clearly separated as a five-year archive;
- a new EPM-relative historical offensive construction model built from
  normalized predicted usage and a monotone creation-capacity shortfall;
- a `Build a Five` calculator for any five 2025-26 players, with raw EPM usage
  demand, normalized allocation, neutral offense/defense/net, model fit versus
  EPM, and the full raw-feature-to-contribution ledger;
- a scroll-contained historical leaderboard with click-to-sort headers, a
  sticky header row, and a frozen lineup column for wide-table navigation;
- a mechanism-vs-identity scatter map with possession and reliability encoding;
- comma-separated multi-player search;
- a player lens showing qualifying lineup context and common teammates without
  presenting the result as a player rating;
- comparison of up to three lineups on the same model layers;
- the recovered five-year model's raw EPM role-demand sum, explicitly labeled
  as a diagnostic that can exceed 100% and is not lineup usage;
- the latest exact-five 2014+ defensive match directly in recovered lineup
  details and comparisons, including its PBP mechanism defense, when one exists;
- a complete 15-feature contribution ledger;
- construction-input percentiles against all 459 lineups;
- explicit separation of EPM baseline, modeled mechanism, shrunk identity, raw
  residual, and early/late diagnostics;
- deep links, responsive layouts, keyboard navigation, and accessible controls.

The original deployment remains untouched.

## 2014-2026 Historical EPM Refit

The rebuilt app carries a 2014-2026 regular-season model and history lane.
The 2014 boundary is deliberate: it is the feature-complete modern lineup
window covered by the validated raw-lineup repair and complete on-ball/passing
inputs.

`scripts/build_historical_five_man_epm_fit.py` builds the model from:

- clean exact-five scoreboard possessions from
  `LINEUP_NET_RATING_POSSESSIONS*.parquet`;
- season-end Dunks & Threes EPM/E-Skills snapshots for each season;
- historical `PASSING_on-ball-time%` and
  `PASSING_Potential Assists/100`;
- forward-only ridge validation with a 2018-2023 development window and a
  later-season 2024-2026 holdout-style readout. That later window was already
  opened by the first refit and is no longer a pristine confirmation sample.

Each player's Dunks & Threes `p_usg` is divided by the five-player sum for every
displayed allocation and concentration feature:

```text
normalized player usage_i = p_usg_i / sum(p_usg_1 ... p_usg_5)
sum(normalized player usage_i) = 100%
```

The raw five-player `p_usg` sum is retained as role demand. It may exceed 100%
and must never be labeled lineup usage. The signed surplus regressor was
removed because it duplicated the overload and shortfall hinges. Demand below
92% receives a fixed monotone penalty of `-1.5` points per 10 percentage points
of shortfall; this coefficient cannot flip sign during ridge fitting.

The retrospective offensive target is actual season-relative ORtg minus the
five-player oEPM expectation adjusted for the defenses actually faced. The
development ablation now directly selects the published group set. Usage alone
won; on-ball and playmaking remain visible diagnostics with zero scoring
contribution rather than being presented as validated adjustments. Its
possession-weighted MSE improvement over EPM alone is:

- `+1.50%` across the 2018-2023 development folds;
- `+2.24%` in the later 2024-2026 seasons.

The 5,000-possession exact-lineup shrinkage also improved confirmation
early-to-late MSE by `+1.23%`. Final neutral-court display is:

```text
modeled offense = sum(five oEPM) + construction fit + shrunk exact-five evidence
modeled defense = sum(five dEPM)
modeled net = modeled offense + modeled defense
```

The defensive construction challenger is intentionally not published. The
strongest development challenger (defensive activity plus possession finish)
improved development MSE by `+1.95%` but lost `-1.25%` in the 2024-2026
confirmation sample. Usage/on-ball workload combinations did not beat the
D-EPM baseline in development. Public defense therefore stays at the five-player
D-EPM sum, while actual defense and its possession mechanisms remain visible.

## Hypothetical Five Builder

The builder payload is generated by
`scripts/build_historical_five_man_epm_fit.py` at
`five-man-fit-lab/data/builder_players.json`. It contains 2025-26 player inputs
for players with at least 100 offensive possessions and the exact means,
scales, coefficients, and feature groups from the frozen historical offensive
fit.

For any five selected players, the client reconstructs all 16 published
features, normalizes their five `p_usg` predictions to a 100% allocation, and
applies the frozen equation. The on-ball and playmaking inputs remain in the
ledger with zero coefficients so the rejected groups are explicit. It displays
both the raw five-player EPM usage sum and the normalized shares. The estimate
is:

```text
hypothetical offense = sum(five oEPM) + construction fit
hypothetical defense = sum(five dEPM)
hypothetical net = hypothetical offense + hypothetical defense
```

Exact-five identity is fixed at zero because the hypothetical group has no
observed shared sample. The calculator does not simulate positions, coaching,
minutes, injuries, matchups, or future player development. It is a
same-input-scale comparison tool, not a forecast of an actual season.

The 2014-2026 leaderboard mirrors the recovered five-year table approach. Its
lead sortable columns are possessions, total fit versus EPM, actual
opponent-adjusted offensive performance versus the five-player O-EPM
expectation, and raw EPM usage sum. Normalized top-two usage, top-two on-ball,
potential-assist supply, usage fit, and the remaining result/mechanism fields
stay available to the right.
Click a header once for its default order and again to reverse it; the header
row and lineup identity stay pinned while the table scrolls. Opening a row
exposes every raw feature value beside its points-per-100 contribution and
labels rejected groups as diagnostic only.

`scripts/export_historical_lineup_fit_lab.py` then combines the model with:

- actual scoreboard points and exact five-man possessions from
  `LINEUP_NET_RATING_POSSESSIONS*.parquet`;
- repo-owned `FIRST_CHANCE` and `SECOND_CHANCE_CLEAN` possession rows for an
  exact additive mechanism ledger;
- `ASSIST_POINTS` and `RIM_ASSIST` possession rows for supporting playmaking
  fingerprints.

The public historical product uses EPM, not DPM or another substitute player
rating. It reports modeled neutral offense/defense/net, actual ORtg/DRtg/net,
and offense/defense versus that season's league environment. Positive
defensive values always mean better defense.

The PBP mechanism ledger is separate from the actual scoreboard. For offense:

```text
shot access
+ shot selection
+ shotmaking
+ free throws
+ second chance
= expected-FT lineup scoring versus the season
```

The defensive ledger uses the same five components with signs oriented so
positive means better defense. Shot selection and shotmaking further expose
rim, midrange, and three-point frequency/conversion atoms; free throws expose
frequency and conversion atoms. The parent components close exactly on every
processed possession and again for every exported lineup. Because the mechanism
lane uses expected FT value, it should not be forced to equal the actual
scoreboard ORtg/DRtg.

The app also exposes turnover, transition, assisted-points, and rim-assist
fingerprints on both ends. The usage-versus-defense view compares normalized
top-two EPM usage share with actual defense above the D-EPM expectation.
Lineup-season weights use defensive possessions capped at 1,000 so a single
high-minute group cannot dominate the display. The all-season correlation is
`0.02`; it is descriptive, not causal, and does not justify a defensive fit
term.

The old DPM-adjusted experiment is retained only as
`scripts/audit_historical_lineup_dpm.py`; its default output is isolated under
`nba_pipeline/validation/` and cannot overwrite the product payload.

## Original Training-Pipeline Recovery Boundary

A genuine historical model now exists, but recovering the original lost model
would still require the exact original:

1. exact offensive and defensive five-man possession rows;
2. date-appropriate oEPM/dEPM inputs and the original missing-data rules;
3. full-season on-ball and potential-assist fields;
4. the feature transformations that produced the published construction inputs;
5. the forward fold definitions, baseline loss, and 1,000-possession shrinkage
   selection;
6. a frozen export contract reproducing the recovered 45-column lineup table.

Do not reverse-engineer new coefficients from the 459 published rows and call
that the original training model. The recovered contribution values are enough
to audit and serve the published artifact, but not to establish its original
sample construction or validation protocol.

## Validation

From `five-man-fit-lab/`:

```bash
npm run lint
npm test
```

The test suite covers the production render, the 459-row recovered EPM archive,
the 1,469-row historical refit, the 2025-26 hypothetical builder, normalized usage closure, offensive fit
accounting, rejected defensive adjustment, absence of DPM in the public
product, and exact accounting for both observed mechanism ledgers.
