# NBA Playoff Droppers: Translation, Archetypes, and Exposure

## Status

Research-only baseline. No RAPM, VPM, prior, public leaderboard, or playoff
forecast is changed. All postseason data through 2026 have now been inspected;
they are development evidence for future versions, not an untouched holdout.

The canonical generated receipts are:

- `results/playoff_droppers/` for target-season regular-season profiles;
- `results/playoff_droppers_lagged/` for the five-season profile ending before
  the target season.

The scripts are `scripts/build_playoff_rs_factor_profiles.py` and
`scripts/analyze_playoff_droppers.py`.

## What the model is trying to distinguish

“Playoff dropper” is not one estimand. A useful system has to keep at least four
questions separate:

1. **Environment change:** how offense and defense change for the selected
   players and teams that reach the playoffs.
2. **Rotation survival:** which players lose or gain lineup share, including
   rostered players who fall to zero minutes.
3. **Persistent player translation:** whether a player's earlier postseason
   residual improves prediction of later postseasons after an entering rating.
4. **Matchup exposure:** whether an opponent can force a particular player or
   lineup into a weak action often enough to matter in that series.

A regular-season/playoff box-score split answers only the first two imperfectly.
A postseason-only RAPM can describe the third retrospectively but does not prove
that the label transports. A real predictive playoff adjustment must improve
future possessions or games after all selection and timing rules are frozen.

## Baseline design

### Player-history translation

- Base rating: pre-playoff EPM already frozen for the target postseason.
- Outcome: actual playoff possession points, centered within postseason.
- Earlier-playoff update: separate offense and defense ridge at alpha
  `24000/24000`, using at most the prior eight postseasons.
- Development: 2007-2018 playoffs.
- Historical stress: 2019-2026 playoffs, plus an excluding-2020 sensitivity.
- Evaluation: equal postseason weight and game-clustered bootstrap uncertainty.

The history-only stress deltas, in MSE times one million, were:

| Candidate | Delta | 95% interval |
| --- | ---: | ---: |
| offense history | +11.82 | [-47.04, +68.75] |
| defense history | +9.43 | [-48.37, +62.41] |
| offense + defense history | +19.85 | [-64.46, +93.06] |

Negative favors the candidate. Earlier postseason residuals did not improve
later-postseason prediction. The generated 2026 tails are therefore inspection
lists, not established “82-game” or “16-game” player labels.

### Entering archetypes

The primary panel uses only target-season regular-season information:

- positive-good offensive and defensive TS, TOV, and rebound factors;
- offensive three-frequency;
- creation load from final creation volume, on-ball time, and potential assists;
- assisted-shot dependence from assisted two/three shares and low on-ball time;
- transition dependence from fast-break scoring share.

The defensive hypothesis was registered as:

```text
rebound_only = max(-z(dTS), 0) * max(z(dREB), 0)
exploitation = defensive lineup rebound_only * offensive lineup positive 3P frequency
```

All ordinary factor main effects remain in the model. This matters because high
dREB is normally valuable; the hypothesis is that rebounding can mask a separate
shot-defense weakness, not that rebounding itself becomes bad.

Each addition is nested on top of the same parent prediction. Ridge strength is
selected only on development postseasons by leave-one-postseason-out MSE and a
one-standard-error most-shrunk rule. The grid includes an exact null update.

The current-regular-season candidates had directional CV minima, but their
uncertainty admitted the null:

| Nested candidate | CV-min alpha | CV delta ×1e6 | season SE ×1e6 | selected |
| --- | ---: | ---: | ---: | --- |
| seven factor main effects | 30,000 | -34.64 | 36.16 | null |
| rebound-only beyond factors | null | 0.00 | 0.00 | null |
| spacing exploitation beyond rebound-only | 3,000 | -1.10 | 4.52 | null |
| offensive archetypes beyond factors | 3,000 | -54.11 | 60.89 | null |
| joint offensive + defensive archetypes | 300 | -8.72 | 14.61 | null |

The five-season lagged sensitivity also selected the null for every update. Its
strict ten-player coverage was only 70.4%, versus 99.3% for current-RS factors.
No blend was selected after opening the stress years.

## What actually changes in the playoffs

For common 2019-2026 player-seasons, weighted by common phase exposure, offense
fell `4.46` points per 100 possessions. Almost all of that was first-chance
offense (`-4.45`); second-chance value was essentially flat (`-0.01`). The
largest measured changes were:

- transition value: `-0.66` per 100;
- rim frequency value: `-0.89`;
- rim conversion value: `-0.91`;
- midrange conversion value: `-1.12`;
- three-frequency value: `-0.64`;
- three-point conversion value: `-1.02`;
- turnovers: `+0.23`, while bad-pass turnovers fell `0.41`;
- free-throw value: `+0.11`.

These are descriptive phase changes among selected playoff participants, not a
causal decomposition of the playoff penalty. They nevertheless support a more
specific picture than “more isolation”: easy transition offense contracts, rim
and three shotmaking both worsen, and the decline is not primarily a generic
turnover explosion or a loss of second chances.

The common-observed-player rotation diagnostic also did not isolate the proposed
rebound-only archetype. Its mean possession-per-game change was `-5.09`, versus
`-4.89` for other players. This comparison omits rostered players who receive
zero postseason minutes and is not yet a selection model.

## Answer to the low-dTS / high-dREB hypothesis

With the repo's positive-good sign convention, lower dTS means **more** shot-
defense vulnerability, not less. High dREB can offset some average possession
cost by ending misses, but it does not make the player less targetable. The
relevant playoff mechanism would be opponent-specific: can the offense move the
player away from the rim, force switches or closeouts, and convert the resulting
shot-quality advantage?

The universal low-dTS/high-dREB product did not clear the development gate, with
or without offensive spacing. That rejects this crude archetype adjustment; it
does not prove that individual slow-footed rebounders are never exploitable in a
particular series.

## Architecture for the next, better model

The next version should be an **exposure-conditioned translation model**, not a
larger list of player stereotypes:

1. Build transaction-aware playoff rosters and model rotation share with zero
   minutes included. Selection into playoff minutes is part of the outcome.
2. Define each series before looking at its results. Attach opponent personnel,
   likely matchup assignments, screen involvement, switch frequency, spacing,
   rim pressure, and lineup size available before each game.
3. Estimate expected weak-skill exposure from regular-season actions, then
   separate exposure from suppression. A player cannot be credited with
   “surviving” an action the opponent never forced.
4. Add team, coach, series, and health random effects around a shrunk player
   translation. This prevents a stable scheme or injury from becoming permanent
   player identity.
5. Model offense through role changes: transition opportunity, assisted-shot
   dependence, self-created rim/three volume, late-clock burden, passing release
   valves, and who absorbs the displaced usage.
6. Fit TS, TOV, and second-chance translations on one shared possession universe
   with exact point-valued closure. Keep environment shifts separate from
   player-centered residuals.
7. Test adaptation within series: entering expectation, games 1-2, and games 3+
   after the opponent reveals its coverage. Do not use later-series information
   to predict earlier games.
8. Report both predictive accuracy and calibration of the tails. Require
   split-postseason and leave-one-series-out stability before displaying a
   player-level “dropper” label.
9. Keep additive player quality separate from pair/lineup synergy and from a
   full game forecast with availability and minutes.
10. Freeze the complete candidate before 2027. Since 2019-2026 has been opened,
    no redesign may call those seasons untouched confirmation.

The core lesson from the baseline is useful: the playoff environment clearly
changes, but generic player-history and broad archetype penalties do not yet
transport. The model should predict **which weakness an opponent can expose in
this series, at what frequency, and with what counterfactual replacement**, not
declare a player universally playoff-proof or playoff-fragile.
