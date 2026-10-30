# Primary-defender net-DECOMP split

Research-only NBA matchup-assignment diagnostic. It asks whether a player's
defensive net shot value differs when the matchup provider assigns that player
as the primary defender versus when one of the other four defenders receives
the assignment.

This is not total defense, causal matchup credit, or an identified help-defense
metric.

## Model

`scripts/run_primary_defender_net_decomp.py` reconstructs the clean-chance
possession key and keeps one common 2024-25 regular-season sample:

- exactly one first-chance FGA in the possession;
- one unambiguous NBA matchup `DEF_ID`;
- exact provider-shooter agreement;
- the assigned defender appears exactly once in the defensive five; and
- the same five defenders are present at the shot and terminal possession row.

Multiple provider assignments remain multi-link events and are excluded from
this first strict arm; the runner never chooses the first ID.

For defender `i`, the role column is

```text
role_i = is_primary_i - player_primary_share_i
```

where primary share is calculated on the exact fitted assignment sample. The
role column sums to zero over every player's defensive shot exposures. Offense,
base defense, and role are fit jointly; role alpha is selected with held-out
games and the one-standard-error rule.

The six preferred net shot atoms are fit with the same matrix and penalty:

```text
rim frequency + rim conversion
+ mid frequency + mid conversion
+ three frequency + three conversion
= net shot parent
```

Reader-facing defense is flipped once from points allowed to positive-good.
For parent `P`, role contrast `C`, and primary share `q`:

```text
primary conditional     = P + (1-q)C
nonprimary conditional  = P - qC
primary contribution    = q * primary conditional
nonprimary contribution = (1-q) * nonprimary conditional
P = primary contribution + nonprimary contribution
```

The runner publishes the algebraic sum of the six child solves as the parent
and retains the separately solved parent as a numerical audit. All published
player and component identities close to machine precision.

## Reader output

Use `*_reader.csv` as the default player file. It is ranked by primary-minus-
nonprimary advantage and, by default, includes players with at least 3,000
assignment exposures across at least two seasons. It uses full provider names,
two decimals for impact values, one decimal for primary-assignment share, and
integer counts. The wider research CSVs are capped at four decimals; the CV
receipt retains extra precision because model-selection differences are small.

The reader columns are overall assigned-shot net defense, conditional primary
defense, conditional nonprimary defense, their difference, primary share,
assignment counts, total exposure, and seasons. Positive impact is good.

## 2021-2025 result

The initial 2024-25 run was a construction smoke test, not a sufficient model
decision. The canonical research run now pools five regular seasons and centers
primary share separately for every player-season.

The strict five-season sample contains 753,830 assignment-resolved first-chance
FGA rows across 991 defenders. Annual eligible coverage is stable between
80.01% and 81.95% of reconstructed first-chance FGA.

Alpha selection holds out one complete season at a time. Because every alpha
is scored on the identical season folds, the one-standard-error calculation
uses paired fold-error differences; using the variation in absolute season MSE
would be dominated by ordinary differences in season scoring variance.

| Role model | Mean held-out MSE |
| --- | ---: |
| Selected/minimum alpha 3,000 | 1.1180701 |
| Near-zero role, alpha 1e12 | 1.1182508 |
| Difference | -0.0001807 (-0.0162%) |

Alpha 3,000 beats the near-zero role arm in all five held-out seasons. The mean
paired gain is `0.0001807` MSE with paired season standard error `0.0000443`.
The paired one-SE rule selects alpha 3,000; the old absolute-error one-SE rule
would incorrectly select role-off because between-season MSE levels vary much
more than the paired model differences.

This supports a real aggregate primary-assignment role signal. It does not by
itself establish reliable player rankings.

### Player reliability

With the selected alpha fixed, the runner refits 2021-2023 and 2024-2025 blocks
separately. Among 314 players with at least 1,000 assignment exposures in both
blocks:

- Pearson correlation of primary-minus-nonprimary contrast: `0.306`;
- Spearman rank correlation: `0.280`;
- sign agreement: `59.2%`.

That is modest repeatability. Some broad archetypes and individual players are
consistent, but many player estimates move substantially or reverse sign.
Treat the five-year player surface as a mechanism diagnostic and candidate
hierarchical feature, not a standalone leaderboard, public DECOMP input, or
VPM prior.

## Interpretation boundary

- `primary` means the provider-assigned defender on the shot.
- `nonprimary` means the focal player was one of the other four defenders. It
  mixes help, rotation, scheme, switching, and ordinary off-ball presence.
- Frequency cells describe assignment-conditioned shot mix on observed FGA.
  Shot-only assignments do not identify deterrence.
- This sample cannot assign turnover, foul, or rebound responsibility.
- The parent is an assigned-FGA net shot parent, not full defensive RAPM.

The next admissible research step is an archetype/position parent with a more
strongly shrunk player residual, followed by a separate ShotQuality provider
sensitivity. Provider identities must never be coalesced. No output should
enter public DECOMP or VPM without stronger player repeatability and an
incremental summed forward gate.

Generated receipts live under the gitignored
`results/primary_defender_net_decomp/` directory: five-year summary, component
detail, season-held-out CV table, split reliability table, manifest, and the
compact default reader CSV.
