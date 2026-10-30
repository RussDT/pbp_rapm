# VPM Actuals: realized game-credit allocation

October 26, 2026. This replaces the rejected same-game box-to-outcome estimator
in [the original architecture](VPM_ACTUAL_IMPACT_ARCHITECTURE.md). The owner
clarified the objective: **divide credit for the game**, then sum game credits
into season values. Do not fit a new rebound-to-game-outcome slope.

The first release is the completed **2025–26 regular season**, subject to its
explicit source exclusions. A season total means the sum of its included games,
not an invented full-82-game value. Previous historical actual-impact boards
remain withheld. This release does not change selected predictive VPM, its
current pointer, the full-feature research queue or any 2027 outcome boundary.

## Fixed values, explicit responsibility

For each observed possession and each native Net lane, take its actual value
`y`. Subtract the full-source league baseline `b` once per actual possession.
Positive-good defensive value is the negative of the opponent's value. Retained
penalty FTs carry points but zero extra possession exposure.

Divide that fixed budget into:

1. **Identified event credit.** Apply the Net accounting tariffs to an exact
   actor and action. No box-to-impact coefficients are learned.
2. **Shared remainder.** After identified credits, share the remaining lane
   budget equally among the source lineup's five players.
3. **VPM responsibility contrast.** In unidentified lanes only, add the player's
   frozen preseason VPM lane rate divided by 100, minus the mean rate of that
   lineup, once per actual possession. These contrasts sum to zero in every
   lineup and lane. They redistribute credit without changing the outcome.

Formally, where player `i` belongs to that possession's side:

```
credit_i,k = identified_i,k
           + (actual_lane_budget_k - sum_j identified_j,k) / 5
           + allowed_k * exposure * (preseason_VPM_i,k - lineup_mean_k) / 100
```

An identified actor can differ from the terminal five after a substitution
within a possession. The exact actor keeps their event credit; shared credit
uses the raw terminal-action lineup. This limitation is reported rather than
repairing a lineup or inventing exposure.

The VPM contrast uses existing **selected v61 defense-mapper100** preseason
boards at the matching season cut. No current October2026 board is transported
back into October2025 games. Missing preseason players receive an explicitly
flagged neutral responsibility contrast; no career weights or predictive states
are manufactured. This is a declared attribution convention, not a fitted
posterior or uniquely identified causal assignment.

## Where identified credits go

| Evidence | Credit lane | Assignment |
|---|---|---|
| First-chance FGA | Zone conversion | Shooter receives actual points plus the miss-continuation tariff, minus that zone's Net reference |
| First-chance FT | FT | Shooter receives actual FT points minus the applicable replacement-shot reference |
| Personal turnover before a scoring opportunity | TOV | Responsible player loses the non-turnover opportunity tariff |
| Exactly identified steal | Defensive TOV | Stealer receives the deleted opportunity tariff |
| Exactly identified first-chance block | Defensive zone conversion | Blocker receives the corresponding shot-conversion prevention value |
| Offensive rebound in the FGA-miss chain | SC | Rebounder receives the change from miss-continuation value to the retained scoring-chance value |
| Defensive rebound in that chain | Defensive SC | Rebounder receives the continuation value extinguished |
| Later FGA, FT or turnover | SC | Identified actor receives the next change in continuation value |
| Unidentified actor or mechanism | Same measured lane | Remains shared; no fuzzy defender or actor identity is substituted |

An assist does not establish a unique fraction of a made shot's value, and an
ordinary miss does not identify who forced it. Those mechanisms are not
presented as known individual event credits. A rebound never supplies evidence
for who forced the preceding miss.

### Rebound-chain accounting

Use the same physical Net tariff formulas on the new admitted event source:

- `kappa(zone) = downstream SC points / first-chance misses from that zone`;
- `q(zone) = (FG points + kappa * misses) / first-chance FGA`;
- `V = mean(first-chance points + continuation bonus | non-turnover)`.

After the first missed FGA, start the continuation state at `kappa(first zone)`.
Each transition receives reward plus new state minus old state:

```
offensive rebound:  V - old_state
later FGA:         actual_FG_points + new_miss_kappa - old_state
later FT:          actual_FT_points - old_state
turnover/completion/end: 0 - old_state
```

At a defensive rebound the offensive completion loss is shared; the defender
who completes the possession gets the positive extinguished state. Shot-defense
responsibility for later scoring remains shared. The transitions telescope to
`actual_SC_points - first_miss_kappa`. This prevents a rebound from acquiring
the preceding shot-conversion credit.

These are accounting opportunity tariffs, not coefficients fitted by
correlating a rebound count with game results. They are retrospective full-source
prices, not pregame forecasts. FT misses follow the existing Net convention:
they do not begin the FGA second-chance segment. A subsequent turnover does not
charge a second first-chance opportunity tariff after an FT opportunity.

### VPM responsibility masks

- Offense: shot-opportunity frequency and its two midrange-reference contrasts;
  unidentified first-chance turnovers.
- Defense: shot opportunities, unresolved shot conversion, FT defense, and
  unidentified first-chance turnovers. An entirely identified block conversion
  receives no additional contrast.
- Neither side receives a VPM contrast in SC. Identified offensive conversion
  and FT value receive none either.

The original learned defensive-rebound price is absent. Every export and reader
shows identified/shared/VPM values. An equal-shared control removes only the
last term and preserves the same measured team outcomes.

## Event source and conservation

Action-level checks exposed a distinction between the frozen historical lane
cache and the source contract needed here. For example, game0022500001's raw
event703 is Thompson's missed three; the cached terminal707 reports a midrange
first miss. Whole-game scoring closure alone does not certify those actor/zone
labels. Preserve that archive and its earlier numerical receipts.

This product uses a **separate event-owned possession adapter**. Exact offensive
actors, defensive rebounds, turnovers, final FTs and period transitions define
the source. Same-clock and-ones stay with their basket; retained penalty FTs are
zero-exposure administrative rows. Every actual scoring action must have an
exact game/player/team identity, raw lineups must contain ten distinct members
of the exact game rosters, and raw personal points must match the independent
observed-count source. No actor is filled from a position or approximate name.

The starting game census retains the existing source-admitted2026 games.
Any additional actor/source failures exclude whole games and remain listed.
The new source and prices are written only to the new actual-credit root.
Predictive targets, archived fits and native historical exports are untouched.

The eight displayed lanes use the ordinary midrange-reference rollup, with the
same reference carried in midrange. They sum to the nine native lanes exactly.

For every team game:

```
sum(player offensive + defensive credit)
  + league_PPP * (own possessions - opponent possessions)
  = actual scoreboard margin
```

Offensive and defensive rates use their own side exposures. Every season lane
value is the literal sum of its game values; its rate is100 times the value
divided by side possessions. No final ranking normalization, player haircut,
position recentering or hidden team residual changes those values.

## Implementation and verification

RAPM owns `scripts/vpm_actual_credit_source_v2.py`, the isolated event adapter.
The sibling VPM harness owns:

- `vpm/harness/actual_credit_v2.py`, registration, source admission and output;
- `actual_credit_math.py`, conserved allocation and aggregation;
- `actual_credit_audit.py`, an independent replay using raw event keys and
  explicit formulas, without importing the producer or allocation kernel;
- `actual_credit_reader.py` / `actual_credit_view.html`, exact game floats,
  season downloads and full possession-credit calculations.

```
python vpm/run_harness.py --config vpm/configs/actual_credit_v2.json \
  actual-credit --stage all
```

The immutable root is `vpm/results/harness/vpm_actual_credit_v2`. The original
same-game estimator roots remain rejected/withheld. A producer receipt is not
a claim that individual causal credit is identified.

The first event-owned `vpm_actual_credit_v1` pass remains withheld separately:
some retained one-shot penalties are labeled only “Free Throw 1 of 1” in the
archive, with the penalty described on the preceding foul. V2 uses basket and
foul context, retains an and-one despite harmless clock drift, and does not
create a new possession or replacement-FGA reference for retained penalties.
The earlier root, original adapter and numerical verifier remain unchanged.

Tests exercise shooter conversion, rebound-identity invariance, fixed budgets,
zero-sum prior responsibility, exact side-weighted game-to-season aggregation,
and-one/retained-FT/dead-ball handling, rebound-chain telescoping and rejection
of personal scoring mismatches. Independent verification checks every retained
raw action key and source field, first-/second-chance scoring and zone count,
identified actor price, persisted allocation, exposure, rate, Net rollup,
game-to-season sum and scoreboard bridge. Browser verification must also match
the possession trace back to the selected player's published game value.

No future-prediction, unique causal identification or individual confidence
interval claim accompanies this accounting release. Home-court, opponent
strength and garbage time are not removed from realized team credit.

## Build status

The initial2026 build is running. Final coverage, audit results, browser receipt
and public deployment will be recorded here after verification.
