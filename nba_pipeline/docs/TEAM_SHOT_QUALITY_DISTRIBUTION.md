# Team shot-quality distributions

Owner-directed descriptive exploration, 2026-09-11. The question is how many
low-value and high-value scoring opportunities a team generates or allows per
100 attempts. The owner defines the home-run threshold as expected points
`>=1.35`. This is not a player attribution model or production metric change.

## FGA control

Use the canonical retrospective `shot_quality_season_ev` in
`results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet`. The 2026 RS sample
has 219,159 unique FGA, 1,230 games and 30 teams with 82 games each. The defending
team is the other of exactly two shooting teams in each game. No keys or EVs
are missing; counts and bins close exactly. Around 0.70 means `[0.65,0.75)`.

The initial FGA-only OKC defensive control allows 11.76 shots at >=1.35 per
100 opponent FGA, versus 13.93 leaguewide. Do not mix this denominator with the
FT-inclusive opportunity distribution.

## FT-inclusive descriptive prototype

Generated reproduction scripts and receipts are under
`results/team_sq_distribution/`: `build.py`, `build_defense.py`, `build_tsa.py`,
`tsa_distribution_2026.csv`, `tsa_data.json`, and `unpaired_one_shot_fts.csv`.
This is a bounded exploration, not a new active portfolio project.

- One FGA or ordinary two-/three-FT trip is one TSA. Count trips from terminal
  FT events; separate trips can occur for the same player at the same clock
  after an offensive rebound, so do not deduplicate on clock/player alone.
- Expected FT value is the shooter's observed RS FT make fraction times trip
  length. This is explicitly retrospective, not a predictive talent estimate.
- Pair and-ones to a unique made FGA by game, period, displayed clock and shooter,
  requiring the FT event number after the made event. Add expected FT value to
  that FGA's SQ without adding a TSA. Late corrected event numbering can leave
  real and-ones unmatched; retain the unpaired receipt rather than guessing.
- Keep non-and-one FGA, matched FGA-plus-and-one, and FT trips separate in the
  chart. An observed and-one is outcome-conditioned; this chart does not claim
  to estimate pre-release foul probability or optimal decision value.
- Exclude 1,633 technical/flagrant/clear-path FTs and 321 unpaired one-shot FTs
  leaguewide. The latter include retained-possession fouls and unresolved
  attachments. This prototype is not an exhaustive all-FT accounting surface.
- Include second-chance FGAs. Turnovers, rebound continuation and the value of
  retaining possession after special fouls remain outside the distribution.

The prototype has 243,676 TSA = 219,159 FGA + 24,517 FT trips, with 5,825 matched
and-ones embedded in FGA. Counts close and summed expected points equal original
FGA SQ plus matched FT bonuses plus trip expected points within 1e-8.

OKC offense has 8,118 TSA = 7,307 FGA + 811 FT trips, including 183 matched
and-ones. Its >=1.35 share is 24.02 per 100 TSA versus league 23.22. Shai's
540/614 FT rate (87.95%) prices two FTs at 1.75896 expected points, equivalent
to the immediate scoring expectation of a 58.63% three. This does not include
rebounding or other continuation value.

The delivered inline chart expands the right tail into 0.10-point bins and
marks both 1.35 and Shai's two-FT expectation. It supports team and offense/
defense selection. Verified actual marks, no horizontal overflow at 736/360px,
light/dark rendering, mobile screenshot and defense selector updates.

## Nash 2004–05 Phoenix on/off

`results/team_sq_distribution/build_nash_2005.py` reproduces the owner-requested
raw offensive on/off distribution using Nash ID 959 and historical `NBA05`
plus the same canonical SQ table. The sample begins with all 82 Phoenix RS
games and 7,018 FGA. Normalize numeric/string game IDs to ten characters before
joining. FGA status uses O1–O5; ordinary FT trip status uses the offensive
lineup at the preceding opposing foul at the same clock, before substitutions
during free throws. Require five distinct known offensive players.

The resolved sample contains 7,919 TSA. Exclusions: 9 incomplete-lineup FGA,
2 incomplete-lineup FT trips, 2 trips with unresolved start/foul, 57 special FT
events and 4 unmatched one-shot FT events. Of 187 matched and-ones, 186 remain
in the lineup-complete sample. Export `nash_2005_opportunities.csv` retains
row-level status; `nash_2005_unresolved_trips.csv` preserves failed attachments.

| Per 100 TSA | Nash on | Nash off, all games | On minus off |
| --- | ---: | ---: | ---: |
| Home runs, EV >=1.35 | 28.40 | 23.54 | +4.86 |
| Low value, EV <0.80 | 17.33 | 25.30 | -7.97 |
| Expected points | 114.72 | 107.88 | +6.84 |

On: 5,239 TSA (4,636 FGA +603 FT trips) in 75 games. Off: 2,680 TSA
(2,373 FGA +307 FT trips) across all 82 games. Restricting off opportunities to
the 75 games Nash played leaves 1,991 TSA and preserves the pattern:
home-run gap +4.39, low-value gap -8.59, expected-points gap +6.49 per100 TSA.
These are descriptive on/off differences, including Nash's own attempts,
not causal player credit or teammate/opponent/score-adjusted estimates.
The historical PBP model has no tracking-level contest information.

### Nash same-opportunity actuals audit

`results/team_sq_distribution/audit_nash_2005_actuals.py` attaches actual FGA
points, actual matched and-one makes, and complete ordered FT-trip results to
the identical 7,919 resolved rows. Every trip must contain exactly the numbered
FT sequence and no FT may be reused. No actuals are replaced with expected FT
values. Receipts: `nash_2005_actual_comparison.csv`,
`nash_2005_opportunities_with_actuals.csv`, and
`nash_2005_calibration_by_band.csv`.

| Points per 100 TSA | Nash on | Nash off | On minus off |
| --- | ---: | ---: | ---: |
| Expected | 114.72 | 107.88 | +6.84 |
| Actual | 117.56 | 105.63 | +11.93 |
| Actual minus expected | +2.84 | -2.25 | +5.08 |

Restricting off rows to Nash's 75 games gives expected +6.49, actual +11.38,
and residual +4.90. The model understates the observed scoring-efficiency
on/off gap by 5.08 points/100 TSA. This is not a possession-level ORtg gap,
and the residual mixes unmeasured context, shotmaking and sampling variation;
it is not identified as luck or causal Nash credit.

### Residual gap by the original 0.10-point buckets

`results/team_sq_distribution/build_nash_bucket_residuals.py` decomposes the
same-opportunity residual gap exactly. For each SQ bucket, subtract off-court
sum(actual-EV)/all off-court TSA from on-court sum(actual-EV)/all on-court TSA,
then multiply by 100. The bucket contributions sum to +5.0844569482 (all games)
and +4.8976955062 (same games) within 1e-10. Do not sum unweighted conditional
per-shot residuals: bucket frequencies differ.

The two largest contributions are 0.90–1.00 (+2.23) and 1.40–1.50 (+1.98),
together +4.21 of the +5.08 gap. In 0.90–1.00, on expected/actual=0.969/1.003
(n=1,042), off=0.966/0.838 (n=327). In 1.40–1.50, on=1.448/1.572 (n=699),
off=1.450/1.417 (n=259). Thus both on-court underprediction and off-court
overprediction matter. The conditional mix inside a fixed bucket is not held
constant. The 1.40–1.50 on-court excess comes from FG/and-one rows rather than
ordinary FT trips: those 304 FT trips average 1.469 expected/1.464 actual.

Receipts: `nash_2005_bucket_residuals.csv` and `nash_bucket_residuals.json`.
The inline figure shows signed contribution bars and an inspectable bucket
with counts, frequency, expected and actual rates, and weighted residuals.
