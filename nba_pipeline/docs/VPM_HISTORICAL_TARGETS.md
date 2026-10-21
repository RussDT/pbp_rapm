# Historical predictive VPM target adapter

The October20,2026 owner request extends predictive Parent VPM to2014 with four
annual cuts and exact Net lanes. This document covers **target measurements and
pricing**, not completion of the dated prior fits or ratings series.

## Source contract

`nba_pipeline/scripts/historical_targets.py` owns the versioned extension of
`export_vpm_net_lane_atoms.py`. It delegates all clean possession construction to
the canonical RAPM implementation. Raw PBP and canonical `RAPMYY[_PS]` files
exist from2000 through2026 for the requested warm-up. The2015 RS and2016–26
RS/PS files reuse the verified v61 price-free measurement archive; other files
are newly generated from raw PBP with full-season shooter FT lookup disabled.
Observed FG and made/missed FT points therefore remain actual outcomes.

Generated files live under `nba_pipeline/results/vpm_historical_targets_v1/`:

- `atoms_YEAR_PHASE.parquet`: canonical game/event/lineup/date, actual points,
  first/second chance, first-zone attempts/points/misses, FT-reference weight.
- `games_YEAR_PHASE.parquet`: unfiltered per-game first misses, continuation
  points, zone FGA/FG points and matched historical dates.
- `historical_YEAR_PHASE.json`: source and output hashes, row/game counts,
  independent raw-event scoring comparisons, mandatory exclusions and closure.
- `manifest.json`: combined measurement receipt after all tasks succeed.

The source measurement files retain quarantined observations for audit.
**Every caller must apply each receipt's `excluded_game_ids` to both atom rows
and per-game pricing counts before fitting.** These include the original four
2014/2015 scoring conflicts plus any independently discovered early conflicts.
The separate historical source gate prices deduplicated raw event1 makes as2/3
and event3 made FT as1; unexplained game-score differences are excluded rather
than manually patched. This is a sample-accounting gate, not proof of complete
provider game or event coverage.

Pre2014 canonical rows may contain zero placeholders because the shared
processor permits at least four identified players per side. Preserve this
source convention; do not fit a player0. Likewise, do not infer a complete2011
season merely from a successfully built parquet. The independent raw-game and
canonical-game census, missing dates and source availability must remain visible
in the final history receipt.

## Dated pricing and exact lanes

The adapter requires identical target/pricing game populations and identical
game dates. It rejects duplicate game counts, duplicated possession identities,
noninteger actual points and any target or pricing date at/after the cut.
Filters for a calibration cohort or rolling deployment apply to both populations.

`vpm_net_lane_targets.targets` remains the sole target algebra implementation.
It prices continuation within the admitted season/phase, pays the same bonus
back from second chance, uses the same turnover sample and weights for each
child, and preserves the parent's actual scoring. It returns nine additive
primitives plus two reference contrasts. `compact(y)` yields the standard
midrange-reference eight lanes:

`rim frequency + rim conversion + mid value + three frequency + three conversion
+ FT + TOV + second chance = actual points`.

Rim/three frequency use their net attempt value **minus the midrange reference**.
The corresponding midrange reference is carried in mid value. This transform
changes interpretation only; it does not alter parent targets or saved totals.

## ShotQuality boundary

The approved clean annual SQ reconstruction has audited scored residuals from
2016 through2026, trained strictly on earlier seasons with before-game player
history. It uses2014 league warm-up and2015 training. The older1997–2015
Russell historical bridge is a retrospectively trained backcast and is not
admitted as historically predictive evidence here.

Consequently, the extension has **no SQ correction before2016**. The raw Net
files contain no manufactured residual columns. Optional `load_before_cut`
merges modern clean residuals only after exact game/event/lineup/date/actual
point checks. It records `sq_active` explicitly. Its zero early correction means
an inactive model term, not an observed zero residual or reconstructed SQ.
Whole-era no-SQ modern validation belongs to the parent historical experiment.

## Interface and reproduction

From RAPM root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python nba_pipeline/scripts/historical_targets.py \
  --years 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 \
  --workers 2
```

Python API after adding `nba_pipeline/scripts` to the import path:

```python
from historical_targets import load_before_cut, target_matrix
atoms, games = load_before_cut(root, cut, years, start=None, sq=True)
y, receipt = target_matrix(atoms, games, cut, weights=td_weights)
# atoms.mid_residual / atoms.three_residual now contain identically repriced SQ.
```

For an entirely actual-point cohort use `sq=False`. To construct only a fully
matured365-day training window pass `start=cohort_cut`, `cut=label_end`, and then
independently prove `label_end <= mapper_emission_cut`. The target adapter cannot
infer the caller's mapper maturity date.

Focused tests cover future pricing, extra past pricing games, mismatched dates,
quarantine/future filtering on both populations, weighted turnover and standard
midrange-reference closure. Generated artifacts remain private and untracked.
No2027 outcomes, public deployment or model selection occurs in this adapter.

## Independent checks during preparation

The fresh2000 playoff prefix before June1 has11,521 possessions across66 games.
Rebuilding canonical Net from a newly truncated raw file reproduces the
standard midrange-reference target within **2.85e-14**, both unweighted and
TD550 weighted. Receipt: `independent_target_audit_2000_PS_2000-06-01.json`;
reproduce with `nba_pipeline/scripts/historical_targets_audit.py`.

The32 canonical2000–2015 RS/PS files have no missing possession dates. Complete
positive disjoint5v5 filtering retains at least one row in every game; it removes
roughly0.1–1.9% of pre2014 rows and none in2014/2015. The full census is
`historical_lineup_coverage.json`. Raw2000 and2011 regular-season counts are each
one below their ordinary season-length reference;2013 has1229 actual games
because Boston/Indiana was cancelled, not a missing-game inference.

The independent accounting regression explicitly distinguishes the missed-shot
token `MISS` from the surname **Missi**. The older general2014/2015 gate used
substring matching; blindly extending it creates false2025/2026 exclusions.
The historical adapter uses word boundaries and tests made/missed Missi FT.
Canonical values and the older frozen audit remain unchanged.

After that correction,34 modern2016–2026 games still conflict with canonical
actual totals. An independent final-score-string check agrees with raw made
FG/FT accounting in33/34 cases and with canonical totals in0/34; the remaining
raw/final disagreement stays quarantined. These new historical exclusions
therefore change the modern history sample relative to selected v61. They do
not overwrite selected ratings or retroactively revise its published evidence.
See generated `modern_score_conflicts_final_score_check.json`.
