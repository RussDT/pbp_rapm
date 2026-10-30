# VPM versus EPM: exploratory player-cohort audit

Owner request October 15, 2026: identify player cohorts where current VPM
performs poorly against EPM. Uses the repaired own-SQ **v40** control,
`vpm_field_benchmark_clean_sq_v1`, not the rejected v43 forward-mean
challenger or original v37 targets. No model, scale, rating or publication
changes. All outcomes through 2026 remain spent development evidence.

## Method

Same 7,230 full games / 180 team-seasons over 2021–2026. Every player retains
the benchmark's exact minutes, rating, native-coverage flag and shared fill.
Compare the original VPM forecast with a hybrid replacing **only a cohort's
VPM rating with its saved EPM rating**, leaving all other players at VPM.
Positive MSE gain means this substitution helps. The control and full EPM
predictions are independently reconstructed from player contributions.

Game margin contributions use signed home-minus-away minutes divided by game
duration, times pace/100. The shared three-point home term stays fixed. Team
quality uses minute-weighted player sums with each system/cohort's seasonal
team mean removed exactly as in the benchmark. No forecast intercept is fit.
Each year receives equal weight. Intervals use 4,000 resamples with frozen
seed20261015: whole seasons for team results and a season-block game
sensitivity, plus paired within-season game clusters for game results.

An additional symmetric squared-error allocation is saved for each partition.
If `delta_c` is the EPM-minus-VPM margin contribution from cohort c and
`delta` is the total model difference, its allocation is
`delta_c * (2 * VPM_error - delta)`. These allocations sum exactly to VPM
squared error minus EPM squared error within every dimension. Hybrid swap
gains themselves **do not add** across cohorts because squared error includes
interactions. Positive allocation attributes a share of the model-error gap,
not causal player credit or a ground-truth individual rating error.

## Cohort contract

Ten dimensions / 52 mutually exclusive cells within dimensions: position,
projected age, prior NBA minutes, returning/new team, detailed team status,
native rating coverage, shot workload, block rate, offensive archetype and
preseason EPM band. Dimensions overlap and must not be combined as independent
evidence. These are exploratory comparisons; all quoted intervals are nominal,
not adjusted simultaneous claims across the 52 cells.

- Exact ID + previous-season `boxscore.csv` supplies position, archetype,
  TSA/100 and blocks/100; require at least250 prior-season minutes for the
  last three. Mixed positions remain explicit unknown/hybrid rather than
  silently assigned. These snapshot publication/classification vintages are
  uncertified and descriptive, not prospectively admitted model features.
- Age is previous-season `age_lookup.parquet` age plus one, with invalid or
  absent prior-year ages left unknown. It is an annual age proxy.
- Prior NBA exposure and veteran flags are the benchmark's saved pre-cut
  ledger. Its pre-2014 supplement is sufficient for the veteran boundary;
  do not claim complete career minutes for older players.
- Team membership uses exact prior-year franchise appearances and realized
  current game-team minutes. An established new-to-team player has at least250
  prior NBA minutes, at least250 minutes in the previous-season box panel,
  and no prior-year appearance for the current franchise. This includes
  in-season moves; it is not a pre-opening-night transaction-certified roster.
- New-to-team NBA entrants have fewer than250 prior NBA minutes. This is an
  experience class, not an inferred draft/undrafted or first-season identity.
- Other new-to-team players have veteran exposure but a missing or under250
  prior-season box record. They include thin-history/returning cases; the
  audit does not infer an injury, retirement or specific absence reason.

## Findings

Baseline team-net RMSE is **3.1861**, versus full EPM **3.0839**. Substitution
results below assess whole-model prediction, not an independent cohort target.
Team MSE gains use whole-season nominal95% intervals.

| Cohort replaced with EPM | Share of minutes | Game MSE gain | Years game substitution helps | Team-net MSE gain [interval] | Hybrid team RMSE |
|---|---:|---:|---:|---:|---:|
| Established player new to team | 20.6% | +0.5671 | 5/6 | +0.3898 [+0.1060,+0.7000] | 3.1243 |
| Other new-to-team / prior-year gap | 1.7% | +0.3611 | 4/6 | +0.1756 [+0.0375,+0.3337] | 3.1584 |
| New-to-team NBA entrants | 10.3% | −0.3982 | 0/6 | −0.1208 [−0.1811,−0.0551] | 3.2050 |
| Prior-year movement shooters | 11.6% | −0.7377 | 0/6 | −0.2714 [−0.5406,−0.0286] | 3.2284 |

Established team changers are the main actionable **diagnostic lead**: the
same players have native VPM and EPM, so their gain is not a missing-rating
fill artifact. EPM rates them about0.319 points/100 lower on average at
actual-minute weights. The game gain's whole-season interval is
**[−0.1415,+1.4308]**, so game superiority is still unresolved; most gain
occurs in2024–2025. Team substitution helps5/6 seasons as well.

All new-to-team players together give game gain **+0.4886** and team gain
**+0.4271 [+0.1404,+0.7251]**, with hybrid team RMSE **3.1184**. This accounts
for a substantial part of the descriptive season-level deficit, but is a
post-hoc hybrid, not a new validated ensemble. The native-only game sensitivity
has the same direction.

Under23 players, players with250–1999 prior NBA minutes, and players with
at least2 prior-year blocks/100 all have EPM-favoring point estimates; their
whole-season intervals cross zero. There is no defensible blanket claim
that young players, centers or high-volume scorers are the failure cohort.
The entrant result in fact favors keeping VPM in that specific slice.

Conversely, swapping to EPM for movement shooters hurts game prediction in
all six years, with game MSE gain **−0.7377 [−1.2521,−0.2980]** using whole
season blocks. Swapping players rated4+ EPM also hurts in5/6 years (gain
**−0.4602 [−0.7936,−0.1398]**). These are exploratory observed strengths,
not proof that every player in either cohort is individually better rated.

## Interpretation and next investigation

The leading question is **portability across team context / interrupted
recent history**, rather than a global rating multiplier or universal rookie
penalty. An old team's lineup correction may transport poorly; prior-skill
states may adapt too slowly; EPM may supply different offense/defense
allocation. This audit does not distinguish those explanations.

The older multi-center defensive-allocation finding concerned the different
EPM12 component-prior architecture. It must not be imported as an established
failure of this own-skill O/D parent. Current position/block slices remain
inconclusive, and no offense-versus-defense error attribution is claimed from
net game margins alone.

Next investigate changing-team players' own prior versus lineup correction,
separate O/D future-possession errors and pre/post-move timing on these spent
years, then preregister any correction and retain2026–27 for confirmation.
Do not select a shrinkage, cohort override or EPM blend from this scorecard.

## Reproduction and verification

From RAPM:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python ../vpm/vpm/scripts/audit_epm_cohorts.py \
  --source ../vpm/vpm/results/harness/vpm_field_benchmark_clean_sq_v1 \
  --output ../vpm/vpm/results/harness/vpm_epm_cohort_audit_20261015 \
  --box ../vpm/boxscore.csv \
  --ages nba_pipeline/results/estimated_skills/age_lookup.parquet
```

Ignored/private outputs: `cohort_summary.csv`, `annual_cohorts.csv` and
`cohort_manifest.json`. Metadata, input, script hashes and weighting policy
are saved. Code asserts exact benchmark source hashes, complete game/team
sample sizes, ambiguous ID/season rejection, previous-season metadata fences,
reconstructed game predictions, exact game/team allocation closure in all ten
partitions, and unchanged input hashes after evaluation. The original boards,
benchmark, penalties and all2027 data remain untouched.
