# Alternate Clean 3-Factor Decomposition

This document describes the parallel clean 3-factor build added on top of the existing six-factor pipeline.

## Goal

The standard `TS` / `TOV` split mixes different possession universes:

- `TS` is effectively shot/scoring-attempt scoped
- `TOV` is full-possession scoped

That means the standard offensive split is useful, but it is not a perfectly clean first-chance decomposition.

The alternate build fixes that by keeping all three offensive channels on a shared possession denominator:

- `ALT_TS`
- `ALT_EFG`
- `ALT_SQ`
- `ALT_MAKE`
- `ALT_FT`
- `ALT_FT_FREQ`
- `ALT_FT_SEVERITY`
- `ALT_TOV`
- `ALT_TOV_VALUE`
- `ALT_BADPASS_TOV_VALUE`
- `ALT_SCORING_TOV_VALUE`
- `ALT_FC_COMPLETION`
- `SECOND_CHANCE_CLEAN`

The intended structure is:

- `RAPM_pts = FIRST_CHANCE_pts + SECOND_CHANCE_CLEAN_pts` on the shared possession rows
- `FIRST_CHANCE` is the pre-miss scoring layer
- `ALT_TS` is the shared-denominator first-chance scoring surrogate
- `ALT_EFG` is the shared-denominator non-FT scoring layer
- `ALT_TOV` is the matching first-chance turnover-rate surface
- `ALT_TOV_VALUE` is the point-valued first-chance turnover complement to `ALT_TS`
- `ALT_BADPASS_TOV_VALUE` / `ALT_SCORING_TOV_VALUE` split that point-valued turnover-loss complement by first-chance turnover type
- `ALT_FT` is the shared-denominator FT layer
- `ALT_FT_FREQ` / `ALT_FT_SEVERITY` split that FT layer into trip frequency value and trip severity value
- `ALT_FC_COMPLETION` is the point-valued first-chance complement to `ALT_EFG_VALUE + ALT_FT`
- `ALT_SQ` / `ALT_MAKE` decompose `ALT_EFG` on the same denominator for ShotQuality seasons
- `ALT_EFG_VALUE` removes the fixed replacement-shot baseline from `ALT_EFG`, while keeping the same possession rows, so rim/mid/three shot-location values can be solved without a seventh player baseline bucket
- `ALT_EFG_RIM` / `ALT_EFG_MID` / `ALT_EFG_THREE` decompose `ALT_EFG_VALUE` into first-chance non-FT shot-location value on the same denominator
- `ALT_EFG_*_FREQ` and `ALT_EFG_*_FG` further split each rim/mid/three value into shot-location frequency value and within-zone make value

## New Processed Metrics

### `FIRST_CHANCE`

- built on the standard RAPM possession denominator
- uses RAPM-style scoring, including expected FT value
- stores:
  - `Net_Diff` = first-chance points before the first offensive FGA miss
  - `Is_Turnover` = whether the possession ended with a turnover before the first offensive miss
  - `Is_BadPass_TOV` = whether that first-chance turnover was a bad pass
  - `FC_FT_Diff` on all seasons, using the same foul-type-aware FT baseline logic as the standard TS decomposition, with first-chance EFG average PPS as the replacement-shot constant
  - `FC_FT_FREQ_Diff` / `FC_FT_SEVERITY_Diff` on all seasons, with the row identity `FC_FT_Diff = FC_FT_FREQ_Diff + FC_FT_SEVERITY_Diff`
  - `FC_EFG_Diff` on all seasons, defined as `First_Chance_Diff - FC_FT_Diff`
  - `FC_SQ_Diff` / `FC_MAKE_Diff` on ShotQuality seasons, with the same calibrated EV logic as the standard TS decomposition
  - `FC_EFG_BASELINE_Diff`, `FC_EFG_VALUE_Diff`, `FC_RIM_EFG_AA_Diff`, `FC_MID_EFG_AA_Diff`, and `FC_THREE_EFG_AA_Diff` on all seasons, with row identities `FC_EFG_Diff = FC_EFG_BASELINE_Diff + FC_EFG_VALUE_Diff` and `FC_EFG_VALUE_Diff = FC_RIM_EFG_AA_Diff + FC_MID_EFG_AA_Diff + FC_THREE_EFG_AA_Diff`
  - `FC_RIM_EFG_FREQ_Diff`, `FC_RIM_EFG_FG_Diff`, `FC_MID_EFG_FREQ_Diff`, `FC_MID_EFG_FG_Diff`, `FC_THREE_EFG_FREQ_Diff`, and `FC_THREE_EFG_FG_Diff` on all seasons, with the row identity `FC_EFG_VALUE_Diff = all six frequency/FG components`

### `SECOND_CHANCE_CLEAN`

- built on the same possession rows as `FIRST_CHANCE`
- uses the same RAPM-style scoring
- stores second-chance scoring after the first offensive FGA miss

Legacy `SECOND_CHANCE` remains unchanged.

## `ALT_TS` / `ALT_EFG` / `ALT_SQ` / `ALT_MAKE` / `ALT_FT` / `ALT_TOV`

These are not separate parquets. They are `rapm.py` aliases that load `FIRST_CHANCE` and derive same-sample baselines at solve time.

For a given run:

- `fc_non_tov_avg = average(FIRST_CHANCE Net_Diff on non-turnover rows from the exact loaded sample)`
- `fc_efg_avg = average(FC_EFG_Diff on non-turnover rows from the exact loaded sample)`
- `fc_sq_avg = average(FC_SQ_Diff on non-turnover rows from the exact loaded sample)`
- `fc_make_avg = average(FC_MAKE_Diff on non-turnover rows from the exact loaded sample)`
- `fc_ft_avg = average(FC_FT_Diff on non-turnover rows from the exact loaded sample)`
- `fc_ft_freq_avg = average(FC_FT_FREQ_Diff on non-turnover rows from the exact loaded sample)`
- `fc_ft_severity_avg = average(FC_FT_SEVERITY_Diff on non-turnover rows from the exact loaded sample)`

Then the row construction is:

- `ALT_TS = FIRST_CHANCE Net_Diff` on non-turnover rows, else `fc_non_tov_avg`
- `ALT_EFG = FC_EFG_Diff` on non-turnover rows, else `fc_efg_avg`
- `ALT_SQ = FC_SQ_Diff` on non-turnover rows, else `fc_sq_avg`
- `ALT_MAKE = FC_MAKE_Diff` on non-turnover rows, else `fc_make_avg`
- `ALT_FT = FC_FT_Diff` on non-turnover rows, else `fc_ft_avg`
- `ALT_FT_FREQ = FC_FT_FREQ_Diff` on non-turnover rows, else `fc_ft_freq_avg`
- `ALT_FT_SEVERITY = FC_FT_SEVERITY_Diff` on non-turnover rows, else `fc_ft_severity_avg`
- `ALT_TOV = -Is_Turnover`
- `ALT_BADPASS_TOV = -Is_BadPass_TOV`
- `ALT_SCORING_TOV = -(Is_Turnover - Is_BadPass_TOV)`

This keeps the baseline aligned to the exact run sample instead of hardcoding season-level constants into processed files.

`ALT_TOV` remains a first-chance turnover-rate RAPM. It is useful as a turnover skill surface, but it is not the exact point-valued scoring complement because its turnover-row value is `-1`.

For exact point-value first-chance identities, use the newer aliases:

- `ALT_TOV_VALUE = -fc_non_tov_avg` on true first-chance turnover rows, else `0`
- `ALT_BADPASS_TOV_VALUE` applies that same point adjustment only to bad-pass turnover rows
- `ALT_SCORING_TOV_VALUE` applies that same point adjustment only to non-bad-pass first-chance turnover rows
- `ALT_FC_COMPLETION = FIRST_CHANCE Net_Diff - ALT_EFG_VALUE_target - ALT_FT_target`, where turnover rows use same-sample non-turnover baselines for `ALT_EFG_VALUE` and `ALT_FT`

Those aliases produce the point-value identities:

- `FIRST_CHANCE = ALT_TS + ALT_TOV_VALUE`
- `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE = ALT_TOV_VALUE`
- `FIRST_CHANCE = ALT_EFG_VALUE + ALT_FT + ALT_FC_COMPLETION`

`FIRST_CHANCE_CLEAN` is the direct parquet form of the preferred first-chance target: it uses the same standard possession rows, keeps ExpFT-adjusted first-chance scoring on non-turnovers, and writes `0` on first-chance turnover rows. It should be used when solving the first-chance parent directly instead of reconstructing it from separate `ALT_TS + ALT_TOV_VALUE` solves.

True first-chance turnover rows require no prior first-chance FGA/FT completion event inside the same possession. If a make, FGA, or FT completion already happened, that scoring remains first-chance scoring, but the later terminal turnover is not the `ALT_TOV_VALUE` bucket.

At the solver-alias layer, `ALT_TOV_VALUE` is the clean first-chance turnover-completion parent and equals `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE`. New production EFG-value weighted-factor exports should use that value as the public TOV bucket so BP/scoring drilldowns add to the displayed parent. `ALT_FC_COMPLETION` remains an audit/glue alias for the older `FIRST_CHANCE - ALT_EFG_VALUE - ALT_FT` construction, not the preferred user-facing TOV value.

## `ALT_FT` Frequency / Severity Decomposition

`ALT_FT` can be split without changing the possession universe. The processor identifies each contiguous first-chance FT trip inside a possession and first computes the same row-summed FT value used by `FC_FT_Diff`.

For each FT trip:

- `ft_trip_value = sum(FC_FT row value inside the trip)`
- `avg_ft_trip_value = average(ft_trip_value across first-chance FT trips in the processed file)`
- `FT frequency value = avg_ft_trip_value`, carried once per trip
- `FT severity value = ft_trip_value - avg_ft_trip_value`

Those trip-level values are then aggregated back to the same end-of-possession rows as every other clean first-chance component:

- `FC_FT_FREQ_Diff + FC_FT_SEVERITY_Diff = FC_FT_Diff`

Interpretation:

- `ALT_FT_FREQ` is the value of generating more first-chance FT trips, priced at the league/sample average FT-trip value
- `ALT_FT_SEVERITY` is the value above or below that average trip, combining trip type mix and FT shooting
- the split is intentionally two-part; the trip-type mix and FT shooting pieces both live inside severity unless a later drilldown needs to separate them

## `ALT_EFG_VALUE` Shot-Location Decomposition

`ALT_EFG` remains the clean accounting parent defined as `ALT_TS - ALT_FT`. `ALT_EFG_VALUE` is the shot-location value parent used for the six-bucket rim/mid/three display. It keeps every `FIRST_CHANCE` possession row in the solve, but subtracts the fixed replacement-shot baseline before solving player coefficients.

For first-chance FGA rows, the processor computes:

- `fc_efg_avg_pts` = average actual first-chance FGA points per shot in the processed file
- `rim_avg_pts`, `mid_avg_pts`, `three_avg_pts` = average actual first-chance FGA points per shot inside each zone

The accounting layer is:

- `FC_EFG_BASELINE = fc_efg_avg_pts` on FGA rows
- `FC_EFG_BASELINE = fc_efg_avg_pts / 2` on normal 2-shot foul FT rows
- `FC_EFG_BASELINE = fc_efg_avg_pts / 3` on normal 3-shot foul FT rows
- `FC_EFG_BASELINE = 0` on and-1 and technical FT rows
- `FC_EFG_VALUE = FC_EFG - FC_EFG_BASELINE`

The first shot-location layer is:

- selected zone above-average value = `actual_points - fc_efg_avg_pts`
- unselected zones = `0`

The second layer splits each selected zone into:

- `zone_freq = zone_avg_pts - fc_efg_avg_pts`
- `zone_fg = actual_points - zone_avg_pts`

For FT baseline rows, all six shot-location components are `0`; the fixed FT replacement-shot value is carried only by `FC_EFG_BASELINE`, not by a player-facing shot-location bucket. For turnover rows, processed row values remain `0`; the `rapm.py` aliases replace them with the exact loaded-sample non-turnover average for each component, matching the rest of the alternate first-chance alias system.

The resulting aliases are:

- accounting parent: `ALT_EFG`
- six-bucket parent: `ALT_EFG_VALUE`
- first layer: `ALT_EFG_RIM`, `ALT_EFG_MID`, `ALT_EFG_THREE`
- second layer: `ALT_EFG_RIM_FREQ`, `ALT_EFG_RIM_FG`, `ALT_EFG_MID_FREQ`, `ALT_EFG_MID_FG`, `ALT_EFG_THREE_FREQ`, `ALT_EFG_THREE_FG`

Interpretation:

- rim frequency value is positive when a player shifts first-chance shots toward rim attempts, relative to average first-chance EFG shot value
- mid frequency value is negative when a player shifts first-chance shots toward midrange attempts; reducing midrange frequency creates positive value by avoiding below-average shot value
- three frequency value is whatever the current loaded-sample three-point PPS says relative to the average first-chance EFG shot value
- each zone FG value is the within-zone make component after crediting/debiting the zone's average shot value

Important caveat:

- `ALT_EFG` and `ALT_FT` are available on all seasons because the FT split does not require `initial_ev`
- `ALT_FT_FREQ` and `ALT_FT_SEVERITY` are also available on all seasons because they split the existing first-chance FT row values
- `ALT_SQ` / `ALT_MAKE` still depend on `initial_ev`, so they are only available on ShotQuality seasons (`24-26` in the current repo)
- `ALT_TS` remains the public first-chance scoring surrogate on all first-chance possessions; `ALT_EFG` / `ALT_FT` are shared-denominator components of that surrogate, not literal classic `eFG%` / `FT%` stats on shot-attempt denominators
- `ALT_TOV` is now a true first-chance turnover metric, not a scoring complement
- for direct first-chance parent solves, use `FIRST_CHANCE_CLEAN`; for additive display drilldowns, use `ALT_TOV_VALUE` with `ALT_BADPASS_TOV_VALUE + ALT_SCORING_TOV_VALUE = ALT_TOV_VALUE`
- all first-chance aliases and parents use the same solve-time offense/defense recentering operator, so additive identities survive player coefficient centering up to CSV rounding
- always verify the processed parquet row algebra before interpreting coefficient sums: row-level identities should have zero max gap and zero sum gap, while separately solved RAPM coefficients can show tiny ridge/rounding gaps
- production display exports should use the six atomic EFG components and derive rim/mid/three/shot totals from them; the broader `ALT_EFG_RIM`, `ALT_EFG_MID`, and `ALT_EFG_THREE` solves are audit-only

## Output Surface

The standard `weighted_factors_*.csv` output is unchanged.

The active public Alt3 build writes:

- `weighted_factors_alt3_efg_value_*`

This is the current interpretation path. It uses direct `SECOND_CHANCE_CLEAN`
for `oSC` / `dSC` and uses `ALT_EFG_BASELINE` as the balancing bucket that
closes the public display to total RAPM.

The player-facing builder writes both CSV and parquet artifacts by default:

- `master_results/weighted_factors_alt3_efg_value_*.csv`
- `master_results/weighted_factors_alt3_efg_value_*.parquet`

The daily pipeline refreshes the active 2026-intersecting rolling set and syncs
both formats to the downstream `rapms` repo.

The older alternate build writes a legacy/audit artifact:

- `weighted_factors_alt3_*`

That older file family rebuilds residual `oSC` / `dSC` buckets from
`off - oFC` and `def - dFC`. Keep it for comparison, player-universe
compatibility, and historical audits, but do not use it as the current public
Alt3 display model.

The team-level companion is built with:

```bash
python nba_pipeline/scripts/build_team_alt3_efg_value_weighted_factors.py 24 26 ALL --alpha 25
```

It uses the same processed possession/component targets as the player EFG-value bundle, but replaces player stint columns with one offensive-team and one defensive-team indicator. Team defense columns are sign-flipped in the public output so positive means good defense. The default output family is:

- `master_results/team_weighted_factors_alt3_efg_value_{years}_{season}_a{alpha}.csv`
- `master_results/team_weighted_factors_alt3_efg_value_{years}_{season}_a{alpha}.parquet`

Columns:

- ShotQuality windows:
  - `oALT_TS`, `oALT_EFG`, `oALT_SQ`, `oALT_MAKE`, `oALT_FT`, `oALT_TOV`, `oALT_TOV_bp`, `oALT_TOV_sc`, `oSC`
  - `dALT_TS`, `dALT_EFG`, `dALT_SQ`, `dALT_MAKE`, `dALT_FT`, `dALT_TOV`, `dALT_TOV_bp`, `dALT_TOV_sc`, `dSC`
- legacy non-ShotQuality windows:
  - `oALT_TS`, `oALT_EFG`, `oALT_FT`, `oALT_TOV`, `oALT_TOV_bp`, `oALT_TOV_sc`, `oSC`
  - `dALT_TS`, `dALT_EFG`, `dALT_FT`, `dALT_TOV`, `dALT_TOV_bp`, `dALT_TOV_sc`, `dSC`
- `oFC`, `dFC`
- `off`, `def`, `net_rapm`, `RESID`
- `possessions`, `off_poss`, `def_poss`

Legacy `weighted_factors_alt3_*` definitions:

- ShotQuality windows:
  - `oALT_EFG = oALT_SQ + oALT_MAKE`
  - `dALT_EFG = dALT_SQ + dALT_MAKE`
  - `oALT_TS = oALT_EFG + oALT_FT`
  - `dALT_TS = dALT_EFG + dALT_FT`
  - `oFC = oALT_TS + oALT_TOV`
  - `dFC = dALT_TS + dALT_TOV`
- legacy non-ShotQuality windows:
  - `oALT_TS = oALT_EFG + oALT_FT`
  - `dALT_TS = dALT_EFG + dALT_FT`
  - `oFC = oALT_TS + oALT_TOV`
  - `dFC = dALT_TS + dALT_TOV`
- `oSC = off - oFC`
- `dSC = def - dFC`

Before publishing the legacy file family, the alt3 factor columns are recentered in place by player possession weights: offensive components use `off_poss`, defensive components use `def_poss`. `off`, `def`, and `net_rapm` remain the raw RAPM totals; `oFC` / `dFC` and residual `oSC` / `dSC` are rebuilt after centering so the legacy decomposition still adds up at the player level. This keeps weighted-mean factor cards centered while preserving `oSC = off - oFC` and `dSC = def - dFC`.

The published rolling `weighted_factors_alt3_*_all_rb_se_a2000_4000.csv` files now use point-valued first-chance turnover-loss drilldowns for the public `ALT_TOV` split. `ALT_BADPASS_TOV_VALUE` and `ALT_SCORING_TOV_VALUE` value their turnover rows against the same first-chance average-points baseline used by the clean alt3 accounting, then the bundle centers the child columns by player possession weights and rebuilds the parent:

- `oALT_TOV = oALT_TOV_bp + oALT_TOV_sc`
- `dALT_TOV = dALT_TOV_bp + dALT_TOV_sc`
- `oFC = oALT_TS + oALT_TOV`
- `dFC = dALT_TS + dALT_TOV`
- `oSC = off - oFC`
- `dSC = def - dFC`

Use `scripts/patch_alt3_tov_value_splits.py` to repair already-published rolling a2000/4000 files from the solved `ALT_BADPASS_TOV_VALUE` / `ALT_SCORING_TOV_VALUE` outputs. The older rate-style `ALT_TOV`, `ALT_BADPASS_TOV`, and `ALT_SCORING_TOV` result files remain useful audit surfaces, but they are not the preferred public BP/scoring TOV split for those rolling cards. `RESID` can show `0.010` in fixed three-decimal public CSVs because `off`, `def`, and `net_rapm` are already rounded; the child-parent and FC/SC accounting identities are exact before display rounding.

## Validation Notes

What is exact:

- the clean possession split itself:
  - `FIRST_CHANCE + SECOND_CHANCE_CLEAN = RAPM scoring layer` on the processed possession rows
- on all seasons:
  - `FC_EFG_Diff + FC_FT_Diff = First_Chance_Diff`
  - `FC_FT_FREQ_Diff + FC_FT_SEVERITY_Diff = FC_FT_Diff`
  - `ALT_EFG + ALT_FT = ALT_TS`
  - `FC_EFG_BASELINE_Diff + FC_EFG_VALUE_Diff = FC_EFG_Diff`
  - `FC_RIM_EFG_AA_Diff + FC_MID_EFG_AA_Diff + FC_THREE_EFG_AA_Diff = FC_EFG_VALUE_Diff`
  - `FC_RIM_EFG_FREQ_Diff + FC_RIM_EFG_FG_Diff + FC_MID_EFG_FREQ_Diff + FC_MID_EFG_FG_Diff + FC_THREE_EFG_FREQ_Diff + FC_THREE_EFG_FG_Diff = FC_EFG_VALUE_Diff`
  - `ALT_EFG_RIM + ALT_EFG_MID + ALT_EFG_THREE = ALT_EFG_VALUE`
  - `ALT_EFG_RIM_FREQ + ALT_EFG_RIM_FG + ALT_EFG_MID_FREQ + ALT_EFG_MID_FG + ALT_EFG_THREE_FREQ + ALT_EFG_THREE_FG = ALT_EFG_VALUE`
- on ShotQuality seasons, the non-turnover first-chance TS split:
  - `FC_SQ_Diff + FC_MAKE_Diff + FC_FT_Diff = First_Chance_Diff`
- on ShotQuality seasons, the richer alias decomposition on the loaded sample:
  - `ALT_SQ + ALT_MAKE = ALT_EFG`
  - `ALT_SQ + ALT_MAKE + ALT_FT = ALT_TS`
  - `ALT_FT_FREQ + ALT_FT_SEVERITY = ALT_FT`
- the point-valued first-chance aliases on the loaded sample:
  - `ALT_TS + ALT_TOV_VALUE = FIRST_CHANCE`
  - `ALT_EFG_VALUE + ALT_FT + ALT_FC_COMPLETION = FIRST_CHANCE`
- the turnover child split on the loaded sample:
  - `ALT_BADPASS_TOV + ALT_SCORING_TOV = ALT_TOV`

What matters operationally:

- the alternate clean 3-factor net/off/def regression still reconstructs RAPM extremely tightly on the player surface
- the active public `weighted_factors_alt3_efg_value_*` artifact displays direct `SECOND_CHANCE_CLEAN` as `oSC` / `dSC`; `ALT_EFG_BASELINE` is the balancing bucket that closes the EFG-value display to `off` / `def`
- the legacy `weighted_factors_alt3_*` artifact uses `oSC` / `dSC` as balancing buckets so that older alt surface still sums exactly to `off` / `def`
- the alt turnover sub-split is chained the same way as the standard TOV sub-split:
  - `ALT_BADPASS_TOV / ALT_SCORING_TOV -> ALT_TOV -> net RAPM`
- on a `26 RS` validation run, the alternate net regression lands at `R² ≈ 0.9992`
- on that same `26 RS` run, the `ALT_TS ~ ALT_SQ + ALT_MAKE + ALT_FT` player-surface check lands at `R² ≈ 0.99997` with coefficients effectively `1`
- on that same `26 RS` run, the `ALT_TOV ~ ALT_BADPASS_TOV + ALT_SCORING_TOV` player-surface check lands at `R² ≈ 0.99982` offense and `≈ 0.99988` defense

## Operator Notes

New processed metric names:

- `FIRST_CHANCE`
- `SECOND_CHANCE_CLEAN`

New analysis prefixes:

- `ALT_TS`
- `ALT_EFG`
- `ALT_EFG_VALUE`
- `ALT_EFG_RIM`
- `ALT_EFG_MID`
- `ALT_EFG_THREE`
- `ALT_EFG_RIM_FREQ`
- `ALT_EFG_RIM_FG`
- `ALT_EFG_MID_FREQ`
- `ALT_EFG_MID_FG`
- `ALT_EFG_THREE_FREQ`
- `ALT_EFG_THREE_FG`
- `ALT_SQ`
- `ALT_MAKE`
- `ALT_FT`
- `ALT_TOV`
- `ALT_TOV_VALUE`
- `ALT_BADPASS_TOV_VALUE`
- `ALT_SCORING_TOV_VALUE`
- `ALT_FC_COMPLETION`
- `ALT_BADPASS_TOV`
- `ALT_SCORING_TOV`

Example commands:

```bash
python nba_pipeline/scripts/reprocess_metric.py FIRST_CHANCE 26 26
python nba_pipeline/scripts/reprocess_metric.py SECOND_CHANCE_CLEAN 26 26
python nba_pipeline/scripts/rapm.py FIRST_CHANCE 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_TS 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_EFG 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_EFG_VALUE 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_EFG_RIM_FREQ 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_EFG_RIM_FG 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_SQ 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_MAKE 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_FT 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_TOV 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_TOV_VALUE 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_BADPASS_TOV_VALUE 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_SCORING_TOV_VALUE 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_FC_COMPLETION 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_BADPASS_TOV 26 26 RS
python nba_pipeline/scripts/rapm.py ALT_SCORING_TOV 26 26 RS
python nba_pipeline/scripts/03_run_rapm_analysis.py 26 26 RS
```
