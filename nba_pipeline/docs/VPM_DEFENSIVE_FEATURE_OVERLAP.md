# Defensive overlap, contested rebounds, and shooting-foul source correction

October19,2026. This diagnoses the existing Parent VPM experiment; it creates no
new architecture, fitted ratings, feature selection or outcome score.

## Overlap measurement

Exposure-weighted correlations on638 complete-case players in the2021–26
matched calibration window, each with at least2000 defensive possessions:

|Pair|Correlation|
|---|---:|
|Blocks and rim contests|0.7350|
|Blocks and rim points saved|0.6816|
|Rim contests and rim points saved|0.4601|
|Total DREB and contested DREB|0.9200|

The five-variable block/DREB/rim-saved/rim-contest/contested-DREB diagnostic has
VIF3.54/7.05/1.88/3.08/10.73 respectively. These are player-feature diagnostics,
not the full lineup information matrix's condition number or evidence of
causal double-counting. Ridge stabilizes fits but does not identify overlapping
features as separate basketball mechanisms. The expanded mapper lowers its
conditional block slope from1.246 to0.607 while assigning a rim-saved slope1.489
and a contest slope−0.180. It redistributes fitted association; it does not
simply add every standalone effect at full strength.

The big-player prior jump documented in `VPM_MODEL_MAP.md` already occurs in
the three-feature core without either rim field. Therefore correlation among
the three rim/block measurements cannot explain the entire prior shift.

## Contested defensive rebounds are available

The dated game-log field is `DREB_CONTEST` in
`results/estimated_skills/tracking_backfill/tracking_backfill.parquet`.
The admitted monthly DFGA donor carries309115 measured contested-DREB
player-games in16663 games across2014–26, with exact PBP exposure checks.
That donor is narrower than the full available tracking archive. Missing
tracking measurements are not zero. The separate next-game OU model is
documented in `NBA_TRACKING_NEXT_GAME.md`; its fitted2022 parameters must not
be back-applied to earlier historical VPM dates.

The Parent defense bank exported `def_contested_dreb100` with its registered
fixed TD550/K1000 smoother. It was tested in core+rebounds, stops/finishing,
all-fields and the second-chance component-prior candidate. The fixed parent
screen has development game RMSE13.778768 for core and13.744256 for
core+contestedDREB. The original rim/other-foul winner reaches13.729755.
This only tests an **addition to total DREB**. No replacement test or validated
contested/uncontested split has been run; omission from the selected recipe
does not establish that the measurement is useless or unavailable.

## Shooting-foul bug and source-only repair

The original builder selected `event_action_type == 2`. In these archived PBP
files, foul action-subtype values are zero even when descriptions explicitly
say `S.FOUL`. Consequently all original `def_shooting_pf100` observations,
snapshots and coefficients were zero. The declared seven-field expanded
model actually had **six active numeric fields**. The prior event-count/limit
audits and numerical tests missed that degenerate extraction; their earlier
pass does not establish this field's validity.

The corrected builder recognizes observed shooting-foul descriptions, rejects
a degenerate complete-file extraction, and writes the separate immutable root
`results/estimated_skills/defense_bank_v51_shooting_pf_repaired/`. The original
source, models and forecasts are unchanged, with the old builder preserved in
`defense_bank_v51/source_snapshot/` and the source defect recorded in
`source_field_issue.json`.

All26 raw RS/PS sources now contain327958 recognized/admitted shooting fouls;
an independent literal-description recount verifies355432 shooting-feature
player-games and all244909 exported packet cells. Other six added source lanes
remain exactly identical. Snapshot dates exclude same-day/future observations.
All26 focused source/numerical/context/evaluation tests pass, including the
zero-subtype shooting-foul regression fixture. The original ratings reader and
review manifest now disclose the inactive field. **No calibration, selection,
ratings or forecast scores have been rerun with corrected shooting fouls.**

The private sibling diagnostic `defensive_feature_overlap.json` saves the full
correlation matrix and tested scope. This correction supersedes claims that
the original expanded board actually used a measured shooting-foul predictor.
