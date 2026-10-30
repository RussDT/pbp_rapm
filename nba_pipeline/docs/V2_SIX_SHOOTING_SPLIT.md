# Russell v2: five-year six-part offensive shooting split

Current owner-selected revision (September 18, 2026): restore DECOMP expected-FT
premium, move replacement-shot value back into SQ, then retune all six offensive
penalties. Five regular seasons: 2021–22 through 2025–26.

## Target contract

Frozen `../vpm/vpm/data/russell_sq_v2_ev_21_26.parquet` supplies mid/three EV,
including dated shooter talent. Actual rim scoring stays in SQ/rim; rim residual
is zero. This is not direct openness tracking.

- SQ/rim: mid/three EV + actual rim points + FT replacement-shot baseline.
- Make: actual mid/three points minus v2 EV.
- FT: expected FT points minus that replacement-shot baseline.

The FT rules match `process_rapm_blocks/clean_chance_utils.py`: canonical season
FT% from the read-only `player_stats_with_metrics` lookup; missing percentages
fall back to actual outcomes. Average season raw FGA points is split equally
across a normal two- or three-shot FT trip. And-one, technical and unclassified
FT descriptions carry zero replacement baseline. Cache source rates locally.
No production processor or DECOMP output is changed.

Example: two FTs at 80%, with a 1.1-point shot baseline, contribute 1.1 to SQ
and 0.5 to FT. The previous gross-FT board assigned zero SQ and actual makes to
FT; that mechanically penalized foul drawers' SQ. The new targets close exactly:
`SQ + make + FT = actual FG + expected FT`;
`actual points - expected total = actual FT - expected FT`.

Same sample as before: 1,178,050 single-shooter opportunities, all 6,150 RS games,
1,060,394 FGA, 266,462 FTA, 1,029 pooled players. Raw game/event keys are deduped;
conflicting keys, missing v2 mids/threes, multi-shooter/multi-FGA segments and
invalid lineups are excluded. TS terminal events define opportunities within
periods in chronological clock order. One FGA or standalone FT trip is one
opportunity; FG + and-one stays together. This is not FGA+.44FTA or a possession
denominator. No turnovers or rebound-continuation value enter the model.

## Model and additive display

Each target fits season effects, one shooter, four nonshooters and five defenders
jointly. Six independent own/nonshooter penalties; defense fixed at 5000.
No priors, decay or rubberband controls. This is distinct from production
possession-level DECOMP even though the FT target accounting now matches it.

For observed shooting share u, own contribution is u times own conditional rate;
teammate contribution is (1-u) times teammate conditional rate. Center both role
rates by the same exposure-weighted parent offset. Contributions are points per
100 on-court team shooting opportunities; conditional rates are per 100 own or
teammate opportunities. Six contributions sum exactly to total_shot_impact.
The FT lane includes trip frequency, not just FT%. These are descriptive effects,
not isolated passing, gravity or causal creation credit.

## Retuned penalties

| Component | Own alpha | Teammate alpha |
|---|---:|---:|
| sq | 30 | 1000 |
| make | 10000 | 30000 |
| ft | 3 | 1e+12 |

Full search: 2,299,968 combinations, own grid [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0],
teammate grid [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0, 1000000000000.0]. The expanded range includes 1e12 as an effectively
zero-coefficient option. Earlier capped searches are superseded.
Minimum pairs: {'sq': [30.0, 1000.0], 'make': [10000.0, 30000.0], 'ft': [1.0, 30000.0]}.
Minimum MSE 1.4016153306; selected MSE 1.4016166169.

Three grouped-game folds, equal game weights, selection against summed **actual**
FG+FT points. The largest product of six alphas within one paired fold-difference
SE of the minimum wins; lower MSE breaks equal-strength ties. Annual boards reuse
the pooled selection. These are grid-selected settings, not universal optima.

Crucially, every inner fold rebuilds shooter-season FT% from training-only makes
and attempts and replacement prices from training-only FG points/FGA. These fold
prices use the retained opportunity sample; published board targets use the
canonical complete-season lookup and raw-season FGA baseline. Training-season
means also exclude validation games. Actual held-out FT results enter scoring,
never the training FT rates. Frozen v2 measurement remains a retrospective input;
this is not a pristine as-of end-to-end SQ training replay.

## Chronological stress checks

`validate_v2_six_shooting_split.py` trains through 2022/2023/2024/2025 and tests
the next completed season. Each cut independently selects from 262,144 six-alpha
combinations using only training games (own 1/10/30/100/300/1000/10000/100000;
teammate 300/1000/3000/10000/30000/100000/1000000/1e12). It also tunes plain shooting
RAPM and a centered shooter-role control against actual training points. Defense
is 5000 throughout. New-season reference means come from the last training season.
All comparisons use identical test opportunities. Lineup-only forecasts use
training shooting shares rather than the future observed shooter; known-shooter
comparisons give both role models the same observed shooter. New player columns
have zero learned coefficients. CIs use 2,000 paired game bootstraps stratified by
season, with equal season weight. Negative deltas favor the challenger.

| Comparison | MSE delta | 95% CI | Seasons won |
|---|---:|---|---:|
| six_known_shooter vs centered_known_shooter | -0.00017996 | [-0.00035516, +0.00000924] | 1/4 |
| six_lineup_only vs plain_lineup_rapm | -0.00007298 | [-0.00018992, +0.00004539] | 2/4 |
| six_lineup_only vs centered_lineup_only | -0.00008580 | [-0.00019615, +0.00003055] | 2/4 |
| centered_lineup_only vs plain_lineup_rapm | +0.00001282 | [+0.00000329, +0.00002255] | 1/4 |

Any interval crossing zero is inconclusive. This is retrospective development
stress evidence on spent 2023–2026 seasons. It does not establish untouched
confirmation; 2027 remains untouched. Correct accounting does not by itself
prove a predictive advantage.

## Reproduction and publication

Run `build_v2_six_shooting_split.py --stage all`, then
`validate_v2_six_shooting_split.py`, then `publish_v2_six_site_assets.py`.
Use OPENBLAS_NUM_THREADS=4 and OMP_NUM_THREADS=4. Stages prepare/cv/board are
restartable. Generated artifacts stay out of root git:
- `results/v2_six_shooting_premium_2022_2026_rs/`: rates, opportunities, CV, boards.
- `results/v2_six_shooting_premium_validation/`: nested choices, game errors, CIs.
- Old gross-FT artifacts remain in `results/v2_six_shooting_2022_2026_rs/` and
  `results/v2_six_shooting_validation/` as superseded audit controls.

Eight targeted tests cover trip allocation, training-only target construction,
role design, ridge equivalence, joint loss algebra, season baselines and absence
of future shooter identity from lineup-only forecasts. Artifact checks verify
identical sample/actual outcomes, expected closure and the exact FT conversion
bridge (maximum error 4.5e-16), finite coefficients and player contribution closure.

The existing public Site checkout remains
`results/sq_shooter_teammate_2024_2026_rs/site/`, project
`appgprj_6aadb72bafb8819186176c5fe1afc94b` at
https://own-teammate-shot-quality.russdt.chatgpt.site . Aggregate CSVs retain all
six conditional fields, six contributions, totals and counts. Shot-level data
and fold arrays are not published. Original three-year downloads remain archived.

Site version 4 deployed successfully at the existing URL. Local filters, six conditional CSV fields and 390px mobile containment passed. The teammate-FT raw block is effectively zero; exposure centering leaves a shared conditional offset, not estimated player-specific variation.

## Same-basis oTS comparison

Owner clarified the comparison must share the basis. `build_matched_shooting_ots.py`
fits a single offensive parent on the exact opportunity rows, same summed expected-FT
target, season centering, exposure-weighted display centering and per100 units.
Offense alpha3000, defense5000; no role interaction. Label `oTS — same sample`,
not a copied production oTS. All six-component values are unchanged. All3896 rows
have identical exposure counts. CSV includes normal_ots, normal_ots_opportunities
and normal_ots_source. The full-sample join was superseded before publication.
Pooled matched Jokic9.223 vs split9.777; Shai4.705 vs split5.423. A test verifies
invariance to redistributing SQ/FT targets or changing shooter at fixed lineups.
The historical actual-point chronological control remains separate.

Research overlay: [BURDEN_ADJUSTED_TEAMMATE_SQ.md](./BURDEN_ADJUSTED_TEAMMATE_SQ.md) describes the role/talent baseline plus teammate surplus. They divide teammate SQ; original components and total are unchanged.

Reader comparison: `gain_vs_ots = total_shot_impact - normal_ots`, exported in all result CSVs and the default descending sort. It stays contribution-scaled in both display modes and means model difference, not time improvement. The player-name column is sticky during horizontal scrolling.

The reader also includes [own and teammate turnover value](./OWN_TEAMMATE_TOV.md) in a separate possession-based section. Do not add these turnover-count units to shooting-point totals.

## Clickable gain decomposition

Every player name opens a dialog for the selected window. `build_matched_shooting_ots.py` now also solves its same design/penalty system for each of the three centered target columns. Linear ridge plus common exposure centering ensures normal_ots_sq + normal_ots_make + normal_ots_ft = original normal_ots. Each `gain_{sq,make,ft}_vs_ots` is own+teammate minus that matched normal component; the three changes sum to gain_vs_ots. This is an exact component accounting, not a causal or path-independent attribution of changes to role architecture versus alpha selection. Show both influences explicitly. Modal values always use additive contribution units, regardless of main-table conditional toggle. Turnovers and role/talent residuals never enter this gain. All original3896 result rows remain unchanged; new component/gain columns are in CSVs. Tested linearity, closure, pooled/annual clickable names, closing and mobile containment.

Rim assists/100 is shown next to teammate surplus and exported as rim_assists_per100. Join exact year/nba_id from player_stats_export.csv PASSING_AtRimAssists/100 with OffPoss in player_role_features_2014_2026.parquet. Five-year values weight annual published rates by source offensive possessions; source rates are rounded, so this is a pooled-rate approximation. Require coverage for every player season on the board; missing stays blank. This observed rate is not an impact component and is unchanged by the conditional toggle.
