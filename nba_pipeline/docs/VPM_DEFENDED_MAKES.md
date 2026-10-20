# Defended makes in the forward Parent VPM prior

Owner requested October19,2026 after inspecting the forward Parent coefficients,
suggesting defended makes beside defended attempts might explain Jokic's high D.
Treat that player as a diagnostic, never an optimization target. This stays in
the existing VPM research slot. BAM work remains separately pending.

## Frozen matched comparison, before new prior fitting

Owner clarified that the requested field is **rim defended makes**, not all
field goals, and asked for the same treatment used for DFGA. The preliminary
all-FG paired-count adapter is superseded before any VPM candidate fit/score;
its ignored source snapshot is preserved in vpm_defended_makes_v1. No broad
FG feature or jumper makes are part of the final comparison.

Control: saved `forward_expanded_mpg_a10p0` in sibling v53. Candidate adds one
numeric defense feature plus missing flag: `own_rim_dfgm100`. All original O/D
feature transforms are retained. Re-estimate the joint mapper with ridge10,
residual2000/4000, matured365-day future-lineup cohorts and outputscale1.
Hold lineup4000/8000/TD550, actualFT, repaired SQ residual12000/48000,
entrant models/blend, game census, minutes/fills, seven tests and EPM fixed.
No parameter grid, player adjustment, role restriction or post-result scaling.

RAPM owns validated rim player-game makes/attempts and exact defensive exposure
in monthly_defense_v1/rim_counts.parquet, 2014–2026. Preserve all existing
recovered cumulative-era gates, consecutive-appearance checks, trade/gap
exclusions and provider-family separation. Missing/unmatched/outage rows
remain absent, never zero. No raw events or defender joins enter VPM.

Fit rim DFGM/100 directly using the same annual past-only two-state OU
count-rate filter as DFGA: same bounds, exposure likelihood, differential-
evolution seed20260928, maxiter60, popsize15, tolerance1e-7, no polish.
Each annual fit trains strictly before its vintage cut (COVID December2020),
excludes first-season/first-appearance losses, verifies compiled against slow
replay, and exports latest strictly pre-cut postgame states with max age730.
Rim DFGA and existing rim savings remain unchanged. Independently estimated
rates are not silently clipped to each other; audit makes/attempt bounds.

Existing rim savings already use measured rim makes and satisfy exactly
saved100 =2*contests100*suppression. Quantify source/feature overlap and whether
the new filtered makes supply useful information or just another correlated
representation. Do not claim an independent new event source or guaranteed
negative coefficient/Jokic correction.

Compare control/candidate on2020–22 development and2023–26 spent stress,
all seven unchanged tests and paired intervals; report EPM without claiming a
win from an unresolved difference. Also score native future actual possessions
and prior-only forecasts, matching the previous four-season diagnostic.
Current September1,2026 output shows priors/final ratings and deltas, including
Jokic and the previous repaired control. No2027 outcomes, automatic promotion,
publication, or changes to existing saved boards/readers. A reader comparison
will remain within the existing local8767 review surface.

## Completed result

The matched addition does not resolve the owner’s Jokic concern. Its current
conditional slope is **−0.207168 D-prior points/100 per extra rim make allowed
per100 defensive possessions**. Correlated coefficients refit jointly: rim
savings1.0712→0.9099, rim contests−0.2480→−0.1361, DFGA0.1132→0.1297,
contestedDREB0.8213→0.8238. Jokic’s new makes forecast is6.02699/100 beside
10.79525 rim attempts/100. His priorD rises2.620965→2.872803; finalD
3.017349→3.171080; total9.874686→10.033024. Wembanyama priorD
5.227531→5.498813, finalD6.380155→6.527903, total9.083126→9.234357.
A negative added-event slope does not force a player decline when all
conditional slopes are re-estimated.

|Comparison|Current forward|+ rim defended makes|
|---|---:|---:|
|2020–22 development game RMSE|13.740987|13.741643|
|Development team-net RMSE|2.821401|2.820043|
|2023–26 spent game RMSE|13.895289|13.893452|
|Spent team-net RMSE|3.091360|3.087822|

The literal fixed-increment development comparison **retains current forward**.
This is separate from the original13-way v53 selection, whose winner remains
the old-label feature extension. No new selected release is created.

All seven stress MSE point estimates favor the addition, but its game-RMSE
change is only−0.001837. GameMSE gain+0.051038[+0.000011,+0.106171]
barely excludes zero under the existing4000-replicate unadjusted bootstrap;
it is not a robust new confirmation across multiple spent tests. Team-net
gain+0.021861[−0.029651,+0.061279] crosses zero, as does low-continuity game
error. Win-percentage and veteran intervals are in the curated seven-test
receipt. EPM remains an external benchmark; no imitation or fresh-season win
claim.

Native423,644 future actual possessions in4,273 games give finalMSE
1.414935547 versus current1.414933460: gain−2.087e−6[−6.669,+2.772]e−6.
Prior-onlyMSE1.415543770 versus1.415538267: gain−5.503e−6
[−14.392,+3.309]e−6. Neither establishes an incremental improvement.
Prior-only intercepts use past TD550 actual means minus past prior lineup
means, with no future score/venue/clock fit. Versus previous repaired,
prior-only gain454.46e−6[296.97,612.08]e−6 persists; final gain79.62e−6
[−11.24,167.03]e−6 remains unresolved.

## Source and numerical checks

Eleven annual differential-evolution fits use49,576→289,868 past measured
player-games and finish no later than the preceding completed season. Current
uses the2025 vintage, fit_end2025-06-22; subsequent2025–26 games update latent
states with frozen parameters. Compiled/slow fit replay gaps are below2e−16.
A separately truncated slow replay reproduces13,092 dated feature cells
(max1.42e−14); all observation/fit dates precede each cut. A fresh rebuild
from repaired archive/PBP sources reproduces314,552 raw rim-count rows exactly.

The2026 pregame measurement diagnostic under frozen2025 parameters has24,577
player-games, exposure-weighted rim-makes/100 RMSE3.0700 versus population
3.2414. This is spent2026 skill evidence and is distinct from summed VPM
increment evidence. The diagnostic export covers2026, not an asserted complete
historical out-of-sample skill archive.

Measured makes never exceed measured attempts. Independently filtered forecast
makes exceed the separately filtered rim-attempt forecast in20 of7,539 paired
states, including two current full-universe rows; those exceptions are
exported as `independent_count_bound_exceptions.csv` without post-result
clipping. Joint coherent count modeling remains a limitation of this requested
DFGA-style independent-count experiment. Current full-universe available-state
correlations: makes/contests0.779 and makes/rim-savings0.005. These are
unweighted descriptive correlations, not independent-source evidence.

Eight mapper emissions independently reproduce full cohort geometry exactly;
removing the added directions restores the original coefficients to7.77e−16.
Every original current/historical final control rating reproduces within1e−8.
Independent629-source,56-score-row,21-paired-point and player-minute score
checks pass; maximum score/pair gaps are7.11e−15/2.66e−15. Five focused tests
pass, including an independent likelihood check and unchanged-transform test.

The independent pair audit caught an evaluation-wrapper alias overwriting a
literal VPM reference with the primary model. Copying the reference before
aliasing fixes only those paired comparisons. Original code/pairs/lock are
archived under `preflight_pair_reference_alias`; hashes prove all model,
current-rating, game/team prediction and scorecard artifacts unchanged.
This was a bookkeeping repair, with no refit, tuning or new outcome choice.

## Reader and reproduction

[Side-by-side priors and ratings](http://127.0.0.1:8767/rim-defended-makes.html)
uses the existing local8767 server,582 recent positive-playing-exposure players
and all1556 saved players, searchable/sortable defensive/overall views and CSV.
Current forward and previous repaired boards remain available and unchanged.

RAPM source commands:

```sh
.venv/bin/python nba_pipeline/scripts/export_vpm_rim_defended_makes.py --workers 3
.venv/bin/python nba_pipeline/scripts/audit_vpm_rim_defended_makes.py
```

Sibling harness/audit/reader:

```sh
../rapm/.venv/bin/python vpm/run_harness.py \
  --config vpm/configs/research_harness_v54_rim_defended_makes.json \
  rim-defended-makes --stage all
../rapm/.venv/bin/python vpm/scripts/audit_rim_defended_makes.py \
  vpm/results/harness/vpm_rim_defended_makes_v1
../rapm/.venv/bin/python vpm/scripts/render_rim_defended_makes.py \
  vpm/results/harness/vpm_rim_defended_makes_v1
```

Curated receipt: sibling `vpm/evidence/rim_defended_makes_summary.json`.
Generated source measurements, states, fits, boards, logs and HTML/CSV stay
ignored. Browser checks cover search, both table views,582/1556 universes and
sorting; the new page links to unchanged current ratings/skills. No2027
outcomes, public deployment, existing-board overwrite, or promotion.
