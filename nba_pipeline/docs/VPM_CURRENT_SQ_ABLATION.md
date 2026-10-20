# Current forward Parent: mid/three ShotQuality ablation

Owner requested October19,2026: current VPM with no SQ correction, mid only,
three only, both, plus stronger defensive three-point residual shrinkage.
Same NBA VPM research slot; no2027 outcomes or public promotion.

## Protocol frozen before fitting

Control is retained v53 `forward_expanded_mpg_a10p0`, not role-aware, unsplit
features, unselected rim makes, or Component VPM. Preserve its dated own and
entrant prior vectors exactly, including preprocessing/calibration. Parent
4000/8000, TD550, actualFT, repaired dated SQ/count/continuation source,
lineup census, intercept/reference, minutes/fills, home and evaluation stay
fixed. This is a likelihood-correction ablation, not a refit of the prior or
a new tuning of the aggregate parent.

On each identical dated training design, fit actual-points parent with its
saved combined prior. For each zone, the SQ correction is its zero-prior
residual fit at12000/48000 minus that residual at parent4000/8000. Compare:

1. `no_sq`: direct actual-points parent, neither correction.
2. `mid_only`: only midrange correction.
3. `three_only`: only three-point correction.
4. `both`: both original corrections; must reproduce saved control exactly.
5. `both_d3_96000`: both, three residual penalty12000/96000.
6. `both_d3_192000`: both, three residual penalty12000/192000.

The last two shrink the actual-minus-EV defensive three-point residual more
strongly toward zero. They do not manually shrink total defense or a player's
defensive shooting prior. A joint O/D solve can move the offensive three-point
residual coefficients too; disclose that rather than label them D-only changes.
Mid penalties and the three residual offensive penalty12000 stay fixed.

Emit seven historical preseason cuts2020–26 and the September1,2026 review
cut. Report the same seven conditional game/season tests and paired MSE
intervals, with prior incumbent/previous repaired/EPM forecast parity. Freeze
the best four-way ablation and stronger-D3 choice separately by literal minimum
equal-season2020–22 gameMSE, lexical ties. Also report the joint six-way winner;
no selection from2023–26 stress, native possession results or player movement.

Report native future actual-possession errors on the same accepted game/player
census, paired by game and equal-weighted across2023–26 seasons, including
both versus no SQ, each marginal zone increment, interaction and stronger-D3
versus both. Intervals are unadjusted, tests overlap, and later seasons are
spent evidence. Shared actual future minutes/pace is conditional player-quality
evidence, not a pregame availability forecast. Historical architecture/source
publication availability remains uncertified.

Independently rebuild dated raw targets/weights/design/normal equations and
coefficient/reference arithmetic, prove linear split closure, unchanged prior
vectors and exact control reproduction, and independently reconstruct the
seven score definitions and paired points. Save immutable sibling v57 config,
source locks, ignored geometry/boards/forecasts and small curated evidence.
Show results in a new page on existing8767 without changing current ratings.

## Owner clarification and reporting scope

After all six registered fits completed, the owner clarified: "i mean 48000
was probably fine" and "doent need to double or quad that imo". Retain D3
48000. The report and its four-way choice use only no SQ/mid/three/both.
The two completed exploratory higher-penalty fits remain archived with their
original selection record; they are set aside and not adopted. An appended
`owner_update.json` records this without mutating the locked config, raw fits,
forecasts or original selection receipt. Historical replay commands below
reproduce the full registered experiment; the renderer applies the updated
four-way reporting scope. No further D3 search or fit is authorized here.

## Completed four-way result, October19

**Retain both corrections with the existing12000/48000 residual penalties.**
Both wins the2020–22 four-way development comparison and has the lowest later
game/team-net error among those four fixed arms. Every prior mean and entrant
weight is identical; this isolates the likelihood corrections in current
Parent VPM, not the old v18 replay or a new prior architecture.

|SQ correction|2020–22 game RMSE|2023–26 game RMSE|2023–26 team-net RMSE|
|---|---:|---:|---:|
|None|13.788629|13.956688|3.155076|
|Mid only|13.781589|13.922137|3.106363|
|Three only|13.742301|13.925285|3.131277|
|Both, current|13.740987|13.895289|3.091360|

Both lowers later game RMSE by0.061399 scoreboard points, about0.44%, or0.88%
of MSE. The matched paired game-MSE gain is+1.710078[+1.026498,+2.415750].
Mid added after three gives+0.834498[+0.570901,+1.100007]; three added after mid
gives+0.746836[+0.186493,+1.311292]. Both incremental game gains are positive
with unadjusted intervals excluding zero. Predictive MSE gains are not additive,
even though the coefficient corrections add exactly.

Full seven-test comparison, both versus no SQ (positive favors both):

|Later test|Paired MSE gain|95% interval|
|---|---:|---|
|Game margin|+1.710078|[+1.026498,+2.415750]|
|Low-continuity game|−1.222819|[−3.277490,+0.881712]|
|Team win percentage|+0.000135|[−0.000217,+0.000400]|
|Low-continuity team win|−0.000427|[−0.001660,+0.000847]|
|Team net rating|+0.397996|[−0.260949,+0.925401]|
|Veteran game margin|+1.560543|[+0.895174,+2.221425]|
|Veteran team win|+0.000110|[−0.000219,+0.000401]|

Same4,920 stress games/120 team-seasons; low continuity492 games/13 team-seasons.
The overall/veteran game gains are supported, while season-level improvements
remain unresolved. Low-continuity errors have worse point estimates, also
unresolved. Do not summarize this as an all-seven-test win or new EPM win.

## Native future actual possessions

On the common423,644 native future possessions/4,273 accepted games, MSE is
1.415075143 no SQ,1.415027041 mid only,1.414972855 three only,
1.414933460 both. The saved current control reproduces exactly.

|Native contrast|MSE gain ×1e6|95% interval ×1e6|
|---|---:|---|
|Both versus none|+141.683|[+81.049,+199.821]|
|Mid versus none|+48.103|[+21.977,+74.797]|
|Three versus none|+102.288|[+55.354,+148.432]|
|Both versus mid|+93.580|[+46.241,+141.742]|
|Both versus three|+39.395|[+11.595,+64.578]|

This also supports both conditional increments. Identical prior-only MSE
1.415538267 across all six original arms confirms that the predictive change
comes from the likelihood, not new priors. The source, vintage, spent-window,
actual-minute/pace and repeated-testing limitations above still apply.

## Verification and operation

Immutable sibling config v57 writes
`vpm/results/harness/vpm_current_sq_ablation_v1`. Independent audit rebuilds
all8 raw dated prefixes (15,055,097 rows summed across overlapping prefixes,
not unique possessions),142 locked sources and48 registered arm/cut boards.
Maximum geometry/solution gap4.42e−17, raw-target RHS4.69e−13, raw/centered
rating gap4.54e−13, prior gap0. Eight original control boards reproduce; all
old benchmark forecasts remain unchanged. Independent player-minute score
gap7.11e−15; all126 registered score rows/77 paired points reconstruct.
The scoped report contains98 score rows/63 paired points, omitting the
set-aside higher-D3 arms. Eight unique focused tests pass (three new ablation,
five existing SQ-parent).

The first independent checker omitted the five-player factor in its intercept
reconstruction and stopped. Correcting that audit-only formula resolved it;
models, priors, forecasts and scores were unchanged. The first log is retained
as `audit_before_intercept_fix.log`; a focused five-player centering test now
checks the convention. The successful auditor uses raw source reconstruction,
not the fitting module's stored coefficient arithmetic.

From sibling VPM, historical full reproduction and current reporting:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../rapm/.venv/bin/python vpm/run_harness.py --config vpm/configs/research_harness_v57_current_sq_ablation.json sq-ablation --stage all
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../rapm/.venv/bin/python vpm/scripts/audit_sq_ablation.py vpm/results/harness/vpm_current_sq_ablation_v1
../rapm/.venv/bin/python vpm/scripts/render_sq_ablation.py vpm/results/harness/vpm_current_sq_ablation_v1
```

Retain `owner_update.json` when reproducing the reported scope. The source
runner's registered six-arm settings are historical evidence, not authority
to resume a D3 search after the owner clarification. Curated sibling evidence
is `vpm/evidence/current_sq_ablation_summary.json`; source geometry, fits,
logs, forecasts and HTML/CSV remain ignored. The new
[four-way predictive reader](http://127.0.0.1:8767/sq-ablation.html) preserves
the original ratings pages. Browser checks cover the four rows, marginal
selectors, exact gains,98-row downloaded scorecard and63 reported paired rows.
Both retains D3 at48000; no public deployment or Component VPM promotion.
