# MPG in the selected predictive VPM prior

October 20, 2026. **Decision: retain MPG.** The matched removal test on the
selected clean mapper100 model favors retaining it. This does not establish
that the particular 365-day window or linear formulation is optimal.

## Exact feature and interpretation

`ctx_mpg` is mean regular-season minutes per positive-minute appearance in the
365 days strictly preceding the estimate date. It is not a completed-season
total, a future minutes projection, a time-decay weight, or a post-solve VPM
multiplier. A separate missingness flag accompanies it. `ctx_games` records
the number of appearances for inspection; it does not enter the mapper.

Playing time can carry role and selection information missing from counted
box-score/advanced features. Interpreting this as coaches' information is a
plausible mechanism, not an identified causal effect: injuries, roster depth,
age and coaching preferences also affect MPG. Reconstructing the raw feature
from the admitted player-game lines exactly reproduces all 9,570 emitted
cells across eight comparison dates (maximum gap zero).

## Controlled experiment

Immutable experiment: `vpm_mpg_decision_v1` in the sibling VPM repository.
The control is the selected **clean v61 `dmap_a100`** Parent VPM. The no-MPG
arm removes both `ctx_mpg` and its missingness flag from both sides, then
refits all remaining coefficients. It does not simply zero one coefficient.

The selected recipe first fits the joint O/D future-lineup profile at mapper
ridge10, then fits defense at ridge100 conditional on that offense. The
no-MPG arm repeats those two steps on the reduced basis. Therefore offense
is allowed to change when removing MPG; clean entrant priors, weights,
lineup4000/8000, TD550, actual FT, SQ12000/48000, estimate dates, future game
census, replacement rules, actual minutes and pace remain identical.

Cached v53 full-covariance profiles are mathematically exact for this reduced
basis: an independent reconstruction from each dated cohort's features and
sparse likelihood reproduces the information/rhs within `8.53e-14`, including
eligibility, imputation, centering and missingness. Mature future labels end
before each estimate date. Control coefficients match selected v61 exactly;
all eight control boards match within `1.78e-15`.

Only the predefined development 2020–22 minimum equal-season game MSE decides
between these two arms. Historical 2023–26 is report-only, already spent
stress evidence. No 2027 outcomes are read; no existing selected pointer or
ratings are replaced.

## Prediction results

Lower RMSE is better. Team win RMSE is on a 0–1 win-fraction scale.

| Period / test | Keep MPG | Remove MPG |
|---|---:|---:|
| Development game margin | **13.727018** | 13.739618 |
| Development team net rating | **2.807440** | 2.855957 |
| Stress game margin | **13.888098** | 13.925481 |
| Stress low-continuity game margin | **14.466517** | 14.588433 |
| Stress team win fraction | **0.084123** | 0.086217 |
| Stress low-continuity team win fraction | **0.110299** | 0.113752 |
| Stress team net rating | **3.095818** | 3.182084 |
| Stress veteran-override game margin | **13.968663** | 14.012589 |
| Stress veteran-override team win fraction | **0.087190** | 0.089435 |

The development game-MSE gain is `+0.346094 [-0.147080,+0.864366]`: its interval
crosses zero, although the point estimate wins the declared two-arm rule.
Development team-net gain is `+0.274771 [+0.024965,+0.710468]`.

Stress keep-MPG versus removal MSE gains, positive favoring retention:

| Test | Gain | Paired 95% interval |
|---|---:|---:|
| Game margin | +1.039756 | [+0.678591,+1.422782] |
| Low-continuity game margin | +3.542261 | [+1.984123,+5.014177] |
| Team win fraction | +0.000357 | [+0.000149,+0.000510] |
| Low-continuity team win fraction | +0.000774 | [-0.000014,+0.002064] |
| Team net rating | +0.541568 | [+0.294648,+0.733828] |
| Veteran-override game margin | +1.229117 | [+0.864769,+1.615072] |
| Veteran-override team win fraction | +0.000396 | [+0.000137,+0.000576] |

These are 4,920 stress games and120 team-seasons, with492 low-continuity games
and13 low-continuity team-seasons. All seven point estimates favor MPG; six
paired intervals exclude zero. Games use4000 paired game-cluster draws within
season; team results use whole-season blocks, with equal-season weighting.

On423,644 common native future actual-point possessions across4,273 games,
the prior-only gain is `+121.67e-6 [+68.61,+175.48]e-6`. The final VPM native
gain is smaller and unresolved: `+23.12e-6 [-4.27,+50.28]e-6`. This distinction
matters: a clearer prior-only increment does not imply a clearly established
final possession-level improvement.

The benchmark conditions on common actual future minutes and pace. It tests
player-quality prediction and is not a pregame availability forecast.
Intervals freeze learned mappings/selection and are unadjusted for earlier
research comparisons; architecture choices are retrospective.

## Current rating and reliability diagnostics

At September1,2026 the conditional physical slopes are **+0.043504 offensive
and+0.030668 defensive prior points/100 per extra MPG**, holding the other
mapped features fixed. The feature basis is centered, so a player's MPG
contribution is not simply slope times raw MPG. Entrant blending attenuates
that own-prior contribution when career NBA exposure is below2000 possessions.

| Player | MPG / appearances | Keep O / D / total | Remove O / D / total |
|---|---:|---:|---:|
| Nikola Jokic |34.86 /65|6.838 /2.595 /9.432|6.963 /2.552 /9.516|
| Paul Reed |13.86 /65|0.218 /3.430 /3.649|0.409 /3.500 /3.909|
| Stephen Curry |30.92 /43|5.557 /−0.822 /4.735|5.591 /−0.874 /4.717|
| Victor Wembanyama |29.17 /64|2.704 /5.918 /8.622|2.741 /5.890 /8.631|

Refitting correlated features means the final-rating change need not match the
isolated MPG contribution. For example, MPG's own-prior contribution for Jokic
is+0.502 O/+0.250 D, yet retaining MPG changes final total VPM by−0.083 versus
the refitted no-MPG model. MPG is not the cause of the prominent positive
defensive estimates for Jokic or Reed.

Among582 recent positive-exposure players,44 have only1–5 appearances in the
MPG window,40 have6–10,50 have11–20,88 have21–40 and360 have41+. There is no
explicit appearance-count reliability term on MPG itself. The entrant blend
uses career exposure and therefore does not solve every small recent-sample
case, especially an established player returning from injury. This remains a
real modeling limitation. The correct next sensitivity would compare a
specified reliability-aware MPG formulation in a new matched experiment;
there is no evidence here to invent a shrinkage constant or remove MPG now.

Full current O/D priors, isolated MPG contributions, entrant-attenuated
contributions, final ratings and no-MPG comparisons are saved in
`vpm/results/harness/vpm_mpg_decision_v1/mpg_current_ratings.csv`; all eight
dates are in `mpg_rating_diagnostics.csv`. Curated decision receipt:
`vpm/evidence/mpg_decision_summary.json`, with `keep_mpg: true`.

## Reproduction and audit

From the sibling VPM root, with the paired RAPM virtual environment:

```bash
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
../rapm/.venv/bin/python vpm/scripts/mpg_run_decision.py --stage all

OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
../rapm/.venv/bin/python vpm/scripts/mpg_audit_decision.py \
  vpm/results/harness/vpm_mpg_decision_v1

OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
../rapm/.venv/bin/python vpm/scripts/mpg_finalize_decision.py \
  vpm/results/harness/vpm_mpg_decision_v1

../rapm/.venv/bin/python -m pytest -q tests/test_mpg_decision.py
```

Two focused mathematical tests pass. Independent audit verifies429 locked
sources,16 dated mappings and16 direct raw lineup boards. Maximum independent
coefficient gap4.44e-16, direct lineup rating gap3.55e-15, prior-emission
gap2.66e-15, player-minute forecast reconstruction gap7.11e-15,56 score rows
and7 paired stress points within3.56e-15. No2027 outcomes or public deployment.
