# Current Parent VPM: exact learned Net DECOMP lanes

Owner authorized October20,2026: make current VPM fully decomposable into the
Net DECOMP lanes after completing the defensive calibration tests. This is a
new isolated research adapter plus a coherent decomposition of saved Parent
estimates, not a silent switch to the older component-prior product.

## Frozen construction before new component fits

Use observed FG+FT and the same canonical possession design, first-chance
flags and observed continuation counts as current Parent. Export price-free
first-chance zone attempts/points, FT points and foul-type replacement weights,
zone misses and second-chance actuals. Preserve raw per-game continuation
counts before terminal-row filtering. No provider season FT percentages.
Do not reuse full-season-priced component values at partial historical cuts.
Reprice continuation and zone reference values only with the appropriate
pre-cut source games; mature future-label windows may use their completed
outcome sample, never source beyond label end/model emission.

Use Net's full six-shot grain: rim/mid/three frequency and conversion, plus
FT, net turnover opportunity and net second chance. Frequency receives full
net attempt value; mid-frequency carries the existing foul-type FT reference.
The same continuation bonus is paid out of second chance. Apply same-sample
non-turnover imputation and opposite net turnover value, with exact per-row
sum to actual points. Provide the existing compact five-shot rollup through
explicit linear reference contrasts; no separate player baseline bucket.

Fit each component prior through exactly the original matured future-lineup
profile information, preprocessing, free nuisance effects and ridge. Only the
response changes. Linearity then makes component coefficients sum to the
original parent coefficients without a fitted balancing bucket, equal split,
player-specific allocation or post-hoc scaling. For a defense-only conditional
mapper, decompose its fixed original offense first and use the same defensive
penalty/removed columns on every component. Verify exact sum for control and
the development-selected defensive challenger.

The entrant prior must also be learned componentwise by retargeting its
original season likelihood, fixed scaler, experience weights, prior precision,
season influence and ridge. Do not distribute an entrant residual by hand.
Rebuild/check the original total entrant coefficients before accepting the
component sum. Every player's own and blended component priors must sum to
the exact original own and blended total prior.

Solve component actual targets with the selected parent's common O/D penalty
and TD550; fit the mid/three SQ residual corrections with the frozen original
12000/48000 penalties and zero residual priors. Expose conversion EV plus
actual-minus-EV children and roll them back into their respective shot lanes.
Shared target/design/mean/precision gives exact prior/estimate/intercept
closure. Lane-specific alpha retuning is outside this request's decomposition;
changing those would change the total and require a separate predictive test.

Report all lanes as lineup-adjusted predicted team impact, not own player
box-score credit, uniquely identified causal value or a claim that original
public Net solves used the same priors/penalties. Preserve positive-good D,
separate O/D TD-exposure centering and prediction-preserving intercepts.
Require row/action/price/date/source/independent seven-test/forecast identities
and exact total-rating parity before a reader. Retain raw lanes, priors and
intercepts for all1556 players; default582 recent playing exposures. No2027
outcomes or public deployment. The decomposition should inherit the exact
saved parent forecasts; exact sum is coherence, not evidence of better total
prediction or of individual-lane calibration.

## Status

Complete: archived reproduction plus quarantined-source repair, independent source/model/forecast audits and clean reader. The final clean receipt below supersedes exact archived-total parity for v61.

## Source-integrity amendment before cleaned fits

The exact-reproduction v60 entrant estimator inherits two 2015 games later
quarantined by `long_spm_actual_warmup_v1/score_accounting.json`. Preserve that
reproduction; v61 excludes the four registered 2014/2015 game IDs from both
entrant outcomes and matched pricing counts (only two occur in its 2015–2025
RS training sample). Refit the total entrant mean and each component with the
same fixed scaler, experience weights and penalties. Do not force the cleaned
coefficients back to the archived coefficients. Re-solve and independently
compare clean parent totals and lane sums, reporting changes from the archive.
The own-skill cohort models and all tuning choices remain fixed. This is a
source repair, not performance selection.

## Implemented architecture

This makes the retained forward **Parent VPM fully decomposable**. It does not
reintroduce the old independently tuned ten-component product or add another
impact score. Both the current recipe and the v59 defensive-regularization
candidate have learned own-skill, entrant/background, combined-prior and final
lineup values in the same Net tree:

```
O = oTS + oTOV + oSC
D = dTS + dTOV + dSC
VPM = O + D
TS = Rim + Mid + Three + FT
```

The preferred full-frequency tree has rim/mid/three frequency and conversion,
plus FT, turnover and second chance: nine actual-point lanes. Mid/three
conversion each expose EV and actual-minus-EV, yielding eleven atomic display
lanes. The compact five-shot convention is an exact alternate rollup: move
its two learned mid-reference contrasts from rim/three frequency into mid;
with the two SQ children it has ten atoms. No balancing baseline is added.

For a fixed design and precision, the ridge response to a sum equals the sum
of its responses. Each component therefore uses the parent's likelihood,
preprocessing, exposures and regularization. The profiled future-outcome
mapper and the entrant mapper are also linear in their target RHS. Component
means sum to the total mean before the joint lineup solve. Different independent
component penalties would change the total; this experiment does not select them.
The strong SQ child solves retain 12000/48000; the weak parent residual response
is subtracted from conversion before adding the strong response, exactly as in
Parent VPM. Center O/D on their own full-universe TD exposures and retain each
prediction-preserving intercept.

### Targets and source contract

`export_vpm_net_lane_atoms.py` taps the canonical clean processor before
pricing. It retains game/date, actual FG/FT, first-chance shot counts/misses,
FT-reference opportunities and unfiltered per-game continuation counts.
`vpm_net_lane_targets.py` only prices caller-admitted games; current cuts use
strictly earlier games. Per-zone continuation is raw follow-up points divided
by raw first misses. Frequency is the full zone attempt value times attempts;
conversion is actual FG plus its continuation bonus minus frequency. The same
bonus is paid back from realized second chance. The FT reference enters mid
frequency and leaves FT exactly. Turnovers receive the nonturnover weighted
first-chance mean in each child and its negative in TOV. All nine sum to actual
points. Provider FT percentages and realized future outcomes are unnecessary.

The 23-file measurement archive spans 2015 RS and 2016–2026 RS/PS, 3,012,316
canonical possessions. It retains quarantined rows as source measurements;
clean training explicitly excludes them. Independent comparisons reconstruct
the canonical Net builder on the full 2016 RS and a freshly truncated
January 1 prefix, including TD550. All component gaps are below 1.4e-14,
row closure below 9e-16. All 22 deployment files match the existing actual-point
possession census and raw continuation counts. The initial game-ID count join
failed preflight; it was corrected and all source files rebuilt before fitting.

v60 is the exact archived-parent reproduction. v61 uses the same verified
own-skill component maps, excludes the two applicable quarantined 2015 RS games
from entrant outcomes and pricing, fits a clean total entrant mean directly,
and independently fits its additive means. It then directly solves the clean
parent and each lane. The shared original feature scaler is fixed. The
four-ID exclusion registry comes from the independent historical scoring gate,
not from forecast performance. Preserve v60 separately; do not claim v61 exactly
reproduces the archived ratings.

### Interpretation and operating surface

Lane values are predictive lineup impact in points/100, with defense positive
good. Net SC is follow-up scoring after continuation payback, not a player's
own DREB count or unique causal rebound credit. Lane priors may legitimately
cross basketball-stat categories; they are jointly calibrated impact mappings.
Exact closure establishes coherence, not independent component forecast
calibration or a unique causal explanation. No lane-specific uncertainty claim
or component-specific tuning is made.

Reader: http://127.0.0.1:8767/net-decomposition.html. It exposes final/own/
background/combined values, O/D/total, preferred and compact atoms, player detail,
full downloads, 582 recent players and all 1,556 model players. Current recipe
and defensive candidate are explicitly named. The archived v60 reproduction
is retained under `net-reproduction/net-decomposition.html`; old ratings readers
remain unchanged. This is the September 1 archive, not an opening-night roster
or a running daily update. No public deployment or 2027 outcomes.

## Completed clean receipt — October 20, 2026

The clean entrant source removes 344 possessions and the two matching pricing
games, `0021400843` and `0021401185`. The 2014 quarantines are absent from this
training interval. Maximum archived side-rating movement is 0.011900 at the
current cut and 0.015576 over all eight cuts; this is measured, not forced away.
Maximum archived historical game-margin movement is 0.009727, team net 0.004861.
Jokić and Reed move by less than 0.0001 through the joint lineup solve, despite
zero entrant weight; established-player priors themselves remain unchanged.

| Clean parent recipe | Development game RMSE | Stress game RMSE |
|---|---:|---:|
| Current | 13.741067 | 13.895302 |
| Defensive mapper 100 | 13.727018 | 13.888098 |

The frozen defense choice remains unchanged. Recomputed clean candidate-minus-
current paired gains retain the same conclusion: game +0.200161
[-0.029125,+0.443205], low-continuity game +0.012666
[-0.670477,+0.747274]; low-continuity team win percentage worsens with its
interval below zero. All seven clean paired rows and 56 equal-season scores
are retained, including team/veteran tests. No new selection occurred.

Independent clean audit: 778 source pins, 16 own maps, 8 entrant maps and
16 boards; component-prior emission gap zero, direct lane-rating gap 1.4e-13,
mapper coefficient gaps below 9e-11. Summed lane forecasts equal the direct
clean parent across 8,289 games and 210 team-seasons to 1.2e-12. Independent
score reconstruction matches all 56 RMSEs within 3.6e-15. Three target-algebra
and four defense tests pass. Browser search/player detail/full1556/recent582,
model/basis/detail controls and 582-row atomic-prior CSV verified; CSV/source
gap below 5e-17. Curated source, numerical, score and reader receipts live in
sibling `vpm/evidence/net_lane_clean_summary.json`; v60 archive receipt is
`net_lane_decomposition_summary.json`.

The decomposition exposes the concern rather than solving it cosmetically:
Jokić's current-recipe defense is approximately TS +0.466, TOV -0.550,
SC +3.102 = +3.017. The defense candidate reduces his SC to approximately
+2.587 and total D to +2.595. These are net-continuation lineup estimates,
not proof of individual rebound credit. Reed's candidate D remains +3.430.

### Reproduction

From RAPM, use `nba_pipeline/scripts/export_vpm_net_lane_atoms.py` for the
23-file archive and `audit_vpm_net_lane_source.py` for canonical/full/prefix
checks (CLI `--help` gives source/export parameters). Source audit must pass
before model fitting. From sibling VPM, use immutable v60 with
`net-lane-decomposition --stage all`, its independent audit, then immutable
v61 with `net-lane-clean --stage all`. Run
`vpm/scripts/audit_net_lane_clean.py vpm/results/harness/vpm_net_lane_clean_v1`
and `vpm/scripts/render_net_lane_clean.py` on that same root. The clean audit
reconstructs the target equations separately and re-emits own/background
component means, direct lanes, total-parent forecast parity, seven tests and
paired comparisons. Preserve locked source/helper vintages; this uses existing
local prerequisite caches, not a fresh-install historical publication claim.

## Owner-selected default — October 20, 2026

The owner subsequently chose mapper100. The clean Net reader now defaults to
`dmap_a100`, labeled Selected VPM; mapper10 is the previous comparison. The
ratings and skills pages/exported priors use that same clean board through
`vpm/configs/current_model.json` and `render_selected_ratings.py`. No lane/model
refit or new tuning occurred. The previous candidate-only recommendation above
records the completed experiment; the owner selection is the current operating
choice. Predictive limitations remain unchanged; this is not public deployment.
