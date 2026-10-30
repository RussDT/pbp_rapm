# Own turnovers and teammate turnover suppression

Owner request September 19, 2026: add an own/teammate turnover decomposition to
the Daniel Myers five-year shooting reader. Keep shooting fields untouched and
place turnover values in a separate possession-based section.

## Sample and units

`build_own_teammate_tov.py` loads canonical `processed/TOV22..26.parquet` for RS,
normalizes game/event keys, dedupes exact rows and removes conflicting keys.
Join raw turnover events (event_type5) by exact game/event to recover player1_id.
Require ten unique positive lineup IDs. Turnover events must have exactly one
identified actor among the offensive five; all non-turnover rows remain eligible.
No actor is fabricated for team/unassigned turnovers.

1,171,696 eligible possession opportunities, including 163,243 turnovers.
Exclude 9,624 unresolved turnovers (5.57% of recorded turnovers). Full coverage
receipts are in the generated `coverage.csv`. Because unresolved turnovers are
excluded from both likelihoods, call the result **resolved/player-attributed oTOV**,
not the standard all-turnover oTOV. The same selected sample is used for both
children and parent. Units are turnovers prevented per100 retained possession
opportunities, positive better. These are not points and not the shooting table's
denominator; do not add them to the six shooting columns.

## Model and exact allocation

Parent: one offensive and defensive coefficient per player, response `-Is_Turnover`,
season-mean centered; fixed O/D ridge3000/5000, no decay/rubberband/box priors.

For every possession, consider five focal-player rows with response
`-I(focal player committed the turnover)`. Non-turnovers have five zero responses.
The target sum across these five rows equals the negative turnover indicator.
The joint model can be parameterized as a lineup mean plus centered focal contrast:

```
y_focal = parent_prediction / 5
        + contrast_focal - mean(contrast_offensive_five)
```

Each parent's offensive coefficient contributes equally to the five focal rows.
Its own-row coefficient is therefore `(parent + 4*contrast)/5`; its coefficient
in each other player's row is `(parent - contrast)/5`. Sum the four other-row
coefficients for teammate suppression:

```
own_tov_impact = 100 * (parent + 4*contrast) / 5
teammate_tov_suppression = 100 * 4 * (parent - contrast) / 5
total_otov = 100 * parent
```

Own + teammate closes exactly to parent. Both parent and contrast are centered
by offensive exposure for display. This is an anchored joint mean/contrast model,
not two independently penalized children or a shooter-conditioned fit. It does
not imply that foul/turnover causation is identified from the recorded actor.

The contrast sufficient statistics avoid materializing all five focal rows:
`G = diag(O'1) - O'O/5`; `B = -own_turnover_counts + O'Is_Turnover/5`;
`y'y = .8 * total_turnovers`. The parent and centered contrast likelihoods are
orthogonal, so the parent error is constant when choosing the contrast penalty.
Three grouped-game folds, equal validation game weights, candidate alphas
1/10/100/300/1000/3000/10000/30000/100000/1e12, select strongest within one paired
fold SE. Both minimum and selection are300. Annual boards reuse this selection.
This CV tests focal allocation prediction, not a future-game advantage or a
causal interpretation. No 2027 outcomes enter.

## Interpretation

Own TOV value is centered adjusted own-turnover cost, not the raw count and not
a turnover rate per touch. Raw own turnover count/rate is separately exported.
Teammate suppression can capture taking on handling responsibility that leaves
teammates with fewer opportunities to commit turnovers, plus lineup association.
It is not proof of direct causal protection, pure playmaking or role-adjusted
ball security. This is intentionally not the independent rZTOV skill/allowance
model and imports none of its predicted allowances.

Pooled examples in turnovers prevented/100 retained possessions:
- Jokic: own -2.101, teammate +1.827, total -.274.
- Shai: own -.802, teammate +2.716, total +1.913.
- Doncic: own -2.485, teammate +3.137, total +.652.
- Curry: own -1.471, teammate +1.161, total -.310.

## Outputs and reader

Generated data stays ignored under `results/own_teammate_tov_2022_2026_rs/`.
Run the builder with OPENBLAS_NUM_THREADS=4 / OMP_NUM_THREADS=4, then the existing
`publish_v2_six_site_assets.py`. All pooled/annual/filtered CSVs include the two
children, total, role contrast, own raw counts/rate and possession exposure.
The separate reader table follows existing period/search/minimum-shooting-exposure
filters, has independent sorting and sticky player names. It stays in possession
units when the shooting conditional toggle changes. No existing shooting field
changes; all3896 reader player-window rows have turnover matches.

Tests compare sufficient statistics with an explicit five-focal-row matrix,
check exact parent closure, count zero-turnover opportunities and verify direction.
Artifact checks preserve every prior reader field and close the new decomposition
within1e-12. Browser checks cover filters, annual values and390px containment.

## Point conversion (current display)

Owner requested points. Run convert_own_teammate_tov_points.py after the turnover builder and before publishing. It preserves all original count fields and appends own_tov_points, teammate_tov_points, total_otov_points and tov_points_per_turnover. The price is canonical gross DECOMP-style mean Net_Diff on non-turnover FIRST_CHANCE rows in the matching RS window, with valid lineups and unique event keys. The pooled price is1.147199; annual2022–2026 prices1.124485/1.156825/1.152841/1.146135/1.155518. This is an opportunity-cost conversion, not a fitted rZTOV allowance coefficient. FIRST_CHANCE and TOV differ in boundaries/flags, so do not claim exact row-sample identity or a fresh point-valued joint fit. Apply one scalar per window to both children and parent, preserving closure. No continuation value is included. The reader displays points per100 retained TOV possessions, retains raw ownTOV/100, and explicitly labels shooting as points per100 shooting opportunities; they still cannot be summed directly. Shai pooled becomes own−.9205, teammate+3.1154,total+2.1948. Original CSV fields unchanged; point closure and browser checks pass.
