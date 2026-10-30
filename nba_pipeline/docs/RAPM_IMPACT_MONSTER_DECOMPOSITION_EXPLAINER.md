# How RAPM and the Impact Decomposition Work

> Evidence snapshot: 2026-07-28. This note describes the current repository
> code and local pipeline artifacts. It does not use mock explorer values.

## 30-second summary

RAPM asks a narrow question: **when a player is on the floor, how does his
team's scoring change after accounting for the other nine players?** It
estimates separate offensive and defensive effects from possession-level
lineups, then expresses them as league-centered points per 100 possessions.

The decomposition asks a second question: **where in the possession did that
estimated value appear?** The current VPM board organizes each side into:

```text
offensive impact = oTS + oTOV + oSC
defensive impact = dTS + dTOV + dSC
total impact     = offensive impact + defensive impact
```

`TS` is the shot-value branch, `TOV` is point-valued first-chance turnover
impact, and `SC` is scoring after the first offensive miss. The current VPM
artifact has eight children under each `TS` branch, one `TOV` component, and
one `SC` component: ten components on offense and ten on defense. Its checked
identity tests require every component, factor, side, and net total to add
exactly. [Current VPM manifest](../../vpm-explorer/public/data/vpm-manifest.json)
and [identity tests](../../vpm-explorer/tests/rendered-html.test.mjs#L107-L155).

Positive means helpful on both sides:

- `+1.0` offense means roughly one point added per 100 possessions relative to
  the model's league-centered baseline.
- `+1.0` defense means roughly one opponent point prevented per 100
  possessions.
- Negative values mean the reverse.

These are adjusted lineup estimates, not proof that one visible action caused
the value.

## First: three related trees must not be mixed

The repository currently contains three related but different decompositions.

| Surface | Player-facing structure | Current status |
| --- | --- | --- |
| **Forward Net DECOMP** | Five continuation-priced shot children + FT value + net TOV value + second-chance conversion | Selected `weighted_factors_decomp_net_*` family; 1Y-5Y migration in progress |
| **Gross compatibility DECOMP** | Five gross first-chance shot children + FT value + TOV value + physical second-chance value | Rollback and matched-comparison `weighted_factors_decomp_*` family |
| **Current VPM board** | `TS + TOV + SC` on offense and defense; `TS` has eight EV-target children, for ten components per side | Current frozen EPM12 x2 research board in `vpm-explorer/public/data/` |
| **Older six-factor weighted factors** | `oTS/oTOV/oREB` and `dTS/dTOV/dREB`, fitted to RAPM with a second-stage weighted regression and a residual | Legacy/supporting surface, not the forward Net DECOMP |

The active public RAPM tree is declared in
[PUBLIC_DECOMP.md](./PUBLIC_DECOMP.md#public-tree) and assembled by
[build_decomp_weighted_factors.py](../scripts/build_decomp_weighted_factors.py#L1-L35).
The VPM ten-component tree is declared by its
[live manifest](../../vpm-explorer/public/data/vpm-manifest.json) and its
[product source](../../vpm-explorer/app/page.tsx#L108-L225). The older
six-factor tree is built through a player-level WLS regression in
[03_run_rapm_analysis.py](../scripts/03_run_rapm_analysis.py#L493-L630).

In reader-facing product language, **Impact Monster is the ten-factor
breakdown**. RAPM is the parent impact total; the ten factor types are each
estimated on offense and defense and add back to their corresponding RAPM
side. The `TS/TOV/SC` stages are a friendly organizer, not the definition of
Impact Monster and not another rating to stack on top. The code still calls
the current ten-component artifact the VPM board, while the active public RAPM
DECOMP remains the separate eight-component pipeline output.

## What RAPM is estimating

For each possession, the model knows:

- the five offensive players;
- the five defensive players;
- the possession outcome, such as points, a turnover indicator, or a
  component-specific point value.

Each player receives a separate offensive and defensive coefficient. The same
design is fitted repeatedly with different possession targets to estimate the
parent RAPM and its components. The current solver creates offense and defense
lineup indicators, centers the outcome, fits alternating ridge regressions,
and recenters player coefficients by their offensive or defensive possession
exposure. It exports coefficients in per-100 units.
[Design matrix](../scripts/rapm.py#L2051-L2140),
[ridge solve](../scripts/rapm.py#L2462-L2635), and
[centering/export](../scripts/rapm.py#L3500-L3626).

### A simple lineup analogy

Imagine that two five-player units share most of their minutes. Unit A scores
well, but every time one guard replaces another, the offense improves a little;
every time one center replaces another, the defense improves a little.

A raw plus-minus table sees only the whole five-player result. RAPM treats all
of those substitutions across all games as an overlapping puzzle. It asks
which player coefficients best explain the possession outcomes while
accounting for the teammates and opponents sharing each row. Regularization
keeps the puzzle from assigning extreme credit when two players almost always
appear together.

This adjustment is why RAPM is more informative than simply dividing a
lineup's margin among its five players. It is also why it is not magic:
players with nearly identical substitution patterns remain hard to separate.

## What the current solve adjusts for

The active rolling RAPM DECOMP uses:

- the offensive and defensive players on every possession;
- separate regular-season/playoff season-phase effects to absorb changing
  league baselines;
- score-margin-by-period “rubberband” effects based on the pre-possession
  margin;
- ridge penalties of `2000` on offense and `4000` on defense;
- pure RAPM for the overall parent.

Those choices are fixed by
[run_decomp_rolling.py](../scripts/run_decomp_rolling.py#L36-L55) and its
[component command](../scripts/run_decomp_rolling.py#L101-L125). The
rubberband design uses six non-close score states across four period buckets;
close possessions are the reference.
[Rubberband construction](../scripts/rapm.py#L2247-L2360).

The current VPM board is a different research specification. Its manifest
identifies a 550-day half-life, EPM12 predictive-skill priors at scale `2.0`,
no complete-EPM blend, and no absence adjustment. Its underlying component
alphas were previously selected and then frozen.
[Current manifest](../../vpm-explorer/public/data/vpm-manifest.json) and
[current model card](../../vpm-explorer/app/page.tsx#L1324-L1374).

Neither model automatically controls for every basketball context. In
particular, the published parent coefficients do not isolate play call,
coaching, a specific assigned defender, tracking-based spacing, injuries, or
nonlinear five-player chemistry unless such information is explicitly in that
model.

## The gross compatibility RAPM DECOMP: eight components

The active public RAPM output uses the following player-facing components on
both offense and defense:

1. Rim frequency value
2. Rim finishing value
3. Midrange/base shot value
4. Three-point frequency value
5. Three-point finishing value
6. Free-throw value
7. Point-valued first-chance turnovers
8. Clean second-chance scoring

The first five form the shot tree. On every source action, the processor
defines them so they add to first-chance field-goal value:

```text
rim frequency
+ rim finishing
+ midrange/base value
+ three-point frequency
+ three-point finishing
= first-chance field-goal value
```

The exact action formulas use the loaded sample's average rim, midrange, and
three-point values. Rim and three frequency are valued relative to the same
midrange counterfactual; the middle component carries that common base.
[Shot formulas](../scripts/process_rapm_blocks/clean_chance_utils.py#L18-L60).
The processor rejects action- or possession-level gaps above `1e-9`.
[Possession closure check](../scripts/process_rapm_blocks/clean_chance_utils.py#L847-L870).

This means “rim frequency value” is not merely a rim-attempt percentage, and
“midrange value” is not merely midrange shooting percentage. They are
point-valued accounting pieces built to share one possession denominator.

### Free throws

First-chance free throws use shooter-expected FT value rather than the result
of one make or miss. The FT branch subtracts a foul-type-aware replacement-shot
baseline so the field-goal and FT branches do not double count the possession.
[FT target construction](../scripts/process_rapm_blocks/clean_chance_utils.py#L311-L338).

### Turnovers

`ALT_TOV_VALUE` prices a true first-chance turnover at the loaded sample's
non-turnover first-chance scoring baseline. A turnover is therefore a lost
point opportunity, not just a `0/1` rate.
[Baseline construction](../scripts/rapm.py#L291-L365) and
[turnover target](../scripts/rapm.py#L1579-L1616).

### Second chances

`SECOND_CHANCE_CLEAN` assigns scoring after the first offensive miss to the
second-chance branch on the same standard possession universe. A missed shot
followed by an offensive rebound and no later points contributes zero direct
SC scoring; so does a post-rebound turnover. The opportunity and failed
conversion still affect the lineup context, but SC itself is a points branch.
[First-/second-chance split](../scripts/process_rapm_blocks/clean_chance_utils.py#L514-L535)
and [physical SC target](../scripts/process_rapm_blocks/process_second_chance_clean.py#L13-L32).

### How the public file closes

At the processed-row level, the shot children are exact. The player
coefficients are then obtained from separate ridge solves, so finite solver
tolerance, independent recentering, and CSV rounding can leave small
child-to-parent gaps. The public builder retains those audit gaps, constructs
the displayed EFG parent as the amount needed after FT, TOV, and SC, and then
enforces:

```text
displayed EFG + FT + TOV + SC = offensive RAPM
displayed EFG + FT + TOV + SC = defensive RAPM
```

[Public assembly and audit columns](../scripts/build_decomp_weighted_factors.py#L144-L202).
In the local `22_26` artifact, the rounded offense and defense display sums
match their parents exactly; the independently solved five-shot-child gap is
small but nonzero. That is an accounting audit fact, not a new basketball
component.
[Local 2022-26 artifact](../master_results/weighted_factors_decomp_22_26_all_rb_se_a2000_4000.csv).

One additional publication rule matters: the recovered
`full_possession_v1` policy scales the displayed TOV parent and its two
children by a sample-specific multiplier, then moves the identical delta out
of SC so total RAPM does not change.
[Transfer policy](../scripts/build_decomp_weighted_factors.py#L129-L142) and
[manifest](../config/decomp_tov_sc_transfer_multipliers.json). Thus the active
public file's `oSC/dSC` is an allocated SC display value, while the unmodified
`SECOND_CHANCE_CLEAN` solve remains available as an audit input.

## The current VPM board: six factors, ten components

The current VPM board presents three possession stages on each side:

| Factor | Plain-English meaning | Additive children |
| --- | --- | --- |
| `oTOV` / `dTOV` | Value from avoiding / forcing first-chance turnovers | `ALT_TOV_VALUE` |
| `oTS` / `dTS` | Point-valued first-shot and FT branch; not box-score TS% | Eight shot/FT components |
| `oSC` / `dSC` | Scoring after the first offensive miss / preventing it | `SECOND_CHANCE_CLEAN` |

The eight `TS` children are:

1. Rim frequency
2. Rim finishing
3. Midrange expected value
4. Midrange actual-minus-expected residual
5. Three-point frequency
6. Three-point expected value
7. Three-point actual-minus-expected residual
8. Free-throw value

VPM keeps the rim targets, replaces the midrange and three-point make targets
with leakage-safe walk-forward expected value, and carries
`actual - expected` in two residual targets. EV changes how the signal is
estimated; EV plus residual still targets actual scoring.
[VPM target formula](../scripts/build_vpm_target_swap.py#L1-L20) and
[row-level closure enforcement](../scripts/build_vpm_target_swap.py#L283-L385).

The exact VPM identities are:

```text
oTS = sum(eight offensive shot/FT children)
dTS = sum(eight defensive shot/FT children)

oTOV = o_ALT_TOV_VALUE
dTOV = d_ALT_TOV_VALUE

oSC = o_SECOND_CHANCE_CLEAN
dSC = d_SECOND_CHANCE_CLEAN

VPM offense = oTS + oTOV + oSC
VPM defense = dTS + dTOV + dSC
VPM total   = VPM offense + VPM defense
```

The frozen artifact manifest reports closure within floating-point precision,
and the shipped tests recompute every identity for all players.
[Manifest closure](../../vpm-explorer/public/data/vpm-manifest.json) and
[full-board tests](../../vpm-explorer/tests/rendered-html.test.mjs#L107-L155).

### Reused internal names

Two internal column names change meaning in the VPM target lane:

- public `DECOMP_MID_VALUE` is a reserved zero surface; VPM uses it for the
  midrange actual-minus-EV residual;
- public `DECOMP_EFG` is the public shot parent; VPM uses that registered alias
  for the three-point actual-minus-EV residual.

The VPM-facing product labels **MID RESID** and **3P RESID** are clearer than
the internal field names.
[Target aliases](../scripts/build_vpm_target_swap.py#L55-L66) and
[current product labels](../../vpm-explorer/app/page.tsx#L136-L180).

## Worked synthetic example

> Illustrative only. These numbers do not describe a real player.

Suppose a player's offensive shot-value components are:

| Offensive TS child | Value |
| --- | ---: |
| Rim frequency | `+0.30` |
| Rim finishing | `+0.25` |
| Midrange EV | `-0.10` |
| Midrange residual | `+0.05` |
| Three-point frequency | `+0.20` |
| Three-point EV | `+0.15` |
| Three-point residual | `-0.05` |
| Free throws | `+0.20` |
| **oTS** | **`+1.00`** |

Now add `oTOV = +0.35` and `oSC = +0.15`:

```text
offensive impact = +1.00 + 0.35 + 0.15 = +1.50
```

Suppose the player's defensive factors are `dTS = +0.40`,
`dTOV = +0.20`, and `dSC = -0.10`:

```text
defensive impact = +0.40 + 0.20 - 0.10 = +0.50
total impact     = +1.50 + 0.50       = +2.00 points per 100
```

The negative `dSC` does not mean the player is “a bad rebounder.” It means the
model estimates worse-than-baseline opponent scoring after the first offensive
miss in that player's adjusted defensive minutes. That could reflect rebound
prevention, the quality of attempts allowed after rebounds, conversion, lineup
context, or noise.

## `oSC/dSC` versus `oREB/dREB`

These names are not interchangeable.

- `oSC/dSC` are **point-valued second-chance scoring** branches. In the current
  VPM board they map directly to `SECOND_CHANCE_CLEAN`.
- `oREB/dREB` are **older rebound-opportunity factors**. Their source target is
  the `0/1` offensive-rebound result on rebound chances, and the older
  weighted-factor pipeline maps those coefficients into RAPM through a
  player-level WLS regression.
- `SHOOTER_OREB`, shooter miss recoverability, and the game-held-out
  second-chance opportunity prior are separate research surfaces. They do not
  enter the forward Net DECOMP or VPM total.

[REB target/signs](../scripts/rapm.py#L1717-L1722),
[legacy WLS mapping](../scripts/03_run_rapm_analysis.py#L493-L630), and
[SHOOTER_OREB boundary](./RAPM_METHODOLOGY.md#4b-shooter_oreb-shooter-miss-recoverability).

For product copy, use:

- **offensive second-chance value** for `oSC`;
- **defensive second-chance prevention** for `dSC`;
- **offensive-rebound opportunity impact** only when describing `oREB/dREB` or
  the explicit OREB research surfaces.

## Authoritative deeper splits—and what not to stack

The current pipeline has useful deeper diagnostics:

- **FT frequency + FT severity = FT value.** Frequency prices an FT trip at
  the sample-average trip value; severity is above/below that average and
  combines trip-type mix with expected FT shooting.
  [FT split](../scripts/process_rapm_blocks/clean_chance_utils.py#L322-L375).
- **Bad-pass TOV + other/scoring TOV = TOV.** “Scoring TOV” is the remainder
  after bad passes, including lost balls, travels, offensive fouls, and other
  non-bad-pass turnovers.
  [TOV classification and child targets](./tov-decomposition.md#classification).
- **Rim = rim frequency + rim finishing** and **three = three frequency +
  three finishing/EV** are safe rollups of the declared children.

The active public builder publishes the FT and TOV children as drilldowns.
Assist points, rim assists, passer/receiver, shooter/nonshooter, on-/off-ball,
transition/half-court, shot-clock, defender-role, and OREB-opportunity outputs
are separate mechanism or research surfaces. Some have their own exact parent
identities, but they are **not additional children of the current VPM total**
and must not be stacked onto it.
[Public component list](../scripts/run_decomp_rolling.py#L38-L55),
[role-model status](./RAPM_METHODOLOGY.md#2b-preferred-joint-ots-shooter-interaction-model),
and [current product separation](./VPM_REPRODUCTION.md#vpm-lab).

## Regularization and uncertainty

RAPM has a difficult identification problem: many players share most of their
minutes, and some have small samples. Ridge regularization penalizes large
coefficients unless the possession evidence consistently supports them.

- With no player prior, coefficients shrink toward zero.
- With a player prior, coefficients shrink toward that prior.
- A larger alpha means a stronger pull and less sensitivity to the observed
  sample.
- Time decay reduces the effective weight of older possessions; it does not
  create extra observations.

The implementation's prior-centered objective and fixed offense/defense ridge
blocks are explicit in
[rapm.py](../scripts/rapm.py#L2549-L2614). The current VPM CSV exposes raw and
time-decay-effective possessions, and its manifest records the half-life and
prior source.
[VPM CSV](../../vpm-explorer/public/data/vpm.csv) and
[manifest](../../vpm-explorer/public/data/vpm-manifest.json).

Regularization is not the same as a player confidence interval. The active
public DECOMP CSV and current VPM CSV do not publish per-player standard errors
or credible intervals. Possessions and effective possessions are useful
reliability context, but close player differences should not be treated as
precisely ordered. The game-clustered intervals in the VPM research documents
compare model-level future prediction error; they are not error bars around
an individual player's rating.
[Hardened harness evidence](./VPM_REPRODUCTION.md#hardened-research-harness).

## What the components do not prove

The decomposition does **not** prove:

- that the named player personally took every shot, committed every turnover,
  or collected every rebound represented by his lineup coefficient;
- that “rim access” is purely driving, spacing, passing, or screening;
- that a defensive shot-value coefficient belongs to the nominal nearest
  defender;
- that second-chance value is the same thing as rebounding ability;
- that five player coefficients sum to an exact lineup forecast;
- that pair synergy or role compatibility is already contained in the
  additive total;
- that a small difference between players is statistically meaningful;
- that an outcome-derived component is a causal player skill.

The current VPM product source explicitly labels five-player sums as additive
studies rather than lineup or synergy forecasts, and describes player
coefficients as adjusted signal rather than causal proof.
[Five-player boundary](../../vpm-explorer/app/page.tsx#L1178-L1257) and
[model card](../../vpm-explorer/app/page.tsx#L1324-L1381).

## Glossary

**Adjusted**  
Estimated while the model simultaneously accounts for the teammates and
opponents on the possession.

**Alpha**  
The ridge penalty. Larger values impose stronger shrinkage.

**Component**  
One point-valued child target, such as rim frequency value or turnover value.

**Effective possessions**  
The exposure remaining after time-decay weights are applied.

**League-centered**  
Expressed relative to the model's possession-weighted baseline; zero is not
“no basketball value,” but average within the modeled frame.

**Prior**  
A pre-possession player estimate used as the center of shrinkage. The current
VPM board uses date-safe predictive skill fields, not complete EPM totals.

**RAPM**  
Regularized Adjusted Plus-Minus: a ridge-regression estimate of player impact
from possession outcomes and the ten players on the floor.

**Regularization / shrinkage**  
A conservative pull toward zero or a prior that stabilizes noisy,
collinear estimates.

**Residual**  
A defined remainder. In VPM, midrange and three-point residuals are
actual-minus-expected shot value. In the public RAPM artifact, audit gaps also
record small differences among separately solved identities.

**SC**  
Second-chance scoring after the first offensive miss, not a rebound-rate
statistic.

**TOV value**  
The estimated point cost of first-chance turnovers, not merely turnover
percentage.

**TS**  
In the current product hierarchy, the shot/FT value branch. It is not literal
box-score true-shooting percentage.

**VPM**  
The repository's EV-target, time-decayed, prior-centered research evolution of
the public decomposition. The current displayed board is the EPM12 x2 research
artifact.

**Impact Monster**  
The reader-facing ten-factor breakdown of RAPM. Each factor type has an
offensive and defensive estimate; the pieces close back to their RAPM sides.

## Source-of-truth checklist

Before reusing this text as product help copy:

1. Check the deployed board's `candidate`, `three_factor_identity`, sign, units,
   and closure in
   [`vpm-manifest.json`](../../vpm-explorer/public/data/vpm-manifest.json).
2. Check the active public component list and alphas in
   [`run_decomp_rolling.py`](../scripts/run_decomp_rolling.py#L36-L55).
3. Check whether the TOV/SC allocation policy is still
   `full_possession_v1` in
   [`build_decomp_weighted_factors.py`](../scripts/build_decomp_weighted_factors.py#L129-L142).
4. Do not rename `oSC/dSC` to `oREB/dREB`.
5. Do not add research drilldowns to the ten-component VPM total.
