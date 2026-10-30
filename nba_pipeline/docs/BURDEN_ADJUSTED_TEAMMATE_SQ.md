# Burden-adjusted teammate SQ research diagnostic

Owner request September 18, 2026: investigate playmaking beyond burden absorption,
using playtype difficulty and shooting share as a baseline. This is a descriptive
peer-baseline residual, not an identified causal division between absorption and
creation. It does not replace production zTS, DECOMP, VPM, or the six-part board.

## Inputs and definition

`build_burden_adjusted_teammate_sq.py` uses the existing 2022–2026 RS six-part
opportunity sample and the frozen `td_overall_playtype_mix/attempt_panel_2019_2026.parquet`.
The latter is a prior-season-priced seven-family PBP difficulty proxy (including
transition), **not production zTS or its exact difficulty input**. Pre-2025
transition labels are retrospective. No current shot result assigns difficulty.

Per season, derive own difficulty over that player's FGAs, share over resolved
shooting opportunities, and `(share - .2) * difficulty`. The pooled features
average these annual quantities using teammate-opportunity exposure; the pooled
interaction is not computed from pooled marginal means. Thus changing annual
roles remains represented. Role FGAs come from the broader resolved PBP cache;
the target and exposure counts use the exact six-part board sample. FT-only
opportunities contribute to share but have no invented playtype difficulty.

The teammate talent proxy is prior-season player accuracy excess relative to
league within each broad bucket, with 100 attempts of shrinkage per bucket,
then averaged over a fixed prior league bucket mix. Missing players use neutral
excess and coverage is exported. Average the recipient proxy over FGAs taken
by teammates when the focal player is a nonshooter. This is a coarse proxy,
not purified talent: PBP categories leave context unresolved, and absent buckets
are effectively neutral. Current teammate shot-type mix is not a baseline input,
since creating that mix may be the signal sought.

The baseline predicts **conditional teammate SQ/rim**, using four standardized
features: share, own difficulty, their interaction and teammate talent proxy.
It is estimated separately for pooled and annual boards. Weight by teammate
opportunities; require >=500 teammate opportunities and >=100 own role FGAs.
Each player's prediction is made by a model excluding that player: five fixed
shuffled player folds; inside each training fold select ridge alpha1/10/100/1000
with three folds and the paired one-SE stronger-penalty rule. Low-exposure or
missing-feature rows stay missing, never zero.

`teammate_surplus_conditional = teammate_sq_conditional - burden_baseline_conditional`

Usage-weight both new terms by `(1 - shot_share)` to produce the contribution
view. They sum exactly to the existing teammate-SQ contribution; neither is an
additional component of total offense. All original six-part values stay unchanged.
Do not label the baseline itself measured burden-absorption value: it includes
peer expectations associated with role and talent, including possible real creation.

## Results and limitations

713 eligible pooled players. Held-out-player weighted R-squared:

| Baseline | R² | RMSE |
|---|---:|---:|
| Share only | .3690 | 1.1242 |
| Share + talent proxy | .3797 | 1.1147 |
| Share + difficulty + interaction | .4241 | 1.0740 |
| Full | .4308 | 1.0678 |

These predict estimated coefficients from a common joint RAPM fit. They are not
independent future-game tests: player targets share lineup-estimation covariance.
Five-year exposure and target shrinkage also affect the apparent explanatory fit.
No uncertainty claim or causal inference follows from the R² increment.

Three independent random partitions of games, stratified by season, rebuild both
SQ estimates and role features on each half. Half-sample SQ O-own/team/defense
penalties are 15/500/2500 to preserve penalty per exposure. Each half independently
cross-fits its peer baseline. On 607–618 players eligible in both halves:

- Raw conditional teammate SQ rank correlation: .465–.491.
- Surplus rank correlation: .292–.317.
- Surplus sign agreement: 58.1–60.5%.

Lag prices and prior-season talent inputs are frozen/shared across half splits;
complete-season FT expectations and frozen v2 inputs are also retained. This is
conditional reliability, not date-safe prospective validation. The half-game
reliability is weak enough that player ordering needs caution.

Fixed baseline alphas1/10/100 produce rank correlations .997–.998 with the selected
surplus; alpha1000 yields .909. The tested nearby baseline penalties are less of a
problem than sampling instability. `analyze_burden_teammate_diagnostics.py` reproduces
these ablations and sensitivity results.

Descriptive creation checks use existing 2023–2025 feature-season panels, separately
by season, never fit as baseline inputs. Potential-assist rank correlations fall
from .317–.371 raw to .140–.182 adjusted; rim-assist correlations fall from
.395–.425 to .222–.255. Positive residual relationships do not establish improved
playmaking isolation. Adjustment may remove genuine creation as well as role overlap.
No 2027 outcomes are touched and nothing is admitted as a predictive prior.

Illustrative pooled conditional values (points per100 teammate opportunities):

| Player | Teammate SQ | Baseline | Surplus |
|---|---:|---:|---:|
| Nikola Jokic | 6.294 | .998 | 5.296 |
| Stephen Curry | 4.612 | .955 | 3.657 |
| Shai Gilgeous-Alexander | 2.078 | 2.292 | -.214 |
| Luka Doncic | 4.296 | 2.535 | 1.761 |
| Trae Young | 3.813 | 1.591 | 2.222 |

These are diagnostics, not newly validated playmaking rankings. In particular,
a near-zero surplus does not imply no playmaking: typical creation for the role
may already be in the baseline.

## Reproduce and publish

Run `build_burden_adjusted_teammate_sq.py`, then
`analyze_burden_teammate_diagnostics.py`, then `publish_v2_six_site_assets.py`.
Use OPENBLAS_NUM_THREADS=4 and OMP_NUM_THREADS=4 for the lineup solves.
Generated files remain ignored in `results/burden_teammate_sq_2022_2026_rs/`.
The existing public Site includes both new columns, their conditional forms in
CSV, explicit eligibility blanks and downloadable reliability/creation receipts.
No new scoring fit replaces the original board.

Tests enforce exact residual/contribution identities, exclusion of a player's
own target from their baseline training and missingness below eligibility floors.
