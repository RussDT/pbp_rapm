# rZTOV — mix-adjusted turnovers, priced per season

`scripts/build_rztov_explorer.py` builds rZTOV, a boxscore-style turnover stat that
charges a player against the turnover allowance their offensive load buys, and emits
the explorer payload for the reader page.

The sibling `research/from-impact-to-winning` site now carries the polished public
route `/skills/rztov`. Its checked-in five-season snapshot is refreshed from this
build with `bun run refresh:metrics` in that repo; the route must preserve the same
negative-good rate, two measured lanes, qualification rules, and research-only
boundary documented here.

```
rZTOV = (actual turnovers - allowance) / plays, per 100 offensive possessions
```

Negative is good. `allowance` is the load-weighted break-even — the sum of a
scoring-lane allowance bought by shot attempts and a passing-lane allowance bought
by potential assists, each priced by the joint window fit below — and a **play** is
a shot attempt, a potential assist, or a turnover:

```
plays = v_self + v_ast + v_potast + act      (act = sc_100 + bp_100)
```

The turnover sits inside its own denominator, which is the shape boxscore TOV%
already uses (`TOV / (FGA + 0.44 FTA + TOV)`), extended to cover passing. Without
the `+ act` term a turnover silently removes the chance it happened on from the
rate. `allowance` and `excess` are counts and are unaffected by the choice; only
the divisor moves, so the change is exactly recoverable and required no refit. The
volume-only sum is still carried as `involve` for the qualification gate.

## Involvement areas

Three areas, all regular season, all per 100 offensive possessions:

| Area | Basis | Meaning |
|------|-------|---------|
| `v_self` | true-shooting attempts | self-created attempts |
| `v_ast` | true-shooting attempts | assisted shots |
| `v_potast` | passes | potential assists created |

Actual turnovers are the repo raw-PBP split: `sc_100` (scoring / ball-handling) plus
`bp_100` (bad pass). Offensive fouls stay inside the scoring bucket — there is no
separate screener lane.

**Putback reassignment.** An FGA immediately following the same player's own
offensive rebound is not a self-created look, so it moves from `v_self` to `v_ast`.
The detector reads raw PBP in clock-first order
(`game_id`, `period`, `clock` desc, `event_num` asc) and caches counts to
`results/rztov_putbacks.csv`. Putbacks are counted FGA-basis while `v_self` is
TSA-basis, so the transfer is clamped at `min(v_pb, max(v_self, 0))`; without the
clamp a handful of low-minute bigs go negative.

## Fitting the break-evens: one joint solve, three targets

The three prices are fit **jointly** per trailing window against three count-basis
offensive ridge solves — scoring turnovers, bad-pass turnovers, and total turnovers
(`results/{tov,scoring_tov,badpass_tov}_{lo}_{hi}_rs_se_a1500_1500_results.csv`,
season effects on). One shared price vector, no intercepts, weighted by possessions:

```
y_sc  ~ k_sc  * (sc_100 - be_self*v_self - be_ast*v_ast)
y_bp  ~ k_bp  * (bp_100 - be_pot*v_potast)
y_tov ~ k_tot * (act    - be_self*v_self - be_ast*v_ast - be_pot*v_potast)
```

Solved by alternating least squares: given prices, each lane's kappa and residual
variance are closed-form; given kappas, the three row-blocks stack into one weighted
least-squares with inverse-residual-variance weights. Converges to `1e-10`,
initialized from lane-own fits. Only players with `>= 2000` window possessions and
all three solve values enter.

**Superseded (2026-08-15).** The previous production fit was a single unrestricted
five-term WLS of the total TOV solve on `[act, v_self, v_ast, v_potast]`. It was
replaced because the volumes are collinear — `corr(v_self, v_potast) = +0.71`
across fit players — so the free fit trades weight between the self and passing
prices: the bootstrap sd on the self price was `0.64` with a self-passing draw
correlation of `-0.71`. The joint fit pins the passing price against the bad-pass
solve, where only `v_potast` competes, and the bands tighten to self
`[10.60, 12.26]` (sd `0.51`), assisted `[3.87, 4.95]`, passing `[13.17, 14.35]`
(2022-26 window). A candidate that froze the scoring prices from the scoring-lane
fit alone was also rejected: nothing is frozen, all three prices move in one solve.

**Spillover identification.** On the 2022-26 window the scoring-lane-only anchors
are `9.68% / 5.38%` (self / assisted) and the bad-pass anchor is `14.27%`. The
joint self price lands at `11.36%`, `+1.7` above its own lane's anchor, because
ending a possession with your own attempt also relieves the team's passing
exposure — only the total solve sees that spillover. Players whose shot diet
actually shifted season-over-season show the same premium causally: `+2.7`
percentage points within-player, CI `[+1.0, +4.6]`. The unrestricted fit implied
`+6.1`, outside that interval — collinearity, not basketball.

Kappa notes (logic unchanged from the single-solve era):

- Count targets give each kappa a known physical value of `-1.00`: one extra
  turnover committed is one turnover not prevented. The ridge shrinks it.
- A break-even is a ratio and is **scale-invariant** to uniform shrinkage, which
  lands on kappa alone. Read the kappas as solve recovery. Fitted `|k_tot|` runs
  `0.76`–`0.90` across windows, `|k_sc|` `0.49`–`0.65`, `|k_bp|` `0.61`–`0.70`.

**Forward gate.** On the two available one-season-forward folds (2024→2025,
2025→2026) the joint prices are neutral versus the unrestricted production fit
(fwd r `-0.595 / -0.538` vs `-0.595 / -0.541`); sibling-level differences are
unresolvable at two folds, while the model family beats raw committed turnovers
decisively (P=1.00 both folds). The joint design is selected on identification
and price stability, not on a forward win. A convex hinge on assisted volume is
logged as a named candidate for untouched 2026-27 confirmation; it is not wired.

## Two season adjustments, not one

1. **Refit per trailing 5-season window.** Each season is scored on break-evens fit to
   the window ending that year. `2014`–`2017` predate the first full window and
   inherit the earliest fit; those years carry an `inherited` flag.
2. **Per-lane per-season centering.** The window fit calibrates allowance to actual
   across the whole window, so a single season inside it can sit slightly off. Each
   season's scoring-lane and passing-lane excesses are re-centered separately on
   their possession-weighted means, and each centre (`centre_sc`, `centre_bp` in the
   payload) is folded into that lane's allowance — so league turnovers saved sums to
   zero **inside both lanes** every year, not just in total. Centre the **excess**,
   not the ratio — a ratio centre reintroduces a Jensen artifact. Observed centres
   are small: max `|0.13|` (scoring) and `|0.07|` (passing) turnovers per 100.

The joint prices drift gently across the era (2014-18 window → 2022-26 window):
self-created `10.01% → 11.36%` rising, assisted `6.33% → 4.44%` falling, passing
`13.93% → 13.81%` — essentially era-invariant, holding `13.6`–`14.0` across all
nine windows. The old unrestricted path (`13.49 → 15.80`, `6.48 → 5.05`,
`11.02 → 9.26`) overstated the shot-price drift and manufactured a passing-price
decline; most of that motion was the collinearity wobble, not the era. The
remaining drift survives consistent putback treatment across all windows, which is
why the per-window refit is the design rather than one frozen set.

## Ledger display: two measured lanes

Every displayed row is measured. There are two lanes, and since the joint refit
each lane carries **its own fitted allowance**, not a display split of the total:

| Lane | Volume | Turnovers | Allowance |
|------|--------|-----------|-----------|
| Shot attempts | `v_self + v_ast` | `sc_100` | `be_self*v_self + be_ast*v_ast + centre_sc` |
| Potential assists | `v_potast` | `bp_100` | `be_pot*v_potast + centre_bp` |

Raw PBP never names which attempt type caused a scoring turnover, and the merged
lane does not need it to: the two attempt types share one observed turnover bucket
and one observed volume, so `sTOV% = sc_100 / (v_self + v_ast + sc_100)` is measured
end to end. The mix survives where it belongs, **inside the lane's allowance** — the
self-created share sets the price. Per-lane saved is real: scoring saved is priced
against the scoring-turnover solve and passing saved against the bad-pass solve,
and the two add to the total by construction.

Row algebra on the current build: `sc_excess + bp_excess = excess` and
`-(zTOV% / 100) * plays = saved` hold exactly (CSV rounding `7e-5` across all
5,370 season rows and 718 aggregate rows), and possession-weighted league saved is
zero inside both lanes every season (`< 4e-6`).

### Driver rows

Self-created and assisted shots appear beneath the shot-attempt lane as mix context.
The reader shows only each child's share of shot attempts and its fitted break-even.
It does not show muted estimates for `TOV`, `TOV%`, allowance, zTOV, or saved value:
play-by-play does not identify which kind of shot attempt produced a scoring turnover,
so those measured outcomes stay on the parent lane.

**Superseded (2026-08-10).** An earlier build split `sc_100` across two attempt rows
proportional to `alloc_self * v_self : alloc_ast * v_ast`, from a cross-sectional
regression of `sc_100` on the two volumes. That fit put `alloc_ast = 8.37%` above
`alloc_self = 5.89%` in every window (2014-18: `6.10` vs `5.28`) - an assisted attempt
priced as riskier per attempt than a self-created one, against within-player marginals
of `7.7%` and `2.8%` - because it absorbed selection and collinearity with the dropped
intercept rather than attempt risk. It also forced the ratio between the two displayed
`TOV%` cells to a league constant (`1.4215` on all 710 aggregate rows). The parent lane
removes that split from the measured figures, and the reader no longer displays an
estimated child-row outcome split. `alloc_*` survives in `fit_window()` because the
curve panel uses `alloc_base` / `alloc_ast`; it is no longer carried in the explorer
payload.

Saved is shown in both units side by side — a rate (`Saved/100`) for comparing
players, and a count (`Saved`) for what the player actually did on their own workload.
The counts are whole turnovers rounded by largest remainder against the exact total,
so the rows still add to the total on screen.

## Reader header

The selected-player header is deliberately terse: search, player name, one workload
tree, and the three outcome figures (`rZTOV%`, `TOV saved /100`, and `Total saved`).
Season span, possession count, rank, and the prose calculation do not sit beneath the
player name; those details already live in the scope controls and detailed tables.

The workload tree keeps `Plays /100` as the parent and shows self-created attempts,
assisted attempts, and potential assists as its three visible workload sources. Because
the rZTOV denominator also includes actual turnovers, the three source values sum to
`involve`, while `involve + TOV = plays`; the formula remains in the metric guide rather
than adding another annotation to the compact header. The display must not imply that
turnovers were assigned to a self-created or assisted child.

## Reader comparison layout

The Register is the full-width comparison surface. It keeps the identity column pinned
while the dense metric families scroll, and orders the core fields as impact, workload,
rate, and play shape. `TOV allowed` is the fitted allowance per 100 possessions;
`TOV%` is actual turnovers divided by plays; `rZTOV%` is the actual-minus-allowance
rate on that same denominator. Keeping all three together shows the raw result, the
workload-adjusted expectation, and the gap without collapsing them into one number.

The three play-shape columns remain self-created attempts, assisted shots, and
potential assists, all per 100 offensive possessions. Beneath the Register, the League
map and selected-player season table form a second comparison row: population context
on the left and within-player history on the right. The season table repeats the core
allowance and rate fields so a player's changes can be read with the same semantics as
the cross-sectional Register.

## Aggregate windows

The multi-season board is a **possession-weighted roll-up of separately priced season
rows** — volumes, per-area allowances, and the season centre are all weighted — not a
re-application of the latest break-evens to pooled volumes. A window total therefore
equals the sum of the seasons inside it.

## Qualification

`SEASON_MIN_POSS = 500` rows are published; `qualified = 1` requires `>= 1000`
offensive possessions **and** `>= 10` involvement (the volume-only sum, not plays).
rZTOV divides by plays, so a player with almost no shot or passing load produces a
huge rate off a trivial raw excess. Keep those rows, but do not rank them.

## Validation (2022-26 board, n=568)

| Check | Value |
|-------|-------|
| corr with TOV-value RAPM, raw TOV/100 | `-0.071` |
| ... per-attempt rate | `-0.419` |
| ... rZTOV excess | `-0.761` |
| year-over-year r (n=2,934) | `+0.802` |
| R2w: actual TOV/100 alone | `0.001` |
| R2w: volumes alone | `0.330` |
| R2w: rZTOV x plays | `0.680` |
| R2w: unrestricted, 5 free terms | `0.743` |
| lane fits: R2w vs scoring / bad-pass solves | `0.483` / `0.240` |

The joint prices give up a little total fit versus the unrestricted five-term
regression (`0.680` vs `0.743`; excess corr `-0.761` vs `-0.784` under the old
fit) and buy two measured lanes with stable, identified prices in exchange — the
free fit obtains its extra fit by letting correlated volumes trade weight.

Within-player marginals from adjacent-season deltas (total turnovers): self-created
`7.7%`, assisted `2.8%`, potential assist `9.6%`. The per-self-created-attempt cost
is roughly flat across creation load (`6.4%` at 3-6 attempts, `6.0%` at 20+), so
one linear rate per area holds; what does not hold is one rate across areas. The
lane-anchor-versus-joint-price comparison in the fitting section is the sharper
within-player check on the self-price spillover.

Per-season qualified sd runs `2.48`–`2.91` across 2014-2026, so the scale is stable
enough to compare years directly.

## Outside reference columns

The board carries two window-level columns from other solves, aggregate scope only:

- **TOV RAPM** — the count-basis offensive solve rZTOV is fit against, in turnovers
  prevented per 100.
- **oTOV** — the published point-valued offensive turnover factor, read from
  `master_results/weighted_factors_22_26_all_rb_se_a2000_4000.csv` (all 718 board
  players covered), in points per 100. Different solve: regular season plus playoffs,
  rubberband on, its own alphas.

They are the same turnovers priced two ways and correlate `0.992`, so they are a
familiar anchor, not an independent check. Against rZTOV itself, oTOV correlates
`-0.800` (opposite sign because negative rZTOV is good).

## Outputs

| Path | Contents |
|------|----------|
| `results/rztov_seasons_2014_2026.csv` | the stat — one row per player-season; carries `involve`, `plays`, the per-lane columns `sc_allow`, `bp_allow`, `sc_excess`, `bp_excess` (lane allows include the lane centres), and the season's lane point prices `sc_point_price` / `bp_point_price` with `tov_points_saved_100 = -(sc_excess*sc_point_price + bp_excess*bp_point_price)` |
| `results/rztov_research_22_26.csv` | aggregate research surface for the board window, same lane columns |
| `results/rztov_putbacks.csv` | cached putback counts (rebuild with `--rebuild-putbacks`) |
| `<out-dir>/rztov_explorer_data.json` | explorer payload; per-season `be` blocks carry `self/ast/potast`, the three kappas, and `centre_sc` / `centre_bp` |

All three CSVs are generated artifacts and gitignored under `nba_pipeline/results/`.

The separate research-only [NBA box-estimated rZTOV backcast](./NBA_RZTOV_BOX_BACKCAST.md)
estimates the three workload inputs from an all-era PBP/box contract and fits
era-specific rolling prices from historical TOV, SCORING_TOV, and BADPASS_TOV
count-basis RAPM targets. Its labeled `box_estimated_rztov` series covers
1997-2013; it does not overwrite these observed 2014-2026 rows.

```bash
cd nba_pipeline/scripts
../../.venv/bin/python build_rztov_explorer.py --out-dir <dir> --html ../docs/rztov-explorer.html
```

`--html` rewrites the page's single `const DATA = ...;` line in place with the new
payload (it refuses if the page has zero or several such lines). The build is
deterministic: a rerun on unchanged inputs reproduces the CSV, the page template and
the payload byte for byte.

## Points conversion

An excess turnover is priced in points **per lane and per trailing window**, not with
one flat number. For each five-season window the builder possession-weights every
player's lane excess, flips the sign so positive means saved, and fits a
through-origin, possession-weighted slope against the matching point-valued
net-continuation turnover RAPM. The target is a **near-unpenalized** solve of the same
child lanes,
`results/tov_price_targets/alt_{scoring,badpass}_tov_net_value_{window}_all_rb_se_a20_40`,
built by `scripts/build_tov_price_targets.py` (rubberband, season effects, RS+PS,
offense/defense alpha `20/40`). This uses the same `fit_turnover_point_prices` helper as
the [historical scoring board](./HISTORICAL_SCORING_SECURITY.md), so the two surfaces
agree on every observed season. A season takes the price of the window that ends in
that season; 2014-2017 use the first complete observed window (2014-2018). The
payload's `be[year]` blocks carry `price_sc` / `price_bp` with 90% bootstrap
intervals and the fit R2.

| Window | Scoring-TOV price | 90% interval | Bad-pass-TOV price | 90% interval |
| --- | ---: | ---: | ---: | ---: |
| 2014-2018 | 1.041 | 0.96-1.11 | 1.015 | 0.88-1.19 |
| 2018-2022 | 0.838 | 0.77-0.91 | 0.973 | 0.80-1.11 |
| 2022-2026 | 0.902 | 0.85-0.95 | 1.020 | 0.88-1.19 |

Across the nine observed windows scoring prices run `0.84`-`1.04` and bad-pass prices
`0.94`-`1.04`. The two lanes are not separable: every bad-pass interval overlaps its
scoring interval. These are empirical RAPM bridges, not a claim that every turnover
carries a fixed event value.

**Correction, 2026-10-19: why the target must be unpenalized.** Through 2026-10-18 the
prices regressed on the production alpha-`2000/4000` components and read `0.465`-`0.589`
(scoring) and `0.684`-`0.767` (bad pass). Those numbers were mostly ridge shrinkage. A
slope fitted with RAPM as the dependent variable is biased toward zero by exactly the
shrinkage applied to that RAPM. Noise in the dependent variable does not bias a slope;
shrinkage does. Two checks agree with the corrected prices:

- **Alpha path.** On 2022-2026 the scoring slope rises from `0.535` at `2000/4000` and
  plateaus by `20/40`; alpha `1/2` moves it by only 1-2%.
- **Direct lineup IV.** Regress team first-chance turnovers per 100 on the summed
  on-court box excess of the five offensive players, with the leave-season-out excess
  as the instrument, season-by-type demeaning, and game clusters. It gives
  `0.88`-`1.01` scoring and `0.96`-`0.97` bad pass across 2014-18, 2018-22, and
  2022-26.

The earlier story that bad passes price higher because they become live-ball steals,
and the "resurfacing / attenuation / workload" explanation for the gap, were both
written about the shrunk numbers. They are withdrawn.

**What still separates a price from a kept possession.** The window's net first-chance
value of a kept possession is about `1.25`-`1.31`. The remaining gap has two sources:

- **First-chance coverage.** The target counts only first-chance turnovers, which are
  about 96% of box scoring turnovers and 91% of box bad passes.
- **A genuine shortfall of about 20%.** Some of the avoided turnover reappears as a
  worse shot or as another player's turnover. Regressor reliability is `0.87`-`0.95`,
  so attenuation is minor.

Second-chance turnovers stay outside this price, as they are outside the
first-chance target.

The [Lineup Turnover Board](./LINEUP_TOV_BOARD.md) shows the lineup reading
directly. It solves two-season TD550 on-court turnovers, split into own,
teammates, and team, at the full kept-possession price. The scoring-impact
page displays it beside the box values; it never replaces these prices.

## Explorer page

[`rztov-explorer.html`](./rztov-explorer.html) is a single self-contained page. The payload
sits on one line, `const DATA = {...};`. To refresh it, replace that line with the new
`rztov_explorer_data.json` and leave the rest of the page untouched. The layout, rebuilt
2026-09-25, is organized around one player read:

- **Top bar.** It holds the player search (`/` focuses it; suggestions show career span,
  possessions and rZTOV). It also has a segmented control for aggregate vs season scope,
  the qualification floor, a "How it works" jump to the method section, and a theme
  toggle (auto/light/dark). A dismissible three-step primer follows the top bar.
- **Player column.**
  - The verdict card shows the rZTOV figure with its rank. A plain-language sentence
    gives committed vs allowed, turnovers saved, and the points conversion. Below that
    is a three-by-two stat grid (two columns under 1080px) whose last card, **Total
    points**, is points saved per 100 times possessions / 100 for the selected scope.
    A histogram of the qualified distribution follows, with the player marked.
  - Bullet charts compare the allowance against committed turnovers for each lane and
    the total.
  - The workload-mix bar shows the per-driver break-evens. The full allowance ledger sits
    in a collapsible section.
  - The season chart has a dashed aggregate reference, and the season table follows it.
    Clicking a bar or row rescopes the whole page.
- **League panel.** This is a sticky leaderboard with the unfiltered rank, a mini
  diverging rZTOV bar and a name filter. Column sets are Summary, Workload, Rates,
  Reference (aggregate only) and All. It also has a league map tab, where the circles
  animate between scopes. Arrow keys move the selection and `j`/`k` step through the
  current list.
- **Method and evidence.** This section holds the formula, the fitted parameters, the
  break-even drift, the predictor ladder and the per-volume curve, plus the method prose.
- **State.** Player, scope, floor, sort, view, column set and filter persist in the URL
  hash. The theme and the primer dismissal persist in `localStorage`.

## Publishing constraints

- The tracking vendor supplying the involvement volumes **must not be named** anywhere
  in published output. The explorer payload and template are checked for this.
- Screen/pick data cannot be published. Offensive fouls stay folded into the scoring
  turnover bucket; there is no screener area.
- The points figures use the per-lane, per-window net-continuation prices above. Both
  the point-valued solve and the count solve are ridge-shrunk, and not identically, so
  the conversion is a calibrated bridge rather than an exact event price.
- Before any publish, run
  `nba_pipeline/scripts/audit_scoring_rztov_reconciliation.py`. It re-evaluates the
  page's own `seasonRow`/`finish` arithmetic on the embedded payload and checks the
  result against this CSV, then checks the CSV against the historical scoring board
  (see the [integration handoff](./SCORING_RZTOV_DATABALLR_HANDOFF.md)).
