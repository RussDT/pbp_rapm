# Prior-Informed Time-Decay TOV RAPM (research lane)

**Status (2026-08-15): research prototype with a positive frozen result.**
**Updated 2026-08-16:** the EPM predictive-ledger extension (below) shows a
fully forward-looking prior built *without* tracking data matches or edges
the observed mix prior once foul-drawn attempts are priced into the
predicted volume; blending the observed tracking excess back in adds nothing
detectable. **Also 2026-08-16:** the within-season refresh test and the
as-of daily board (`research_predictive_ztov_daily.py`, last section) show
that refreshing the possession solve inside a season is worth ~90e-6 —
about four times the whole prior's annual gain — while swapping the
season-start EPM snapshot for the latest one at full weight ties the
season-start prior with coverage carry-forward. **Updated 2026-08-18:**
`evaluate --diagnose` shows why the swap ties: the fresh snapshot does carry
information the refreshed solve lacks, but at full weight it is double
counted against the live rows (MSE-optimal weight on the swap 0.50
[0.36,0.64]); the pre-specified half blend beats the season-start prior in
8/8 seasons (−3.2e-6 pooled) and is now the board's in-season default
(`--fresh-weight 0.5`). Nothing here feeds production solves, the
published rZTOV surface, or VPM inputs. Everything through 2025-26 is
development-era evidence; the untouched confirmation season for any
promotion is 2026-27.

## Question

The rZTOV explorer (`RZTOV.md`) is descriptive: it prices what a player's
role volumes say their turnovers *should* be, using the joint two-lane
break-even model. This lane asks the predictive follow-up — does that priced
allowance signal carry information about **future possessions** that the
possession history alone does not? Concretely: insert a mix-adjusted
allowance prior inside a rolling time-decay TOV RAPM solve and test it
against plain and persistence-prior baselines on frozen future seasons.

## Protocol

Script: `nba_pipeline/scripts/research_prior_informed_tov_rapm.py`
(single run, ~25 min; writes `results/tov_prior_rapm_folds.csv`).

- **Data.** `processed/TOV{yy}.parquet`, 2014–2026 regular season, rows with
  five known players a side: 2,969,221 possessions, 1,706 players.
- **Solve.** One joint ridge over offense/defense player blocks,
  `y_100 = mu_t + sum(off_i) + sum(def_j)` (positive off = commits more,
  positive def = forces more), per-season-centered target (season effects),
  time-decay weights `0.5^(days_before_asof / half_life)`, prior-centered
  ridge `(G + aI) b = r + a p`. The level term `mu` is the last train
  season's mean rate, shared by every candidate.
- **Folds.** Train the trailing 5 seasons ending `y`, score every possession
  of season `y+1`. Dev selection on scored seasons 2019–2024; frozen report
  on 2025 and 2026. Scored seasons start at 2019 because 2014–2017 rZTOV
  prices are inherited from the 2014-18 window fit (leak risk).
- **Prior leak safety.** Season `t` excess is priced by the joint window fit
  ending `t` (walk-forward by construction of the explorer build); `|kappa|`
  comes from the window ending the as-of year; the trailing excess average
  is possession- and decay-weighted with the same half-life as the solve.
- **Candidates.**

  | Name | Definition |
  |---|---|
  | C0-box | TD-weighted trailing committed TOV/100, centered; defense 0 |
  | C1-1y | single-season ridge |
  | C2-td5 | trailing-5 TD ridge, no prior |
  | C3-mix | C2 + offense-side prior `p_i = s * |kappa_y| * trailing rZTOV excess_i` |
  | C4-state | C2 + persistence prior (same-recipe solve as-of `y-1`, both sides) |
  | C5-combo | C2 + mix and state priors jointly, grid over both scales |

  C3–C5 inherit C2's selected alpha/half-life rather than re-tuning, which
  biases *against* the priors.
- **Scoring.** Possession MSE in 0/1 units; deltas reported in 1e-6 units.
  Paired game-clustered bootstrap (per-game squared-error sums, 2000 draws,
  seed 11), 5–95% intervals; pooled rows resample season-prefixed games.

## Selection (dev folds 2019–2024)

- C2 selected `alpha=3000`, `half_life=1500` days (dev mean MSE 0.123904).
  The full 16-cell grid spans only ~38e-6 — a compressed ceiling; turnover
  prediction at the possession level is mostly irreducible.
- C1 selected `alpha=3000` (0.123959).
- C3 selected mix scale 0.5 (0.123893). A dev-only extension over scales
  {0.125, 0.25, 0.375, 0.5, 0.75, 1.0} (`--dev-mix-scales`) traces a shallow
  bowl with its bottom at 0.5–0.75 and both neighbors worse, so 0.5 is an
  interior optimum, not a grid-edge artifact.
- C4 selected state scale 0.5 (0.123904 — no dev gain over C2).
- C5 selected `mix=0.5, state=0.0` (0.123893) — the combo **degenerates to
  C3**. Given the mix prior, persistence adds nothing.
- Mix-prior coverage grows from 665 players (as-of 2018) to 1063 (as-of
  2024); uncovered players simply shrink to zero as usual.

## Frozen results (2025 and 2026, never used for selection)

Per-fold possession MSE:

| Candidate | 2025 | 2026 |
|---|---|---|
| C0-box | 0.126896 | 0.127882 |
| C1-1y | 0.126391 | 0.127294 |
| C2-td5 | 0.126310 | 0.127250 |
| C3-mix | **0.126289** | **0.127233** |
| C4-state | 0.126304 | 0.127250 |
| C5-combo | 0.126289 | 0.127233 |

Paired deltas, e-6 (negative = first candidate better), game-clustered
bootstrap 5–95%:

| Pair | 2025 | 2026 | Pooled | P(better), pooled |
|---|---|---|---|---|
| C3-mix vs C2-td5 | −21.46 [−27.64,−15.21] | −16.30 [−22.60,−9.66] | **−18.87 [−23.27,−14.30]** | 1.00 |
| C4-state vs C2-td5 | −6.01 [−11.79,−0.10] | +0.26 [−5.12,+5.58] | −2.86 [−6.63,+1.13] | 0.89 |
| C5-combo vs C4-state | −15.45 | −16.56 | −16.01 [−20.72,−11.40] | 1.00 |
| C2-td5 vs C1-1y | −81.04 | −44.16 | −62.54 [−77.26,−46.76] | 1.00 |
| C2-td5 vs C0-box | −585.85 | −632.50 | −609.25 [−652.60,−566.09] | 1.00 |

## Reading

- **The mix prior transfers to predictive value.** It beats the plain TD
  solve with the full interval clear of zero on both frozen folds
  separately and pooled. The pooled gain (−18.9e-6) is roughly 30% of the
  entire TD5-over-1Y improvement (−62.5e-6) — large for this compressed
  ceiling.
- **Why it works: external information, not shrinkage mechanics.** The
  persistence prior (C4) recycles what the trailing possession window
  already contains and is worth ~nothing (2026 fold exactly neutral). The
  mix prior imports something the possession rows never see — tracking-based
  role volumes priced by the joint lane model — and the combo grid's
  `state=0` selection says it strictly subsumes persistence.
- **Player-level behavior (as-of-2025 maps,
  `results/tov_prior_rapm_maps_2026.csv`).** Established players barely
  move (C2↔C3 offense corr 0.95); the movement concentrates in thin-sample
  players where ridge shrinkage leaves information on the table (delta sd
  0.17 TOV/100, max |delta| 0.86, posterior-vs-prior corr 0.93 on the 756
  covered of 991 active offense players). Down-movers are low-usage guards
  whose role says their near-zero coefficient should be negative (Cam
  Spencer, Jeff Dowtin Jr., Skylar Mays); up-movers are turnover-prone bigs
  (Dwight Howard, DeMarcus Cousins, Adem Bona, Charles Bassey).
- **Baseline ordering confirms the design.** TD5 ≫ 1Y ≫ raw box, all
  P=1.00, with the box gap (~609e-6) an order of magnitude above everything
  else — opponent/lineup adjustment and multi-season pooling do most of the
  work, the prior adds the last increment.

## EPM predictive-ledger prior (Product C, 2026-08-16)

Script: `nba_pipeline/scripts/research_predictive_ztov_prior.py` (~15 min;
writes `results/tov_predictive_ztov_folds.csv`). Question: can the rZTOV
excess be reconstructed from *predicted* skills instead of observed
tracking, giving a prior that needs no tracking data and covers players the
tracking history misses?

### Construction

Per as-of year `y` (scoring season `y+1`), from the latest date-gated EPM
snapshot requested strictly before the scored season's first game
(`cache/epm`, manifest contract: predictions requested for date D exclude
games on D; 2023+ folds pick pre-season October snapshots):

- `act_hat = p_tov_100`.
- Attempt volume on the **tracking TSA basis**: the observed `v_self`/`v_ast`
  come from the `shot_fixed_*` export, which counts fouled non-FGA attempts
  (TSA ≈ FGA + 0.36·FTA; +1.41/100 or +8.5% on average, sd 1.01, and the
  FGA-only shortfall correlates +0.49 with FTA rate). So
  `v_hat = a + b·(p_fg2a_100 + p_fg3a_100) + c·p_fta_100`, a walk-forward
  possession-weighted map fit by `nba_id` on season-end EPM snapshots vs
  observed volume, seasons ≤ `y` (as-of 2025: `0.82 + 0.921·fga +
  0.433·fta`, n=4,952; stable across as-of years, close to the classic
  0.44 TSA weight). An FGA-only volume shorts foul-drawers and biases their
  predicted excess upward — the first Product C run had this bug; its
  numbers are superseded below.
- Self/assisted split: trailing decayed volume-weighted assisted share per
  player (same 1500d half-life), lightly shrunk to league (~0.60); league
  share for players with no history. Season-level assisted share is highly
  persistent (YoY r 0.93), so an assumed split is safe.
- Potential assists: walk-forward season-level linear map AST/100 →
  potential AST/100 fit on the tracking passing export, seasons ≤ `y`,
  ≥500 passes (R² 0.951; slope ~2.0), applied to `p_ast_100`.
- Lane prices and centres recovered exactly (lstsq residual <5e-3 pp) from
  the year-`y` rows of `results/rztov_seasons_2014_2026.csv` — the same
  walk-forward joint window prices the observed prior uses.
- `excess_hat = act_hat − allowance_hat`, recentred over the EPM-covered
  set weighted by `p_mp_48`; prior `p_i = |kappa_y| × excess_hat`.

EPM snapshots see playoffs of season `y` (legitimate pre-scored-season
information the RS-only tracking excess lacks). Only date-safe `p_*` skills
are used, per the sanctioned EPM12 pattern; `eWINS` is forbidden.

Snapshot-cache note: the tables below were computed when the latest cached
snapshot before the 2020-21 tip (2020-12-22) was `2020-06-30`. The daily
extension later fetched monthly in-season snapshots including
`2020-12-01`, so the unchanged rule (latest requested date strictly before
the first scored game) now picks `2020-12-01` for the 2021 fold; a harness
rerun will differ from the printed 2021 row by that snapshot swap only.

### Candidates and dev selection (2019-2024; base fixed at α=3000, h=1500)

| Candidate | Definition | Dev mean MSE |
|---|---|---|
| C2-td5 | no prior (reference) | 0.123904 |
| C3-mix | observed mix prior, frozen scale 0.5 | 0.123893 |
| C6-pz | pure predictive-ledger prior, **scale 1.0** | **0.123887** |
| C7-ext | C3 (0.5) where tracking-covered + pz where EPM-only, ext scale 1.25 | 0.123889 |
| C8-blend | on both-covered `λ·mix + (1−λ)·pz`; λ=0.25, scale 0.75 | **0.123887** |

Grid notes (scale grids {0.25…1.25}): C6 ties 0.75/1.0 with 1.25 worse —
interior optimum; C7's extension scale sits on the grid edge (dominated
candidate, logged, not extended); C8 ties across λ∈{0.25,0.5} ×
scale∈{0.75,1.0}. C6 and C8 tie on dev at display precision.

### Frozen folds (descriptive — see caveat)

| Candidate | 2025 | 2026 |
|---|---|---|
| C2-td5 | 0.126310 | 0.127250 |
| C3-mix | 0.126289 | 0.127233 |
| C6-pz | **0.126281** | 0.127233 |
| C7-ext | 0.126285 | 0.127234 |
| C8-blend | 0.126283 | **0.127230** |

Paired deltas, e-6, game-clustered bootstrap 5–95%:

| Pair | 2025 | 2026 | Pooled | P, pooled |
|---|---|---|---|---|
| C6-pz vs C2-td5 | −29.59 | −17.15 | **−23.35 [−31.02,−15.50]** | 1.00 |
| C6-pz vs C3-mix | −8.13 [−14.67,−1.46] | −0.85 [−7.47,+5.48] | −4.48 [−8.97,+0.22] | 0.94 |
| C7-ext vs C3-mix | −3.58 | +0.86 | −1.35 [−4.86,+2.26] | 0.75 |
| C8-blend vs C2-td5 | −27.85 | −19.68 | **−23.75 [−30.03,−17.37]** | 1.00 |
| C8-blend vs C3-mix | −6.39 [−10.75,−1.83] | −3.38 [−7.51,+0.70] | **−4.88 [−7.97,−1.91]** | 1.00 |
| C8-blend vs C6-pz | +1.74 [−1.10,+4.55] | −2.53 [−5.49,+0.36] | −0.40 [−2.43,+1.66] | 0.61 |

Caveat: the frozen folds were opened on 2026-08-15 for the C3 headline and
have now been reported twice more in this lane (the FGA-only first run,
superseded, and this corrected run — the correction came from auditing the
volume basis, not from frozen performance). Selection was dev-only
throughout, but multiplicity is accruing; treat everything here as
descriptive and the 2026-27 season as the only clean gate.

### Reading

- **The predictive ledger matches or edges the tracking prior.** Pure
  predictive (C6) beats plain TD by more than the tracking prior does
  (−23.4 vs −18.9 pooled) and is nominally better than C3 (−4.48 pooled;
  interval grazes zero, 2025 alone clears). On both-covered players the
  two priors correlate +0.89 to +0.92 per fold. The entire tracking-prior
  gain is reachable with zero tracking data — EPM skills, a season-level
  AST→potential-assist map, and an FGA/FTA→TSA volume map suffice.
- **Blending the observed excess back in adds nothing detectable.** C8 vs
  C6 is −0.40 pooled with a tight interval straddling zero and the folds
  split; C8's edge over C3 is the predictive component doing the work. C7
  (observed where covered, predictive only as fill) is dominated — on
  covered players the predictive version at scale 1.0 beats the observed
  version at 0.5.
- **The solve trusts the predictive version harder.** The pz prior is
  mildly compressed (regression slope ~0.87 on mix, sd ratio 0.89), yet
  dev selects scale 1.0 vs the mix prior's 0.5 — twice the prior weight,
  consistent with EPM's own stabilization making the ledger less noisy
  than raw trailing observed excess.
- **The FTA correction did what it should.** Before it, `pz − mix`
  correlated with FTA rate (foul-drawers looked riskier); after, the
  correlation is −0.12 and heavy foul-drawers moved safer (Giannis, 14.5
  FTA/100: +0.01 → −0.68; Embiid −0.32 in prior units).
- **Surviving divergences** (as-of 2025, both-covered): predictive says
  riskier than tracking for Westbrook (+0.83), Draymond (+0.58), CP3
  (+0.55) — low-FTA players whose EPM `p_tov_100` sits above what their
  role volumes excuse (Westbrook 5.2/100) — and safer for Zubac (−0.71),
  D'Angelo Russell (−0.65), Mitchell (−0.62), DeRozan (−0.61). EPM-only
  coverage (~60–115 players per fold) catches rookies and fringe players
  with no tracking history.

## Within-season refresh and the daily board (2026-08-16)

Script: `nba_pipeline/scripts/research_predictive_ztov_daily.py`, two
subcommands. `evaluate` is the within-season rolling test (writes
`results/tov_predictive_ztov_daily_eval.csv` and
`..._daily_windows.csv`); `board` is the as-of daily job (writes
`results/tov_predictive_ztov_board_{D}.csv` + `.manifest.json` and a
`_latest` copy). Question: the annual harness scores one static map per
season — the daily version can refresh two things, the possession solve
(time-decay weights as of today plus the live season's rows) and the EPM
snapshot behind the prior. Which of the two carries the value?

### Design (pre-declared; no selection)

- Scored seasons `S = 2019..2026`, structural year `y = S−1`: `|kappa_y|`,
  year-`y` lane prices/centres, walk-forward volume and AST→potential-assist
  maps through `y`, trailing assisted share as-of `y` — everything the
  annual harness freezes at `y`. Refresh dates `D_k` = 1st of Dec, Jan,
  Feb, Mar, Apr. The forecast made at `D_k` trains on every possession
  before `D_k` (trailing five seasons plus season `S` to date), weights
  `0.5^((D_k − day)/1500)` so `alpha=3000` keeps its meaning relative to
  the as-of date, complete seasons centred on their own full-season mean and
  the live season on its running mean (≥5,000 rows, else the last complete
  season's mean); it scores `[D_k, D_{k+1})`, the last window running to
  the RS end. Every possession from Dec 1 to the RS end is scored exactly
  once and every candidate shares the same live-season level, so the
  comparison isolates the player maps.
- Active columns at `D_k`: trailing-active ∪ seen in season `S` before
  `D_k`; unseen players sit at 0 (harness convention). The incremental
  algebra is exact: one trailing gram per season plus one gram per window,
  rescaled by the scalar decay from the trailing as-of date to `D_k`.
- Candidates, all sharing `alpha=3000`, `half_life=1500`, pz scale 1.0,
  mix scale 0.5:

  | Name | Definition |
  |---|---|
  | A0-annual-plain | harness C2 map, static all season |
  | A1-annual-pz | harness C6 map (season-start snapshot), static |
  | W0-refresh-plain | solve refreshed at each `D_k`, no prior |
  | W1-refresh-stale | refreshed solve + season-start snapshot prior |
  | W2-refresh-fresh | refreshed solve + latest snapshot ≤ `D_k` with coverage carry-forward — the daily stat as first built (2026-08-16) |
  | W3-refresh-mix | refreshed solve + tracking mix prior frozen at `y` |
  | W2-blend50 (`--diagnose` only) | refreshed solve + `0.5·W1 prior + 0.5·W2 prior` — the daily stat since 2026-08-18 |

- Coverage rule (found on the first smoke test, then made the semantics):
  in-season EPM snapshots list only players with a completed game in
  season `S`, and the pre-season snapshot lacks rookies but carries
  last-season leftovers. `W2` therefore uses the fresh value where the
  in-season snapshot covers a player and the season-start value otherwise
  (an injured returner keeps his preseason prior instead of dropping to
  0 — without this, 55+ priors vanished mid-season in the smoke run).
  Per season the December snapshot covers ~500–550 board players, growing
  to ~520–620 by April; 30–200 players are carried from the season-start
  snapshot and 80–150 (rookies, returners, hardship signings — the Dec 2021
  → Jan 2022 jump 507→612 is the COVID wave) enter only in-season.
- Snapshots: 25 monthly in-season snapshots (Dec–Apr, 2018-19 through
  2022-23) were fetched for this test with `fetch_epm_data.py`; 2023-24
  onward reuses the existing biweekly cache. Season 2021's first window
  is degenerate (`D_1` = 2020-12-01 precedes the 2020-12-22 tip: zero live
  rows, `W2 = W1` there).
- Pairs (pre-declared): W2 vs A1 (primary), W2 vs W1, W1 vs A1, W2 vs W0,
  W2 vs W3, W0 vs A0. Game-clustered bootstrap as in the harness (2,000
  draws, seed 11, 5–95%); pooled cuts resample season-prefixed games. 2025
  and 2026 were already opened at the season level in this lane, so every
  number below is descriptive; 2026-27 stays untouched.

### Results

Possession MSE, all scored windows (Dec 1 → RS end):

| Season | A0 | A1 | W0 | W1 | W2 | W3 |
|---|---|---|---|---|---|---|
| 2019 | 0.122139 | 0.122131 | 0.122054 | 0.122050 | 0.122052 | 0.122047 |
| 2020 | 0.125067 | 0.125041 | 0.124947 | 0.124922 | 0.124919 | 0.124931 |
| 2021 | 0.122908 | 0.122877 | 0.122831 | 0.122807 | 0.122809 | 0.122823 |
| 2022 | 0.122052 | 0.122047 | 0.121969 | 0.121962 | 0.121968 | 0.121965 |
| 2023 | 0.122821 | 0.122800 | 0.122703 | 0.122686 | 0.122682 | 0.122695 |
| 2024 | 0.120424 | 0.120408 | 0.120379 | 0.120367 | 0.120356 | 0.120370 |
| 2025 | 0.125279 | 0.125251 | 0.125205 | 0.125180 | 0.125186 | 0.125190 |
| 2026 | 0.126299 | 0.126293 | 0.126155 | 0.126147 | 0.126148 | 0.126143 |

Paired deltas, e-6 (negative = first candidate better), 5–95% intervals:

| Pair | dev 2019–2024 (1.06M poss) | 2025–2026 (360k) | all (1.42M, 7,400 games) | P, all |
|---|---|---|---|---|
| W2-fresh vs A1-annual-pz | **−84.6 [−95.6,−73.7]** | **−104.9 [−124.9,−85.7]** | **−89.8 [−99.5,−80.3]** | 1.00 |
| W2-fresh vs W1-stale | −1.4 [−5.6,+3.0] | +3.9 [−3.1,+11.3] | **−0.04 [−3.7,+3.5]** | 0.51 |
| W1-stale vs A1-annual-pz | −83.3 [−91.9,−74.4] | −108.8 [−125.1,−92.7] | −89.7 [−97.6,−82.4] | 1.00 |
| W2-fresh vs W0-plain | −16.0 [−21.7,−10.2] | −12.6 [−22.7,−3.4] | **−15.1 [−20.2,−10.1]** | 1.00 |
| W2-fresh vs W3-mix | −7.9 [−12.6,−3.2] | +0.7 [−7.5,+8.9] | −5.7 [−9.7,−1.8] | 0.99 |
| W0-plain vs A0-annual-plain | −86.6 [−96.0,−77.4] | −109.3 [−125.2,−92.5] | −92.3 [−100.5,−84.4] | 1.00 |

Per season, W2 vs W1: 2019 +2.1, 2020 −3.8, 2021 +1.6, 2022 +6.3, 2023
−4.3, 2024 −10.7 [−20.0,−0.9], 2025 +6.0, 2026 +1.9 — three negative,
five positive, and only 2024's interval clears zero. W2 vs W0 is negative
in all eight seasons (five intervals clear).
W2 vs W3 is negative in five (2020, 2021, 2023, 2024 clear) and positive
in three (2019, 2022, 2026, none clear).

By refresh window (pooled across seasons, descriptive): the refresh gain
over the static annual map grows through the season — W1 vs A1 −56.6
(Dec), −73.9 (Jan), −102.8 (Feb), −116.7 (Mar), −106.1 (Apr) — while W2
vs W1 is flat everywhere (+1.7, −2.1, +0.2, +0.1, +0.3) and the prior's
edge over plain refresh holds in every window (−18.0, −17.4, −11.5,
−11.9, −16.8).

### Reading

- **The daily benefit is real, large, and comes from the solve, not the
  snapshot.** Refreshing the possession solve inside the season is worth
  ~90e-6 pooled (P 1.00 in every season and window), roughly four times
  the whole prior's annual gain (−23e-6), and it grows from ~55e-6 in
  December to ~105–117e-6 by March/April as the static map goes stale.
- **Swapping in a fresh EPM snapshot at full weight ties the season-start
  prior with carry-forward.** W2 vs W1 is −0.04e-6 pooled with a ±3.6
  interval; no season, no window, and neither pooled cut leaves zero, and
  the seasons split 3/5. The 2026-08-18 diagnose run (next subsection)
  shows this is *not* "the fresh values carry nothing": the fresh movement
  predicts the stale-prior solve's forward errors, but at full weight it is
  double counted against the live rows, and the two effects cancel almost
  exactly at α=3000. The half-weight blend banks the new half.
- **The prior still earns its keep within season.** W2 vs W0 is −15e-6
  pooled (every season negative). It is smaller than the annual −23e-6
  because the live rows carry more of the information as the season
  progresses, and the pz prior stays nominally ahead of the frozen tracking
  mix prior (−7.9 dev, +0.7 frozen, −5.7 pooled), so C6-pz at scale 1.0
  remains the working default within season too.
- **The snapshot refresh also matters for coverage.** Rookies and other
  newly covered players (80–150 per season) get a prior only from an
  in-season snapshot; injured returners keep their preseason value via
  carry-forward.

### Why the fresh snapshot ties, and what to do with it (`evaluate --diagnose`, 2026-08-18)

Same eight seasons, same forty windows, same base candidates (the base
numbers reproduce exactly); the diagnose mode adds W1/W2 at prior scale
1.5, α=1500 and α=6000, the pre-specified half blend `W2-blend50`
(refreshed solve + `0.5·season-start + 0.5·fresh-with-carry-forward`
prior), and three reads. Outputs:
`results/tov_predictive_ztov_daily_{eval,windows}_diag.csv`,
`tov_predictive_ztov_daily_diag.csv` (slopes, γ, exposure quintiles,
ledger drift) and `tov_predictive_ztov_daily_diag_players.csv`
(per-player per-window ledger, coefficients, live and forward exposure).

1. **Does the prior's movement predict forward errors?** Regress each
   candidate's forward residual on the possession's signed exposure to the
   movement `Δ = p_fresh − p_start` (sum of `Δ_i/100` over its five
   offensive players; 99% of scored rows have nonzero exposure, sd
   0.84/100). Slope 0 = already priced, >0 = under-used, <0 =
   over-corrected. Pooled all (game-clustered 5–95%): A1-annual-pz **+0.57
   [+0.51,+0.63]** (positive control — the annual map has none of the
   season's rows), W0-refresh-plain +0.20 [+0.14,+0.26], **W1-refresh-stale
   +0.26 [+0.20,+0.32]**, **W2-refresh-fresh −0.07 [−0.13,−0.01]**,
   W2-blend50 +0.10 [+0.03,+0.16]; W1 at α=6000 +0.37, W2 at α=6000
   −0.07. Dev (2019–2024) and frozen (2025–2026) cuts give the same
   pattern (W1 +0.26 / +0.26, W2 −0.07 / −0.07). So the refreshed
   stale-prior solve does *not* fully price what the fresh snapshot learned,
   and the full swap overshoots.
2. **The MSE-optimal weight on the swap.** The prior-centred ridge is
   linear in the prior, so `u = e_stale − e_fresh` is exactly the
   per-possession adjustment the swap induces and `γ = Σ e_stale·u / Σ u²`
   is its MSE-optimal multiple (1 = use fully, 0 = pure noise, 0.5 = using
   it and ignoring it tie by construction, since `MSE₂ − MSE₁ =
   mean(u²)(1 − 2γ)`). Pooled all: **γ = 0.50 [0.36,0.64]** (rms adjustment
   0.36/100, best-case gain −3.2e-6); dev 0.555 [0.39,0.72] (−3.9e-6);
   frozen 0.34 [0.07,0.63] (−1.5e-6). Per season 0.43, 0.64, 0.44, 0.29,
   0.80, 0.98 (2024 — the one season where W2 cleared W1), 0.27, 0.42. At
   α=6000 γ is 0.62 (the stale prior costs more when trusted more) but
   W1-a6000 itself is +14.8e-6 worse than W1.
3. **The half blend, pre-specified from the double-counting hypothesis, is
   the practical use.** W2-blend50 vs W1: **−3.17e-6 [−4.98,−1.39], P 1.00
   pooled**; dev −3.84 [−6.01,−1.83]; frozen −1.17 [−4.80,+2.43]; negative
   in 8/8 seasons (2019 −2.7, 2020 −5.3, 2021 −2.6, 2022 −0.7, 2023 −3.9,
   2024 −8.1 [−13.1,−3.1], 2025 −0.2, 2026 −2.1) and 5/5 windows (−2.0,
   −4.1, −3.0, −2.9, −3.9); W2-blend50 vs W2 −3.13 [−4.94,−1.31]. It is
   small — ~3.5% of the ~90e-6 solve-refresh gain, ~20% of the prior's
   in-season gain over plain refresh — but it never has the wrong sign.
4. **The tie was not an α or scale artefact.** Within season the stale
   prior's weighting is locally optimal: W1 at scale 1.5 +3.9e-6, α=1500
   +4.9, α=6000 +14.8 (all worse than W1, intervals clear); W2 at scale
   1.5 +10.2, α=1500 +7.6, α=6000 +10.3 (all worse than W2). Fresh vs
   stale at each weighting: scale 1.5 +6.2 [+0.5,+11.3], α=1500 +2.7,
   α=6000 −4.6 [−8.8,−0.3] — trusting the fresh prior more only helps when
   the stale prior is over-trusted.
5. **Where the swap helps and hurts.** By quintile of the possession's
   summed |Δ| exposure (pooled): W2 vs W1 Q1 +2.3, Q2 +1.5, Q3 −4.7, Q4
   **−11.5 [−20.3,−2.5]**, Q5 **+12.3 [−0.6,+25.3]**; blend vs W1 +0.5,
   −0.6, −4.6 [−8.0,−1.3], −9.2 [−13.4,−5.0], −1.9. The largest movements
   are on regulars (top movement quintile: median 1,800 live possessions,
   ~26 mp/48; |Δ| barely correlates with exposure, r 0.14), and the
   correlation between EPM's movement and what the *plain live solve* had
   already moved a player rises from 0.04 in the smallest-movement quintile
   to 0.54 in the largest — the double-counting signature exactly where the
   full swap hurts.
6. **The ledger does move within a season, sensibly.** Among players in
   both snapshots (live-possession weighted, April): `p_tov_100` sd ≈ 1.0
   with start→April correlation 0.90–0.91 and movement sd 0.44; the
   allowance moves as much (sd 0.46) and in the same direction
   (`corr(Δp_tov, Δallowance)` 0.54 — usage and assist growth raise both),
   so the excess/prior moves sd 0.43 (prior sd 0.53–0.61, start→April
   correlation 0.77–0.87). EPM's movement correlates +0.39 with the plain
   live solve's movement and +0.34 with the stale-prior solve's. Concrete
   2025-26 April movers: Andrew Nembhard `p_tov` 2.79→3.43 but allowance
   2.83→4.99, prior 0.00→−1.18 (live solve had moved −0.44); Cade
   Cunningham `p_tov` 5.90→5.18, allowance 5.58→6.37, prior +0.29→−0.89
   (live −0.23). The expected-turnover numbers are not weird or static; the
   issue is only that half of what they learn in-season, the lineup solve
   has already learned from the same games.

Reading: "does a fresh snapshot make the daily stat more predictive?" —
yes, at half weight; at full weight the gain and the double counting cancel.
The board's in-season default is therefore the half blend (below); W1 and
W2 stay as side columns and as the pre-registered 2026-27 comparators. A
movement-dependent weight (shrink the biggest movements harder, per read 5)
is a logged idea, not a tested candidate.

### The daily board (`board`)

```bash
cd nba_pipeline/scripts
../../.venv/bin/python research_predictive_ztov_daily.py board --as-of 2026-08-16
# during the season, to also pull today's snapshot first (needs EPM_API_KEY;
# `set -a; source .env; set +a` from the repo root; the key is never printed):
../../.venv/bin/python research_predictive_ztov_daily.py board --as-of "$(date +%F)" --fetch
```

Semantics as of `D`: snapshot = latest cached snapshot requested ≤ `D`
(with `--fetch`, today's, skipped if already cached; on a fetch failure it
prints the fetch instructions and falls back to the latest cached one);
structural year = the last complete season that has `|kappa|` and lane
prices in `results/rztov_seasons_2014_2026.csv` (a WARNING is printed when
that lags the newest complete season, i.e. until the explorer build runs
for the new year); training = the structural year's trailing five complete
seasons plus every live-season row before `D`, weights `0.5^((D −
day)/1500)`, complete seasons centred on their own mean and the live season
on its running mean. **In-season prior rule (default `--fresh-weight
0.5`, the diagnose finding above):** `prior = (1 − w)·season-start snapshot
+ w·latest snapshot with carry-forward`, where the season-start snapshot is
the latest one requested on or before the live season's first game day and
carry-forward means players missing from the latest snapshot keep their
season-start prior (`fallback_snapshot` in the manifest, so uncovered
rookies get half of their first in-season prior — exactly what W2-blend50
tested). `--fresh-weight 0` reproduces W1, `1` reproduces W2; the manifest
records `prior_rule` and `fresh_weight`. Before the first in-season
snapshot, or in the offseason, the latest snapshot alone is the prior (the
season-level C6 structure; the offseason board is byte-identical to the
pre-2026-08-18 build). Columns: `off_pz` (the daily stat: refreshed solve
+ blended pz prior at scale 1.0), `off_pz_stale` and `off_pz_fresh` (the
same solve under the season-start-only and latest-only priors — by
linearity `off_pz = (1 − w)·off_pz_stale + w·off_pz_fresh`; equal to
`off_pz` outside a live season), `off_plain`, `off_mix` (frozen tracking
prior, scale 0.5), `prior_pz`, `prior_pz_stale`, `prior_pz_fresh`,
`prior_start` (season-start ledger prior), `prior_mix`, the three
defense-side maps (the prior is offense-only, so they differ only through
the joint solve), decayed/window/live possession exposure, the full pz
ledger from the latest snapshot (`p_tov_100`, `p_mp_48`, `v_hat`,
`ast_share`, `pot_hat`, `allowance_hat`, `excess_hat`, `prior`,
`snapshot`; season-start values fill players the latest snapshot lacks), and
coverage flags. Sign: positive `off` = commits more TOV/100 than lineup
context predicts, positive `def` = forces more. Names come from the rZTOV
research surface, then the daily `master_results/weighted_factors_*.csv`
exports, then EPM.

In-season check (as of 2026-03-01: snapshot `epm_all_2026-03-01_go0.csv`,
578 players; structural year 2025; training 2021–2025 plus live 2026,
1,324,783 rows; carry-forward from `epm_all_2025-10-15_go0.csv` for 110
players): 1,075 offense rows, 673 pz-covered; blend identity gap 2e-16,
solve-linearity gap 1.5e-14; `corr(off_pz_stale, off_pz_fresh)` 0.981; sd
of the stale / fresh / blended prior 0.508 / 0.533 / 0.492.

As of 2026-08-16 (offseason: structural year 2026, snapshot 2026-08-16 =
provider effective 2026-06-13, 601 EPM rows; training 2022–2026, 1,181,320
rows, 2,058 columns, no live season): 1,029 offense rows, 593 pz-covered,
777 mix-covered; `corr(off_pz, off_plain)` 0.903, `corr(off_pz, off_mix)`
0.925, `corr(prior_pz, prior_mix | both)` 0.900; among ≥2,000-possession
players `corr(prior_pz, off_pz)` 0.842. Fewest turnovers versus context:
Shai Gilgeous-Alexander −2.48, Kawhi Leonard −1.86, Jalen Brunson −1.83,
DeMar DeRozan −1.81, Tyrese Maxey −1.70, Tyus Jones −1.70; most: Nick
Richards +2.04, Trey Jemison III +1.95, DeAndre Jordan +1.95, Jericho Sims
+1.79, Zach Edey +1.79 (turnover-prone bigs, as in every version of this
lane).

## Boundary and next steps

- Pre-registered 2026-27 untouched scorecard (score once, no re-tuning):
  incumbent C3-mix (`alpha=3000`, `half_life=1500`, mix scale 0.5);
  primary challenger C6-pz (scale 1.0, no tracking dependency); secondary
  C8-blend (λ=0.25, scale 0.75); all against C2-td5. `|kappa|` from the
  joint window ending the as-of year; EPM snapshot = latest before first
  scored game; volume and AST maps refit walk-forward through the as-of
  year.
- Pre-registered 2026-27 within-season read (score once, same Dec–Apr
  windows, no re-tuning): the daily board as built (W2-blend50: refreshed
  solve + `0.5·season-start + 0.5·latest-with-carry-forward` prior, pz scale
  1.0) versus W1 (season-start snapshot), W2 (latest snapshot at full
  weight) and A1 (static annual C6 map), plus the γ read. Expected on the
  evidence above: blend50 ≤ W1 ≈ W2 ≪ A1 with γ near 0.5 (dev 0.55, frozen
  0.34); a blend50 loss to W1, or γ far from 0.5, would be new information.
  The half weight is a dev-fold quantity (γ selected on 2019–2024 sits at
  0.55; 0.5 was pre-specified from the double-counting hypothesis, not
  tuned), and it must not be re-tuned on 2026-27.
- The daily job is a research board (`results/tov_predictive_ztov_board_*`,
  gitignored). It has no publish step and must not acquire one before the
  2026-27 gate.
- Promotion beyond a research board additionally requires the standard
  overlay rule: only stabilized player-season summaries may ever flow to
  VPM, never fold predictions or raw rows, and only after the confirmation
  gate.
- Logged future work, not v1: re-expressing the posterior as implied
  personal break-evens (the original operator framing — this v1 reversed
  the flow by sending allowance into the predictive solve); defensive-side
  prior candidates; lane-split (scoring vs bad-pass) predictive solves;
  a personal-conversion tier for the AST→potential-assist map (v1 uses one
  global walk-forward line); the hinge-ast convexity candidate already
  logged for 2027 in `RZTOV.md`.

## Reproduce

```bash
.venv/bin/python nba_pipeline/scripts/research_prior_informed_tov_rapm.py
```

EPM predictive-ledger extension (Product C):

```bash
.venv/bin/python nba_pipeline/scripts/research_predictive_ztov_prior.py
```

Dev-only mix-scale extension plus the as-of-2025 map dump (never touches
the frozen folds):

```bash
.venv/bin/python nba_pipeline/scripts/research_prior_informed_tov_rapm.py \
  --dev-mix-scales 0.125,0.25,0.375,0.5,0.75,1.0 --dump-maps
```

Within-season refresh test (all eight seasons; `--seasons 2025` for one) and
the daily board (run from `nba_pipeline/scripts`; the eval needs the monthly
in-season snapshots in `cache/epm`, fetched once with
`fetch_epm_data.py --dates ...` under `EPM_API_KEY`):

```bash
cd nba_pipeline/scripts
../../.venv/bin/python research_predictive_ztov_daily.py evaluate
# adds the scale/alpha grid, W2-blend50, and the slope / gamma / quintile /
# ledger-drift reads; writes the *_diag.csv artifacts, never the main eval files
../../.venv/bin/python research_predictive_ztov_daily.py evaluate --diagnose
# in-season prior = 0.5*season-start + 0.5*latest (carry-forward); 0 = W1, 1 = W2
../../.venv/bin/python research_predictive_ztov_daily.py board --as-of "$(date +%F)" [--fetch] [--fresh-weight 0.5]
```
