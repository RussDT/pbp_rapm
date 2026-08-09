# Playoff Handling In Multi-Year RAPM

Status: implemented and gated (August 2026). The `--ps-prior` solve, the delta
diagnostic, and the forward gate exist and have run on development folds.
Verdict so far: `--season-effects` pooling is the recommended production
config; the `ps-prior` surfaces are descriptive playoff-attribution artifacts
that FAIL the forward-prediction gate and must not replace the canonical
rating. Details in "Development evidence" below.

## The Problem

Every production NBA surface in this repo solves with season type `ALL`, so
playoff possessions are pooled with regular-season possessions at equal
per-possession weight (time decay only re-weights by calendar date, which
slightly *up*-weights playoffs — see below). An earlier analysis observed that
including playoffs systematically lowers playoff participants' RAPM. An August
2026 audit quantified three mechanisms on this repo's own data:

1. **Good-vs-good compression (the dominant net effect).** In 2025-26, the
   average five-man lineup on the floor in playoff possessions sums to `+6.9`
   net RAPM per 100 on both sides, versus `-0.4` in the regular season — a
   `+7.3 pts/100` stronger opponent diet for playoff participants
   (possession-weighted, using `rapm_22_26_all_rb_se_a2000_4000_results.csv`
   as the rating index). Zero-centered ridge shrinks opponent coefficients
   toward zero, so the strength-of-schedule adjustment is systematically
   incomplete and the shortfall lands on the participants themselves. Playoff
   data is a pure good-vs-good sample, so pooling it drags every participant
   toward the mean — the deepest runs hardest.
2. **Environment level.** Playoff scoring on the exact solver target is
   `1.4-3.6 pts/100` lower than RS (2024: `-1.79`, 2025: `-1.36`, 2026:
   `-3.57`). Without a phase intercept this deflates participants' offense and
   inflates their defense; it roughly cancels in net but distorts the O/D
   split. `--season-effects` absorbs it (the estimated 2026 PS dummy in the
   validation run was `-4.6/100`).
3. **Time-decay amplification.** Under `td700`, same-season playoff rows get
   `~1.11x` the weight of that season's RS rows (playoffs are the latest games;
   a `~107`-day recency advantage), amplifying both effects in the td surfaces.

Exposure is material: PS possessions are `~6%` of an ALL season sample, but the
median playoff participant gets `8.5%` of personal possessions from playoffs
and the 90th percentile gets `20.6%`.

## The Mechanism Fix: `--ps-prior` (RS-prior-centered playoff inclusion)

`rapm.py <METRIC> S E ALL --ps-prior` runs a two-stage solve:

- **Stage 1** solves the regular-season rows only, with the same alphas,
  weights, and nuisance blocks as the main solve. The player coefficients are
  re-centered to possession-weighted mean zero per side over RS possessions.
- **Stage 2** re-solves with the ridge penalty centered on the stage-1
  coefficients — `||beta - beta_RS||^2` instead of `||beta||^2` — via the
  existing prior-centered machinery in `simplified_alternating_rapm` (the
  adjusted-residual trick used by `--age-curve`). Players with no RS
  possessions in the window shrink to zero. `--ps-prior-stage2` picks the
  stage-2 sample:
  - `full` (default): refit the entire RS+PS sample. Warning: re-fitting RS
    rows against a prior fit on those same rows reduces the effective
    shrinkage — on 22-26 this expands the rating scale by `~1.28x`
    (`std(net)` 3.16 vs 2.46 for the single-stage `-se` solve, corr `0.992`).
    Most of what this mode changes is shrinkage dose, not playoff handling.
  - `ps-only`: fit only the playoff rows in stage 2, so RS data is used
    exactly once and playoff evidence alone updates players away from their
    RS priors (the APBR "regular-season-informed playoff RAPM" recipe).
    Verified property: players with zero playoff possessions move by exactly
    one recentering constant (delta std `0.0`), and the scale stays at
    stage-1 level. This is the meaningful descriptive variant.

Why this addresses the mechanism: playoff opponents are anchored at their
RS-established levels, so a playoff possession updates a player only by
performance *relative to a realistic expectation against that opposition*.
The good-vs-good compression is removed at its source instead of being
reweighted around. The estimand is "regular-season-established value, updated
by playoff evidence."

Recommended companion flag: `--season-effects`, so the playoff environment
level lands in the `(season, PS)` dummies instead of the player coefficients.

Constraints and composition:

- Requires season type `ALL` (both RS and PS rows must be present).
- Cannot combine with `--cv-alpha` (fix alphas explicitly), `--off-prior` /
  `--def-prior`, `--age-curve`, or the DARKO strength modes.
- Composes with `--timedecay`, `--rubberband`, `--season-effects`,
  `--age-dummies`, and `--fixed-season-baselines`. Nuisance blocks
  (season/age/rubberband) are re-fit in both stages and carry no prior.
- Output filenames gain a `psprior` token (e.g.
  `rapm_22_26_all_se_psprior_results.csv`).

### Companion artifact: `*_psprior_stage.csv`

One row per player: `off/def/net_prior` (stage 1), `off/def/net_final`
(stage 2, identical to the results CSV), `off/def/net_delta`, per-phase
possession counts (`rs_off_poss`, `rs_def_poss`, `ps_off_poss`,
`ps_def_poss`), and `ps_share`. Caveat: priors are centered over RS
possessions while finals are centered over pooled possessions, so deltas carry
a small common offset in addition to each player's playoff update.

### Validation (2025-26 single season, `RAPM 26 26 ALL -se --ps-prior`)

- Stage finals match the results CSV exactly (max abs diff `0.0`).
- `corr(net_prior, net_final) = 0.983`; the PS phase dummy landed at
  `-4.61/100`, consistent with the measured raw environment gap.
- Playoff evidence produces real, playoff-sized updates (e.g. Wembanyama
  `+3.39` net over his RS prior after the 2026 title run) while non-playoff
  players stay anchored.

## Diagnostics

`scripts/diagnose_ps_pooling.py` compares two player results CSVs (e.g.
RS-only baseline vs pooled ALL, or vs `psprior`) and reports playoff-
participant deltas overall, by PS-share quartile, and the top movers, writing
the merged per-player table under `results/diagnostics/`.

```bash
python nba_pipeline/scripts/diagnose_ps_pooling.py \
  --baseline nba_pipeline/results/rapm_22_26_rs_results.csv \
  --candidate nba_pipeline/results/rapm_22_26_all_results.csv \
  --start-year 22 --end-year 26
```

## Forward gate

`scripts/evaluate_ps_handling.py` is the promotion gate. It trains six
candidates per fold — `rs_only`, `all_pooled`, `all_se`, `all_se_psprior`,
`all_se_psprioronly` (`--ps-prior-stage2 ps-only`), and the `all_se_a1500`
shrinkage-dose control — on
a trailing window and scores summed player-coefficient predictions on the
*next* season's RS possessions, level-neutral (both target and prediction are
centered on the test sample, so season intercept handling cannot leak into the
comparison). Paired per-game MSE deltas vs the `rs_only` baseline get a
game-clustered bootstrap CI, matching the repo's other forward gates. Default
folds: train `22-24` -> test `25 RS`, train `23-25` -> test `26 RS`; 2026-27
is untouched and reserved for confirmation.

```bash
python nba_pipeline/scripts/evaluate_ps_handling.py
```

### Development evidence (August 8, 2026)

Gate run: folds `22-24 -> 25 RS` and `23-25 -> 26 RS`, six candidates, default
alphas 3000/3000, bootstrap 2000, seed 20260808. Paired per-game MSE delta vs
`rs_only` in 1e-6 units (negative = better), pooled possession-weighted:

| candidate | delta vs rs_only (e-6) | 95% CI | verdict |
|---|---|---|---|
| all_se | `-9.2` | `[-56.7, +38.2]` | tied with RS-only, nominally best; consistent in both folds |
| all_pooled | `+29.4` | `[-17.9, +74.4]` | tied with RS-only (n.s.) |
| all_se_a1500 | `+156.9` | `[+66.6, +247.7]` | lower alpha hurts — 3000 is on the right side of 1500 |
| all_se_psprioronly | `+244.2` | `[+107.6, +383.1]` | significantly worse in both folds |
| all_se_psprior (full) | `+356.9` | `[+238.1, +472.9]` | significantly worse in both folds |

Readings:

1. **The pooling drag is real but predictively tiny.** Despite the `+7.3
   pts/100` opponent concentration, playoff rows are only `~6%` of an ALL
   sample, and pooled candidates forecast future RS possessions as well as
   RS-only. The descriptive drag on the 22-26 no-decay quartet is
   correspondingly modest: participants mean `-0.07` net vs RS-only, worst
   PS-share quartile `-0.16`, deepest quartile `+0.05` (winners' genuinely
   strong playoff evidence offsets the compression).
2. **`--season-effects` is the right production default for ALL runs.** It
   fixes the O/D environment distortion (the estimated 2026 PS dummy is
   `-4.6/100`) and is never worse on the gate.
3. **Playoff evidence at alpha 3000 does not transfer to next-RS
   forecasting.** Both psprior modes lose the gate; the full mode loses
   partly through its scale expansion (compare `all_se_a1500`), and the
   ps-only mode loses because `~48k` playoff possessions per window produce
   updates too aggressive for their information content. A future sweep of a
   larger stage-2 alpha (stronger shrinkage on the playoff update) is the
   natural follow-up and can reuse this gate unchanged.
4. **The ps-only surface is the correct descriptive answer to "how should a
   playoff run change the rating."** On 22-26 (`rapm_22_26_all_se_psprioronly`
   vs RS-only): non-participants move by one constant; early-exit
   participants drop modestly (Q1-Q3 mean `-0.16` to `-0.28`); deep-run
   players rise (`+0.32` mean in the top PS-share quartile). Movement is
   driven by playoff performance against realistically-anchored opponents,
   not by ridge compression. Publish it only labeled as the playoff-updated
   reading surface, never as the canonical forecast-facing rating.

## Also fixed in this pass

`compute_time_decay_weights` previously gave rows with unparseable dates
`age_days = 0`, i.e. silently the maximum weight. Unparseable dates now get
the oldest observed age (minimum weight) with a loud warning, and an
all-unparseable sample falls back to uniform weights with a warning.

## Production status and promotion path

**Applied August 8, 2026 (operator decision, development-validated):**
`--season-effects` was added to the daily plain 1Y-6Y windows and both td700
runs in `daily_rapm_update.sh`, with the `_se` filenames flowing through
rather than being renamed on copy. The 13-year `14 26 ALL --rubberband` run
was deliberately left unchanged (not covered by the gate folds). Old -> new
published names:

- `weighted_factors_{26,25_26,24_26,23_26,22_26,21_26}_all.csv` ->
  `..._all_se.csv` (standard and `alt3` families)
- `weighted_factors_21_26_all_td700.csv` -> `weighted_factors_21_26_all_se_td700.csv`
- `weighted_factors_21_26_all_rb_td700.csv` (Supabase TimedecayRAPM source) ->
  `weighted_factors_21_26_all_rb_se_td700.csv`
- per-run results folders gain `_se` (e.g. `results/rapm_22_26_all_se/`)

Updated in the same pass: the daily sync lists, `upload_timedecay_rapm.py`,
`upload_six_factor.py`, `update_2026_{josh_rapm,purerapm,scaledoutput}.py`,
and the `nba_pipeline/CLAUDE.md` publish list. `update_2026_scposs.py` was
left on the old names because the scposs mapper is skipped in the daily run
pending its schema audit. External consumers that read the old
`weighted_factors_*_all*.csv` names directly from the `rapms` repo must
switch to the `_se` names; stale old-name copies remain there until removed.

Remaining guidance:

- **Do NOT replace any canonical surface with a `psprior` solve.** Both
  stage-2 modes fail the forward gate. `--ps-prior --ps-prior-stage2 ps-only`
  is the sanctioned *descriptive* companion: run it beside the canonical
  solve when a playoff-updated reading is wanted, and lead with the
  `*_psprior_stage.csv` (prior / final / delta / ps_share) as the
  playoff-attribution audit.
- Open research lane: sweep a stronger stage-2 shrinkage for the ps-only
  update through `evaluate_ps_handling.py` before believing any playoff
  information helps future-RS forecasting.
- 2026-27 remains untouched confirmation data for the `-se` flip and any
  future psprior promotion attempt.

Do not downweight playoff possessions as an alternative: it half-mitigates the
compression while diluting real playoff signal, and it has no principled
weight. Do not upweight them for "importance" while the solver is
zero-centered ridge.
