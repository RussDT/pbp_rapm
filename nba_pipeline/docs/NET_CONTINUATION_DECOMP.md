# Net-continuation (kappa) DECOMP tree

Owner decision 2026-09-03: Net DECOMP is the selected forward explanatory
product. The current implementation began as a validation-lane variant of the
gross five-child DECOMP shot tree and
prices every first-chance FGA at its expected remaining-possession value
instead of its gross value. Motivated 2026-08-08: a forced miss is not a
fully dead ball — the offense recovers X% of misses — so gross rim-FG
defense overstates net impact and gross SECOND_CHANCE_CLEAN conflates
opportunity (exposure to recoverable misses) with conversion (rebounding
skill).

## Missing FT percentage policy — corrected 2026-09-05

Owner decision: when shooter season FT percentage is missing, use the actual
FT outcome (1 for a make, 0 for a miss). When available, retain the shooter's
season FT percentage as the expected value. `clean_chance_utils` now applies
this policy to the shared FC/SC event scoring and FT children, including the
case where the stats panel is unavailable. Eight targeted tests pass,
covering known percentages, missing-player makes/misses, unavailable stats,
FT-child closure, and existing shot-tree identities.

The 2026-09-06 artifact repair rebuilds the affected raw targets and applies
exact target-difference ridge corrections with each original solver operator.
See [the repair receipt](FT_FALLBACK_REPAIR.md) for execution status and scope.
No balancing residual is added. The audit below describes pre-correction artifacts.

## Residual audit — 2026-09-05

The displayed residual is **not explained by separate ridge fits alone**.
With identical targets summing to the parent, design matrix, row weights,
penalty matrix, and linear centering, ridge preserves additivity. Source net
closure against gross FC+SC does not establish closure against the independently
built RAPM parent.

A direct audit of all 60 aligned 1997-2026 RS/PS file triples found 610
possessions where `RAPM.Net_Diff != FIRST_CHANCE.Net_Diff +
SECOND_CHANCE_CLEAN.Net_Diff`, out of 7,157,906 rows. There are 609 RS cases
and one 2000 playoff case; the maximum discrepancy is 1.5 points. All exact
possession keys and ten lineup slots align. In 2026, 144 RS rows differ and
all playoff rows match. The actual eight alias targets, including turnover
imputation and its complement, close to FC+SC within `5.33e-14` in 2026.

The confirmed implementation discrepancy is the missing-FT-percentage policy:
`reprocess_metric.py` requests `missing_ft_fallback="actual"` for RAPM, while
`clean_chance_utils._attach_rapm_scoring` uses `FTPerc.fillna(0.75)` for FC/SC.
For example, game 22500004 / event 714 has parent 3.00 versus FC 2.75
(Jones Garcia's made and-one); game 22500019 / event 658 has parent 1.00
versus FC 1.50 (Juzang makes one of two FTs).

Solving only `parent target - sum(component targets)` with the current 2026
smooth-rubberband, RS/PS season-effect, 2000/4000 operator reproduces the
exported offensive residual at correlation `0.999309` and defensive residual
at `0.997119`. Maximum remaining differences are `0.002542` offense and
`0.003077` defense, consistent with the small numerical/export remainder.
Sandfort's offensive residual is -0.132 versus a fitted target-gap effect of
-0.131974; Broome's is -0.129 versus -0.129562.

The 2022-2026 table also contains 500 mismatched target rows, but its saved
August 18 solve logs use the old 24-bucket rubberband/alternating solver.
The 2026 single-year table uses September 3 smooth-rubberband/joint solves.
Do not claim that the current smooth operator exactly reproduces the older
five-year artifact. Existing-file restart behavior can retain old specifications
under the same unsuffixed `rb` output names.

Audit receipts are local generated files under `tmp/net_residual_*` (including
row counts, player gap effects, and a reproducible diagnostic script). No
ratings, processed files, or solver behavior were changed in this audit.
The repair must standardize FT fallback across parent and both chance branches,
rebuild affected targets and artifacts with explicit solver provenance, and
add a direct parent-versus-component target identity gate. Do not absorb this
discrepancy into SC or another balancing child. The historical 0.30 export
ceiling is a guard, not evidence that the residual is mathematically expected.

## Construction

Built inside `process_rapm_blocks/clean_chance_utils.py` in the same pass
as the gross tree; nothing about the gross columns changes.

- Per file, `kappa(zone)` = average downstream second-chance points on
  possessions whose first-chance FGA missed from that zone (rim / mid /
  three). Measured values are stable across 2024-2026 RS+PS:
  rim ~0.45-0.49, mid ~0.27-0.37, three ~0.30-0.34 points per miss.
- Event repricing: `v = ActualNet + kappa(zone) * 1{first-chance FGA
  miss}`; the net tree is the standard five-child construction applied to
  `v` with zone averages recomputed on `v` (`FC_DECOMP_NET_*_Diff`,
  parent `FC_DECOMP_NET_EFG_Diff`).
- Payback ledger: `SC_NET_CONT_Diff = Second_Chance_Diff - kappa` on the
  same possession (`FC_CONT_KAPPA_Diff` is the audit column), so
  gross FC + gross SC = net FC + net SC identically per possession.
- FT-trip misses never start the `after_miss` segment in this codebase
  (their continuation already lives inside FIRST_CHANCE), so no FT kappa
  is needed for closure.
- Closure is enforced at build time (child, kappa-bridge, SC-bridge, and
  grand identities; hard raise above 1e-9).

Aliases `DECOMP_NET_{EFG,RIM_FREQ,RIM_FG,MID_FG,THREE_FREQ,THREE_FG,
MID_VALUE}` are registered in `rapm.py` (baseline map, CLI choices,
family map) and `tune_forward_component_alphas.py`. SECOND_CHANCE_CLEAN
parquets carry `SC_NET_CONT_Diff` + `FC_CONT_KAPPA_Diff` as companion
columns; the family target itself is unchanged.

## Validation (2024-2026 RS+PS, `processed_netcont_validation/`)

`scripts/validate_net_continuation_decomp.py`:

- Legacy columns byte-identical to production on all 12 files (~780k
  possessions); all closure identities exact.
- Matched-alpha gross-vs-net solves (pooled, uniform): six-component
  sums essentially unchanged (corr 0.998, mean |delta| ~0.07 pts/100) —
  the repricing moves attribution between components, not totals.

## What the repricing shows

- Elite rim contest compresses ~24% (Holmgren d_rim_fg -2.68 -> -2.03,
  Wembanyama -2.18 -> -1.66, Kornet -1.89 -> -1.44); symmetric for bad
  contesters (moves toward zero).
- The SC exposure artifact is removed: gross d_SC penalized miss-forcers
  for the recoverable misses they created. Kessler +0.68 (gross "bad SC
  defender") -> -0.07 (average); Holmgren +0.14 -> -0.95; Wembanyama
  -0.96 -> -2.09. Net SC is conversion skill relative to opportunity.
- Jokic's d_SC is unchanged (-2.37): defensive rebounding value is
  conversion, not exposure, and he creates little exposure.
- Offensive finishing compresses the same way (Giannis o_rim_fg 2.80 ->
  2.13): a make's value over the miss counterfactual shrinks because
  misses are partly recoverable.
- Frequency repricing is small but systematic: mid-range misses are both
  frequent and recoverable, so net pricing slightly favors three-taking
  and slightly trims rim-taking relative to gross.
- Genuine finding: miss-forcing wings without rebounding behind them
  (Caruso, Thybulle, Isaac, Ausar Thompson) lose 0.3-0.4 pts/100 of
  shot-defense+SC sum — gross accounting paid full price for forced
  misses their lineups did not secure. Rim-protecting bigs keep their
  value because their own rebounding closes the loop.

## Turnover repricing (`ALT_TOV_NET_VALUE`)

The same logic applies to the turnover counterfactual. Gross
`ALT_TOV_VALUE` prices a clean first-chance turnover at the average
non-turnover first chance (`ALT_TS` baseline, ~1.150). The net
counterfactual adds the expected continuation of the ~41% of non-TOV
first chances that end in a recoverable miss: `ALT_TS_NET` baseline =
mean(`FC_TS_NET_Diff`) = mean(gross FC + kappa) on non-TOV rows, ~1.319
(2024-2026 pooled). `ALT_TOV_NET_VALUE` is the mirror alias on that
baseline; player values rescale by ~1.15x on both sides.

Net relative pricing of defensive events (pooled 2024-2026): steal
+1.32 (possession dead with certainty) > forced rim miss +0.79 net of
continuation > forced three miss +0.74. Gross pricing had misses at full
value and turnovers underpaid. Caruso illustrates the correction: his
shot-defense+SC sum drops 0.41 under net pricing while his TOV value
gains +0.36 — total roughly unchanged, composition corrected (steals
were undervalued relative to contests).

## Net baseline-separated family (audit surface)

The five-child tree welds the mid-numeraire ("a reference mid shot
happened" per FGA) into MID_FG, so that cell mixes pure mid conversion
with a shot-volume bookkeeping term that the net repricing amplifies
~20%. The baseline-separated family (`ALT_EFG_BASELINE_NET` +
`ALT_EFG_{RIM,MID,THREE}_NET`, columns `FC_NET_EFG_BASELINE_Diff` /
`FC_*_EFG_AA_NET_Diff`) removes the weld: each zone cell is pure
conversion above the overall net average and the numeraire is its own
child. Verified 2024-2026: pure conversion cells transform as clean
compressions (gross-net corr 0.998+, slopes rim 0.78 < mid 0.85 <
three 0.89, ordered exactly by kappa/attempt-value), and all
reshuffling concentrates in the BASELINE child (slope 1.20). Use the
five-child tree for display sums, the separated family for cell-level
interpretation.

## Full-split six-child shot family (preferred display grain)

Owner decision 2026-08-08: no baseline bucket may carry player value.
The full-split family (`ALT_EFG_{RIM,MID,THREE}_FREQ_FULL` gross /
`ALT_EFG_{RIM,MID,THREE}_FREQ_NET` net, plus the shared FG conversion
children) prices frequency at the FULL per-attempt zone value, so the
six atoms sum to the entire EFG mass and the former baseline volume
signal reads as deterrence/generation inside the frequency cells
(FT-trip reference entries ride in MID_FREQ, matching the five-child
convention). Signature result: net pricing amplifies frequency cells
(attempts are worth more with continuation priced in) and compresses
conversion cells — value migrates from making shots to controlling
where shots happen and what follows them. Totals match the five-child
tree exactly; closure enforced at 1e-9 for both pricings.

### Primary-defender assignment diagnostic (2024-25)

`scripts/run_primary_defender_net_decomp.py` attaches strict single-defender NBA
matchup assignments to the full-split net atoms on 753,830 common 2021-2025
first-chance FGA rows. The centered player-season role interaction preserves
the assigned-shot defensive parent and publishes exact primary/nonprimary
contributions. Leave-one-season-out validation selects role alpha 3,000: it
beats role-off in all five folds by 0.0162% mean MSE, and its paired gain is
about four standard errors. Individual contrast repeatability remains modest
(`r=0.306`, rank `0.280`, 59.2% sign agreement across early/late blocks), so
keep the player surface diagnostic-only and do not interpret observed-shot
frequency as causal deterrence. Canonical receipt:
[Primary-Defender Net-DECOMP Split](./PRIMARY_DEFENDER_NET_DECOMP.md).

## Seven-component closure (solve-stack invariance)

The complete net stack {five DECOMP_NET children, SC_NET,
ALT_TOV_NET_VALUE} has a row-identical summed solve target with the
gross stack, through two cancellation channels: miss rows (tree grants
+kappa, SC pays -kappa) and turnover rows (shot components impute TOV
rows at their own baselines, and the net baselines are richer by
exactly the mean-kappa ~0.17 that the net TOV price charges). Verified:
at one uniform alpha the seven-component player sums match gross to
solver convergence tolerance (max 0.007 pts/100). Corollaries:

- Comparing partial stacks (e.g. six components without TOV) or mixing
  alphas across slots shows small total drifts; these are shrinkage
  re-slotting, not accounting gaps.
- Under production per-class alphas, gross vs net totals differ
  slightly by design; which parameterization predicts better is a
  forward-gate question, not a closure question.

## VPM forward test (v4 netcont, 2026-08-10)

The registered VPM forward gate for the net tree has now been run as
development evidence (`vpm` repo, `research_harness_v4_netcont.json`).
Pipeline: isolated `processed_netcont_lane/` FC/SC rebuild (row-aligned with
canonical FC/SC/RAPM), `build_vpm_target_swap.py --net-lane` (net EV swap with
`net_EV = ev_pts + kappa_z * (1 - p_make)`, common gross+net identity gate,
`SECOND_CHANCE_CLEAN` rewritten to `SC_NET_CONT`, kappa recovered exactly with
agreement 1.0 and machine-precision closures on all 22 files), then net
forward labels, refit EPM12 priors, and the frozen-alpha evaluation at the
four development cuts.

Results (paired game-clustered bootstraps, 4,000 replicates, identical
possessions): netcont x2 beats its own net EV no-prior by `+176.65e-6`
`[95.31,260.61]` (4/4 cuts, all promotion gates pass) and beats the gross
EPM12 x2 lane by `+18.43e-6` `[2.19,36.01]` (3/4 cuts, `P(<=0)=0.014`) on the
elementwise-identical possession universe. Net EV no-prior vs gross EV
no-prior is neutral (`-5.21e-6`, CI straddles zero): the answer to "are the
gross buckets mispriced downstream?" is that the raw component sum is not
hurt, but priors map measurably better onto net-priced components. All of
this is development evidence; the accepted VPM boundary is unchanged until an
untouched 2026-27 confirmation.

## VPM feature-enriched follow-up (v5 netcont-feat, 2026-08-11): null

The follow-up challenger (`research_harness_v5_netcont_feat.json`) swapped the
12 EPM12-style recipes for the v4feat prior-feature-lab winners
(box/tracking/derived-EPM features, fallback models for rookies, and a
repo-rebuilt decayed foul rate replacing the dead vendor foul column) on the
identical net lane and forward targets. Every recipe beat its incumbent in
label space, but at the solve level netcontfeat x2 does not beat plain netcont
x2 (`-21.28e-6`, CI `[-60.34,+18.78]`, 1/4 cuts, `P(<=0)=0.861`) and is
neutral vs gross EPM12 x2 (`-2.85e-6`). Plain-recipe netcont x2 remains the
leading netcont research base; details in the vpm repo `RESEARCH_HARNESS.md`.

## Preseason full-season alpha validation (v6 offseason)

The private sibling VPM harness now has a separate immutable offseason alpha
experiment, `research_harness_v6_offseason_netcont.json`. It uses October 1
snapshots except December 1, 2020 for the delayed 2020-21 offseason, trains
only on source-season rows earlier than the target and strictly before the
snapshot, and scores the entire next regular season on future actual
possession points. Development targets are 2019-2022; 2023-2026 are a locked
historical stress replay and cannot select the configuration; 2027 is the
registered untouched confirmation.

The alpha search is an exhaustive five-parameter family rather than twenty
independent dials: shared physical offense/defense alphas, one common
turnover/second-chance multiplier, and residual offense/defense alphas. TD550
and prior scale x2 are frozen. Development selection is no-prior, gives each
season equal weight, uses paired game-clustered uncertainty, and chooses the
most-shrunk configuration inside the paired one-standard-error set. Plain
RAPM is tuned separately under the same folds and rule; fixed `2000/4000`
TD550 RAPM and as-of complete EPM remain benchmarks.

The completed 2026-08-21 run selected physical `750/750`, turnover/second-
chance `3000/3000`, and residual `12000/48000`. It did not clear incumbent
v4: development gain was `+0.65e-6 [-25.15,+26.73]`, and selected x2 gained
`+28.50e-6 [-14.50,+71.65]` over v4 x2 on the report-only 2023-2026 stress
seasons. Selected x2 clearly beat TD550 RAPM
(`+295.66e-6 [+150.70,+435.49]`) and its own no-prior row
(`+88.30e-6 [+17.84,+160.31]`), and beat complete EPM
(`+427.71e-6 [+240.34,+611.02]`) on common realized lineups, while TD550 RAPM
versus EPM was inconclusive (`+132.05e-6 [-62.22,+320.41]`). These are
player-quality isolation results with actual future lineups, not a complete
game forecast. Incumbent v4 remains accepted; the selected row is a frozen
challenger and 2027 remains untouched.

The offseason harness now requires an EPM snapshot whose requested date
exactly matches the configured cut. The exact October 1, 2022 snapshot was
fetched after an audit found a June 30 fallback; all 633 EPM and predictive-
skill rows were identical, so the exact-date rerun changed provenance but not
scores. Preserve requested date and provider-effective game date separately.

For inspection only, same-date selected-alpha and v4 boards were estimated on
August 21, 2026 with separate label-matched EPM12 prior maps and the common
pandemic-safe label schedule. Both contain 1,556 players / 582 current players
and close within `8.88e-16`. Current-player net values correlate `0.9892`, with
a median absolute change of `0.240` points per 100; this does not promote the
selected row or alter the public board.

The net-continuation `kappa(zone)` values already embedded in each historical
source season are date-safe for this exact lane because that whole season is
complete before its configured training snapshot. The December 1, 2020
exception prevents the delayed 2020 postseason from crossing the cut. The runner proves that
the target season and every post-snapshot row are absent from training. Do not
generalize this exception to in-season cuts: a partial-season file still
requires cut-local baseline construction or a separate leakage proof.

## Status / boundaries

- Current for forward explanatory DECOMP use, with public migration in
  progress. The completed five-year archive remains isolated; production
  parquets and gross public artifacts stay unchanged until the 1Y-5Y net
  build and each consumer pass their identity and rollback checks.
- This owner-directed explanatory choice does not promote the accepted VPM
  lane or claim an untouched predictive win. VPM promotion still requires its
  registered forward gate, label/mapper rebuild on net targets, and
  the EV-lane composition: EV components carry expected continuation
  `kappa * (1 - p_make)`, residual components carry the realized-minus-
  expected difference, i.e. make-luck deflated by recoverability.

## Historical rolling artifacts

The isolated research lane can now be rebuilt and solved for all complete
five-season windows with `scripts/run_net_decomp_rolling.py`. The runner uses
ALL + rubberband + RS/PS-aware season effects, fixed alphas `2000/4000`, and
writes `weighted_factors_decomp_net_*` artifacts without touching the gross
public family. The full parquet-prep and quarterly TD companion workflow is in
[Rolling net DECOMP and quarterly TD550 RAPM](./ROLLING_NET_DECOMP_AND_QUARTERLY_TD.md).

The same runner now supports the selected current 1Y-5Y release with
`--current-release`. The separate `run_net_sq_decomp_rolling.py` preserves the
same continuation accounting while splitting SQ-v2 EV from realized residuals;
its uniform-alpha profile is an explanatory companion, while its
component-alpha TD550 profile remains predictive research. These specifications
have separate registry IDs and must not be merged under one unlabeled “Net
DECOMP” endpoint.
