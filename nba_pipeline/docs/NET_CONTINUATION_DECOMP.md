# Net-continuation (kappa) DECOMP tree

Validation-lane variant of the public five-child DECOMP shot tree that
prices every first-chance FGA at its expected remaining-possession value
instead of its gross value. Motivated 2026-08-08: a forced miss is not a
fully dead ball — the offense recovers X% of misses — so gross rim-FG
defense overstates net impact and gross SECOND_CHANCE_CLEAN conflates
opportunity (exposure to recoverable misses) with conversion (rebounding
skill).

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

## Status / boundaries

- Research/validation lane only. Production parquets, the public DECOMP
  family, the accepted VPM lane, and all frozen evidence use the gross
  tree unchanged.
- Promotion to any production surface requires its own registered
  forward gate (and for VPM, a label/mapper rebuild on net targets plus
  the EV-lane composition: EV components carry expected continuation
  `kappa * (1 - p_make)`, residual components carry the realized-minus-
  expected difference, i.e. make-luck deflated by recoverability).
