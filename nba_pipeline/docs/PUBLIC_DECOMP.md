# Public Eight-Component DECOMP

This is the active public decomposition. The older `weighted_factors_alt3_*`
and `weighted_factors_alt3_efg_value_*` families remain audit surfaces.

## Public tree

The eight player-facing components are:

1. `DECOMP_RIM_FREQ`
2. `DECOMP_RIM_FG`
3. `DECOMP_MID_FG`
4. `DECOMP_THREE_FREQ`
5. `DECOMP_THREE_FG`
6. `ALT_FT`
7. `ALT_TOV_VALUE`
8. `SECOND_CHANCE_CLEAN`

The first five are aliases over columns in the physical `FIRST_CHANCE`
parquet. `ALT_FT` and `ALT_TOV_VALUE` are aliases over that same file;
`SECOND_CHANCE_CLEAN` is physical.

## Source-row algebra

Let `r`, `m`, and `t` be the loaded file's mean first-chance points per rim,
midrange, and three-point attempt. On a first-chance FGA:

- rim frequency is `r - m` on rim attempts, else zero;
- rim FG is `actual - r` on rim attempts, else zero;
- mid FG is `actual` on midrange attempts and `m` on rim/three attempts;
- three frequency is `t - m` on three-point attempts, else zero;
- three FG is `actual - t` on three-point attempts, else zero.

On an FT event, the foul-type-aware replacement-shot baseline lives in
`DECOMP_MID_FG`; the other four shot children are zero. On non-scoring admin
events, all five are zero. Therefore, on every source action and every
processed possession:

```text
DECOMP_RIM_FREQ
+ DECOMP_RIM_FG
+ DECOMP_MID_FG
+ DECOMP_THREE_FREQ
+ DECOMP_THREE_FG
= DECOMP_EFG
= FC_EFG_Diff
```

The processor raises if either action-level or possession-level closure
exceeds `1e-9`. `FC_DECOMP_MID_VALUE_Diff` is a zero-valued reserved surface;
the private VPM EV-target attachment lane replaces it with the midrange
actual-minus-EV residual.

## Turnover rows and centering

The seven first-chance aliases use the mean of their own non-turnover values
from the exact loaded sample on turnover rows. Time-decay runs use the same
time-decay weights to compute those means. Because mean, turnover imputation,
ridge, and same-side possession centering are linear when the design and
alphas match, the parent/child identities survive through the player table up
to solver and CSV rounding.

Always prove the processed-row identity before interpreting coefficient gaps.
The public builder retains both the direct parent solve and the five-child sum
as audit columns.

## Build commands

Reprocess `FIRST_CHANCE` after pulling these changes, then solve one window:

```bash
python nba_pipeline/scripts/rapm.py DECOMP_RIM_FREQ 22 26 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
python nba_pipeline/scripts/rapm.py DECOMP_RIM_FG 22 26 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
python nba_pipeline/scripts/rapm.py DECOMP_MID_FG 22 26 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
python nba_pipeline/scripts/rapm.py DECOMP_THREE_FREQ 22 26 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
python nba_pipeline/scripts/rapm.py DECOMP_THREE_FG 22 26 ALL --rubberband --season-effects --off-alpha 2000 --def-alpha 4000
```

The rolling orchestrator solves all required parents and drilldowns and emits
the exact 97-column published schema:

```bash
python nba_pipeline/scripts/run_decomp_rolling.py \
  --windows 2022-2026
```

The default transfer manifest is a JSON mapping:

```json
{
  "22_26": 1.1471565039218607
}
```

The recovered manifest lives at
`nba_pipeline/config/decomp_tov_sc_transfer_multipliers.json` and is the
runner default. It contains the 23 windows that received the July 10
allocation in downstream publication history. The example value reproduces
the 2022-26 artifact published on 2026-07-10; it is not a universal constant.
The July 10
`full_possession_v1` policy multiplies the parent and both turnover children
by the window's sample-specific multiplier and moves the exact delta out of
`SECOND_CHANCE_CLEAN`, preserving total RAPM. The rolling builder refuses to
silently use `1.0`; `--allow-unallocated-tov` is an explicit audit escape
hatch.

The daily job rebuilds and publishes the current 1Y-5Y DECOMP files. It does
not run the former `sync_alt3_decomp_to_scposs.py` mapper: that external
script was written for the Alt3 schema and must be audited against the new
97-column DECOMP schema before player `csvs/scposs` files are refreshed.

## Stabilized RAPM and VPM

The private `alt3_3factor_research` repo is the canonical research package for
the stabilized-RAPM and VPM lanes. It contains configs, validation evidence,
EV-target scripts, and intern handoffs. Those scripts intentionally depend on
this repo's `rapm.py`, processed parquets, and local EPM snapshot tooling; the
private package is not a standalone replacement for this production tree.

The production dependency recovered here is the DECOMP source and alias
contract plus `scripts/tune_forward_component_alphas.py`, the shared loader and
target-construction module imported by the private stabilized/VPM scripts. A
private no-prior validation command can be launched without copying code:

```bash
PYTHONPATH=/path/to/pbp_rapm/nba_pipeline/scripts \
  python /path/to/alt3_3factor_research/stabilized_rapm/scripts/run_stabilized_sum_eval.py \
  --help
```

The original EPM snapshot fetcher and `build_component_epm_prior.py` were not
present in either GitHub repo, so new prior generation still requires the
surviving local EPM cache/tooling or a deliberate reimplementation. Existing
private configs, prior manifests, and evidence remain usable. Do not copy the
private VPM shot data into this public repo: some of it is derived from
licensed ShotQuality data.

## Recovery provenance

This implementation was reconstructed on 2026-07-13 from:

- `alt3_3factor_research/docs/ANDREW_HANDOFF_20260707.md`;
- the stabilized-RAPM and VPM replication docs and target-attachment code;
- downstream `rapms` publication history through 2026-07-10;
- the exact 97-column public output schema and numeric July 10 TOV/SC delta;
- the surviving May first-chance event algebra in this repo.

No recovered research result was re-selected or recomputed during this
restore. The private repo remains authoritative for research claims.
