# Missing-FT fallback repair

Owner instruction, 2026-09-06: update affected Net DECOMP numbers after aligning
missing shooter-season FT percentages to actual made/missed outcomes. Known FT
percentages remain expected values.

## Method

`repair_ft_decomp_results.py` applies a generalized-ridge solve to the exact
new-minus-old component target. It keeps the original possession universe,
weights, offense/defense penalties, rubberband specification, and linear
possession centering. Each shared penalty matrix is factored once for multiple
changed targets. Unchanged children are skipped. Parent RAPM is unchanged.
This is a numerical correction, not a new model selection or predictive gate.

The original coefficients, input parquets, and audit manifests are retained
under the gitignored `results/ft_actuals_repair_20260906/` directory. The job
manifests record original models explicitly: some actual-Net five-year tables
still use the legacy 24-bucket surface and must not be patched with the smooth
operator. The legacy transform is frozen with that repair receipt.

No balancing child is introduced. Existing CSV rounding remains visible in the
small residuals. A fresh 2026 FT fit agrees with the coefficient correction to
0.000499 offense / 0.000503 defense points per 100. A weighted multi-target
regression test proves the correction equals a full refit with unequal
penalties and an unpenalized nuisance feature.

## Scope and validation

- The affected 12 season/phase file pairs contain 610 mismatched possessions in
  the original 1997-2026 archive. All corrected pairs close directly to the
  loaded RAPM parent at less than 1e-9 (observed maximum about 5.7e-14).
- These seasons intersect 64 windows per rolling series: actual Net,
  common-alpha SQ Net, and component-alpha TD550 SQ Net VPM. Therefore 192
  rolling tables require repair; 228 other rolling tables remain unchanged.
- 2026 actual Net's maximum displayed net residual falls from 0.147 to 0.004
  points per 100 in the repaired live table.
- The separate 2020-2026 calendar-date no-prior VPM snapshots also depend on
  these inputs. Their partial seasons are rebuilt at the original game cutoff;
  completed-season kappa must not be filtered into a partial-season snapshot.
- Gross compatibility files remain the explicit rollback series. Historical
  research experiment scorecards are not silently rewritten by a board repair.

## Execution status

Completed 2026-09-06: 192 rolling boards (CSV/parquet pairs, component solves,
nuisance exports, manifests, and master copies) and 28 calendar snapshots were
replaced. The isolated Net and SQ input lanes were refreshed. The existing
`nba_no_prior_vpm_td550_snapshots` published table was upserted with 34,075 rows;
its row count and sampled FT values in every snapshot match the repaired local
source. All eight targeted tests pass.

The original legacy 2022-2026 actual-Net operator is preserved: its maximum
remaining parent gap is 0.061 points/100 (previously 0.196). This repair removes
the FT target discrepancy; it does not erase pre-existing legacy numerical
error by inserting a balancing term or changing solver specifications.

Generated receipts under `results/ft_actuals_repair_20260906/` include
`promotion.json`, `snapshot_promotion.json`, `live_verification.json`,
`published_verification.json`, and per-job input/operator/checksum records.
Snapshot component and three-factor identities close within 1.8e-15. Originals
are retained under `published/backup/`; the curated registry inventory is refreshed.
