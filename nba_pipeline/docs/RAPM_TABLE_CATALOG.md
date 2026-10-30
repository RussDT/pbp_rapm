# RAPM Table Catalog and Calculation Queue

Last audited: 2026-09-04

## The Short Answer

The old Alt3 and Alt3 EFG-value tables are legacy/audit surfaces. Owner decision
2026-09-03: **Net DECOMP is the forward explanatory product.** Its accounting
of missed shots, rebounds, and turnovers is cleaner. The gross eight-component
actual-points DECOMP remains the compatibility benchmark while the 1Y-5Y net
release and consumer migration are built. This product decision does not claim
that net VPM has cleared its separate untouched predictive gate.

Do not treat every RAPM-shaped output as one product. The catalog has five
separate offerings:

1. parent impact RAPM;
2. additive impact decomposition;
3. mechanism and rate RAPMs;
4. role and creation attribution RAPMs; and
5. expected-value and predictive research surfaces.

Model trust and file freshness are separate questions. A current method can have
missing current-season artifacts, while a complete historical archive can still
be research-only.

## Status Vocabulary

Use the model statuses defined in the
[Model Assurance Ledger](./MODEL_ASSURANCE_LEDGER.md): `CURRENT`, `RESEARCH`,
`DIAGNOSTIC`, `PARKED/REJECTED`, and `RECOVERY`.

Track artifact readiness separately:

- `COMPLETE`: every declared release window and required format exists and has
  passed its identities.
- `PARTIAL`: some declared outputs exist, but the release family is incomplete.
- `MISSING`: no usable artifact exists for the declared release target.
- `STALE/UNKNOWN`: an artifact exists, but its recency or input contract has not
  been proved.

Product role is a third label when needed: `INCUMBENT`, `CANDIDATE`,
`DIAGNOSTIC`, or `LEGACY/AUDIT`. “Legacy” is not a data-quality verdict.
“Stale” is an artifact condition, not a methodology verdict.

## Offering Hierarchy

| Offering | What it answers | Model status | Artifact readiness in this checkout | Decision |
|---|---|---:|---:|---|
| Parent RAPM | How much did the player affect offense and defense? | CURRENT | PARTIAL | Keep as the top-level impact table. Normalize the current 1Y-5Y release family before publishing. |
| Gross eight-component DECOMP | Where did fitted actual-points impact come from under the former public accounting? | CURRENT compatibility surface | PARTIAL | Retain for rollback, matched comparisons, and consumers awaiting migration. Do not develop new products on it. |
| Net-continuation DECOMP | Where did impact come from after pricing expected continuation? | CURRENT for forward explanatory use; predictive VPM remains RESEARCH | PARTIAL against the expanded 140-window 1Y-5Y historical contract | This is the selected forward product. Complete the full grid, then migrate consumers explicitly. |
| VPM / Impact Monster | What player value is most likely to persist into future possessions? | RESEARCH | COMPLETE only for its declared research snapshots | Keep separate from explanatory DECOMP and label every board with its frozen config. |
| Mechanism RAPMs | How did a player affect a particular rate or tendency? | CURRENT or DIAGNOSTIC by table | COMPLETE for every declared available-current contract; provider-only multi-year gaps remain explicit | Publish as diagnostics with explicit units and sample windows; never sum them into total impact. |
| Alt3 / Alt3 EFG-value | Can we reproduce or migrate the older decomposition? | DIAGNOSTIC | PARTIAL | Its product role is LEGACY/AUDIT. Preserve it, but do not present it as the best current table. |

## Which Decomposition Is Better?

| Surface | Shot accounting | Second-chance accounting | Current role |
|---|---|---|---|
| Older Alt3 | Older three-factor construction; some variants reconstructed second chance as a residual | Schema-dependent residual behavior | Legacy/audit only |
| Alt3 EFG-value | Six atomic shot pieces plus `ALT_EFG_BASELINE` as a balancing/replacement bucket | Pre-net-continuation treatment | Immediate legacy predecessor; useful for migration checks |
| Public DECOMP | Five public shot children plus FT, turnover value, and physical second chance; exact row algebra and no balancing baseline | Actual-points second-chance component | Current public incumbent |
| Net-continuation DECOMP | Reprices misses using expected continuation and reprices turnover loss | Separates opportunity from conversion with an explicit payback ledger | Research-leading successor |

Net-continuation is now the selected explanatory architecture; gross DECOMP is
retained as a compatibility benchmark rather than deleted. On identical
development rows, the selected
net-continuation VPM mapping beat the gross mapping by `+18.43e-6` with a positive
paired interval (`[+2.19, +36.01]`). The no-prior net-versus-gross comparison was
neutral, however, and 2026-27 remains untouched. Therefore the explanatory
product decision does not promote net VPM or broaden its predictive claim. See
[Net-Continuation DECOMP](./NET_CONTINUATION_DECOMP.md).

One implementation detail causes avoidable confusion: the current public
rolling builder can still use `weighted_factors_alt3_*` as an internal base
scaffold. That filename does not make Alt3 the reader-facing product. The
consolidated `weighted_factors_decomp_*` output is the gross compatibility
artifact; the forward artifact uses `weighted_factors_decomp_net_*`.

## Mechanism and Role Inventory

These are useful tables, but they answer narrower questions than impact RAPM.

### Shot location and conversion

- rim frequency;
- midrange frequency;
- three-point frequency;
- rim field-goal percentage;
- midrange field-goal percentage; and
- three-point field-goal percentage.

### Free throws, turnovers, and possession continuation

- free-throw premium and related FT mechanisms;
- turnover and scoring-turnover value;
- bad-pass turnover value;
- offensive-rebound opportunity/value; and
- second-chance conversion/value.

### Passing, creation, and scoring role

- assisted-points parent RAPM;
- passer and receiver children;
- rim-assist diagnostics;
- scoring on-ball and off-ball children;
- shooter/nonshooter conditional TS and the joint shooter interaction; and
- overall or transition playtype-mix diagnostics.

### Defensive and context overlays

- STOP and other defensive event mechanisms;
- ShotQuality/initial-EV targets;
- shot-clock difficulty and grenade burden; and
- primary-defender and matchup research splits.

Each published mechanism table needs a label for target units, offense/defense
sign convention, season window, game scope, adjustment specification, and model
status. Expected-value or context overlays must not be branded as total impact.

## Artifact Audit

This is a checkout-level audit, not a claim about every backup or downstream
repository.

| Artifact family | Observed state on 2026-09-04 |
|---|---|
| Gross compatibility DECOMP, declared 1Y-5Y windows | `1/5` consolidated CSVs present. The 2022-2026 five-year file exists; 2026, 2025-2026, 2024-2026, and 2023-2026 are missing. |
| Actual Net DECOMP full grid | `140/140` unique cuts complete in CSV and parquet. |
| SQ-v2 plus residual explanatory Net DECOMP full grid | `140/140` cuts complete in CSV and parquet under common `2000/4000` alphas. |
| SQ-v2 TD550 Net VPM research full grid | `140/140` cuts complete in CSV and parquet. Existing ten-component and three-factor identities close at machine precision. |
| Quarterly TD550 RAPM archive | `100/100` declared snapshots complete. |
| No-prior net-continuation VPM snapshots | `28/28` snapshots and 34,075 published research rows complete. |
| Alt3 EFG-value current-window files | `0` current files observed. This is acceptable for a legacy surface unless an audit explicitly needs them. |
| Older Alt3 files | Three files observed. Retain as legacy/audit artifacts. |
| Russell ShotQuality v2 historical bridge | Historical target bridge complete for 1997-2015 and native v2 retained for 2016-2026; all 26 historical five-season TD550 no-prior SQ-v2 Net VPM windows are now complete. |

For every current 1Y-5Y cut, actual Net, common-alpha SQ Net, and TD550 SQ Net
have identical player-ID universes and identical possession exposure after
key alignment. Their row counts are 582, 687, 802, 877, and 1,030 from 1Y
through 5Y. The SQ families' ten-component and three-factor identities close
at machine precision; the actual family keeps every fitted parent/child and
top-level residual visible.

The curated machine-readable inventory contains 6 product series and 36
mechanism definitions. Its generated snapshot verifies 140 matching
CSV/parquet pairs for each of the three Net release families; all three now
report `complete`. See
[RAPM Table Registry](./RAPM_TABLE_REGISTRY.md).

The standard parent family also needs a naming and completeness audit. The exact
one-year `weighted_factors_26_all_se.csv` exists, as does a five-year RB/SE
variant, but the expected exact 1Y-6Y `_all_se` family is not complete under one
contract.

## Calculation Queue

### P0 — Migrate the completed 1997-2026 1Y-5Y Net grid

The product decision is made and the current 1Y-5Y actual and SQ-v2 plus
residual release families are calculated and validated. The owner expanded the
historical artifact contract to every rolling 1Y-5Y cut back to 1997: 140
windows per series and 420 boards total. The restartable full-grid queue is
complete; the remaining work is consumer migration:

1. migrate `csvs/scposs`, site, Supabase, and downstream `rapms` consumers;
2. verify those readers against the audited positive-good defense contract; and
3. keep the gross artifacts available for matched comparison and rollback.

The completed 140-window grid is the historical backbone. The full-grid
expansion adds every shorter historical cut without changing the
selected method, alphas, or product labels.

The current-window guard allows at most `0.30` points/100 between the separately
regularized parent and component sum. The observed 3Y maximum is `0.259` for a
289-possession player; the gap remains an explicit residual rather than being
hidden in a balancing component. Source-row, shot-tree, and turnover-split
identities retain their tighter independent checks.

### P1 — Maintain gross compatibility only where needed

The checked-in gross transfer manifest covers all current windows. Rebuild
missing gross files only for an active compatibility consumer, migration test,
or rollback bundle; do not spend release work expanding the old architecture.

### P1 — Normalize the parent release contract

Inventory which consumer expects plain `_all_se` versus `_all_rb_se_a2000_4000`
files, declare one canonical 1Y-5Y parent family, and regenerate only the missing
windows. Do not duplicate identical results under competing names without a
documented compatibility reason.

### P2 — Evaluate the completed historical ShotQuality v2 experiment

The restartable queue attached the 1997-2015 bridge to isolated net targets and
completed all 26 five-season TD550 no-prior SQ-v2 Net VPM windows. Treat the
artifacts as research calculations, including the documented 2015/2016 seam
handling; completion does not promote predictive VPM or alter the public
explanatory decision.

### P2 — Curate mechanism offerings

The curated registry now defines 36 mechanism offerings and schedules current
1Y-5Y windows only when every required processed input exists. Specialized
passer/receiver, scoring-role, shooter/nonshooter, joint-shooter, and
shooter-OREB models retain their dedicated runners. Missing provider-era
windows remain explicit gaps; do not mass-recompute every historical prototype
merely because some target parquet exists.

The current reader-facing mechanism batch is separately curated as a no-prior
TD550 board. It contains rim/mid/three frequency and FG%, team rebound rate,
assisted points, rim assists, made dunks, assisted made dunks, STOP, playtype
proxy points, transition frequency, and transition-rim frequency. Each target
gets separate offense and defense penalties from TD-weighted grouped-game CV.
The repo-native models use 2021-2026; the three provider-enriched playtype and
transition targets remain explicitly 2026-only until older inputs are rebuilt.
Do not substitute the stale playtype-TS-mix table or call these contextual rates
impact components.

The 15-model development batch completed on 2026-09-05. The first grid placed four sides at
its minimum of `100`; a targeted `10/30/50` expansion confirmed that all four
still select `100`, leaving no endpoint selections. Final offense/defense
penalties are recorded in `results/mechanism_td550_current/run_manifest.json`
and the generated `current_mechanism_status.csv`. These are exact-minimum
game-CV artifacts, not yet final publication penalties. A post-run audit found
that paired one-standard-error sensitivity would change 10 of 30 O/D sides,
and the current folds do not test chronological transport or coefficient
stability.

## Publishing Rules

Before any RAPM table is called current:

1. identify its offering family and target units;
2. record model status and artifact readiness separately;
3. declare canonical filename, runner, sample windows, and sign convention;
4. prove the relevant additive or weighted-parent identities;
5. verify all current consumer paths; and
6. update this catalog, the assurance ledger if trust changed, and the artifact
   manifest if the release contract changed.

Do not delete Alt3 or gross artifacts during this migration. Move consumers
first, preserve an audited compatibility/rollback path, and archive old files
only after no active reader depends on them.

## Canonical References

- [Public Eight-Component DECOMP](./PUBLIC_DECOMP.md)
- [Net-Continuation DECOMP](./NET_CONTINUATION_DECOMP.md)
- [Rolling Net DECOMP and Quarterly TD](./ROLLING_NET_DECOMP_AND_QUARTERLY_TD.md)
- [Model Assurance Ledger](./MODEL_ASSURANCE_LEDGER.md)
- [Artifact Manifest](./ARTIFACT_MANIFEST.md)
- [RAPM Methodology](./RAPM_METHODOLOGY.md)
