# RAPM Table Registry

The RAPM registry is the machine-readable inventory for every table we may
eventually expose through an API. It separates a table's basketball meaning,
model status, artifact status, release window, and exact model specification.
The checked-in source is `nba_pipeline/config/rapm_table_registry.json`.

The generated snapshot consists of:

- `nba_pipeline/results/rapm_registry/rapm_table_registry.json`
- `nba_pipeline/results/rapm_registry/series_status.csv`
- `nba_pipeline/results/rapm_registry/mechanism_status.csv`
- `nba_pipeline/results/rapm_registry/current_mechanism_status.csv`
- `nba_pipeline/results/rapm_registry/artifacts.csv`

Generated registry files are operational artifacts and remain out of git.
The generated snapshot records per-format counts. Net release families are
`COMPLETE` only when every declared cut exists in both CSV and parquet form.

## Product families

| Series ID | Meaning | Role |
| --- | --- | --- |
| `parent_rapm` | Total adjusted offense, defense, and net impact | Current parent |
| `net_decomp` | Actual-outcome continuation-priced decomposition | Forward explanatory product |
| `net_decomp_sq_resid` | The same net architecture with SQ-v2 EV and realized residuals separated | Forward explanatory companion |
| `net_vpm_sq_resid_td550` | SQ-v2 net components with frozen component alphas and TD550 | Predictive research; not the explanatory parent |
| `gross_decomp` | Former actual-points DECOMP | Compatibility and rollback |
| `alt3` | Old Alt3 / Alt3 EFG-value architectures | Legacy audit only |

The mechanism inventory is a different namespace. It includes team-factor,
rim/mid/three frequency, rim/mid/three conversion, free-throw, passing,
turnover, rebound, ShotQuality, playtype, transition, STOP, block recovery,
shooter-OREB, passer/receiver, on/off-ball, and shooter/nonshooter role RAPMs.
They are diagnostics in their own target units and must not be summed into the
parent impact result.

### Current mechanism candidate

The owner-selected current mechanism candidate is `current_mechanisms_td550_no_prior`.
It uses regular season plus playoffs, smooth clock-aware rubberband adjustment,
season effects, no player prior, and a 550-day half-life. Offense and defense
ridge penalties are selected separately for each target by five-fold
game-grouped cross-validation. Training and validation loss retain the TD550
weights. The checked-in broad grid is `10` through `70000`; a selected endpoint
is a boundary warning and must be expanded before reader publication.

Repo-native frequency, conversion, rebound, passing, and dunk mechanisms have
long historical inputs. Full-definition STOP is now rebuilt for 2014-2026; its
1997-2013 precursor is turnover-only because the old raw feed lacks opposing
blocked-shot descriptions. Exact provider `PLAYTYPE_PROXY_PTS`,
`TRANSITION_FREQ`, and `TRANSITION_RIM` remain 2026-only. Separately named
1997-2026 `_BACKCAST` series now provide compatible historical transition and
playtype targets without impersonating that provider history. See
[Historical Mechanism Backfills](./HISTORICAL_MECHANISM_BACKFILLS.md). The
models are lineup-adjusted mechanism context in their own units; validation is
not permission to use a result as a VPM prior.

## Release contract

Reader/API rows use the logical key:

```text
series_id + variant_id + start_year + end_year + season_type + player_id
```

Every table also records model status separately from artifact status. A valid
research method can have a missing current artifact; a complete historical
archive can remain research-only. Reader-facing defense is positive-good.

The owner-expanded historical contract is every rolling 1Y, 2Y, 3Y, 4Y, and
5Y cut available from 1997 through 2026. This is 140 unique windows per Net
series: one cut ending in 1997, two ending in 1998, three ending in 1999, four
ending in 2000, and five per end season from 2001 through 2026. Actual Net and
the uniform-alpha SQ explanatory companion use common parent alphas
`2000/4000`. The TD550 SQ version uses frozen component-specific VPM penalties
and retains its research label.

The full-grid expansion completed on 2026-09-05 with 140 CSV/parquet pairs in
each series: 420 boards total. The registry reports all three Net families
`complete`. All 15 current TD550 mechanism models also completed with grouped-
game CV receipts and no final alpha-grid boundary warnings. This establishes
correct held-out-game computation, not final reader validation: the current
implementation chooses the exact CV minimum, centers the target once on the
full sample before splitting, and has no chronological or coefficient-stability
gate. Keep these alphas development-only until train-fold centering, paired
one-standard-error selection, and rolling next-season stability are audited.
The first rolling future screen is now complete for eleven repo-native targets:
trailing-five TD550 fits predict non-overlapping future two-season possession
blocks. It rejects the very low frequency/rebound pairs and selects `1000/1000`
for rim, midrange, and three frequency plus rebound rate. See
[Mechanism RAPM Future-Alpha Validation](./MECHANISM_FUTURE_ALPHA_VALIDATION.md).
An expanded offense-by-defense grid is also complete on four modern cuts for
full-definition STOP and the three compatible historical backcasts. It selects
`3000/3000` for STOP and transition frequency, `3000/10000` for transition-rim
by the conservative one-SE rule, and `3000/10000` for playtype proxy points.
This is development evidence and has not replaced the current board alphas.
All broader mechanism definitions remain complete for their declared
availability. Provider-only fixed-window gaps remain explicit; the new
compatible historical lanes have distinct registry IDs and filenames.

## Build commands

Preview the entire queue:

```bash
.venv/bin/python nba_pipeline/scripts/run_rapm_overnight.py --dry-run
```

Launch it detached:

```bash
.venv/bin/python nba_pipeline/scripts/launch_rapm_overnight.py
```

Build the complete historical 1Y-5Y grid:

```bash
.venv/bin/python nba_pipeline/scripts/run_rapm_overnight.py \
  --full-window-grid --release-end-year 2026 \
  --workers 2 --cores-per-rapm 2
```

The launcher writes `nba_pipeline/results/rapm_overnight/launch.json` and a
top-level log in `nba_pipeline/logs/rapm_overnight/`. The worker updates
`run_state.json` after every stage and writes a separate stage log. Re-running
without `--force` reuses complete outputs. Independent stages continue after a
failure; dependent historical SQ solving is skipped if historical target
attachment fails.

Build only the metadata snapshot with:

```bash
.venv/bin/python nba_pipeline/scripts/build_rapm_table_registry.py
```

Preview or run the current TD550 mechanism board with:

```bash
.venv/bin/python nba_pipeline/scripts/run_mechanism_rapm_registry.py \
  --current-td550 --dry-run
.venv/bin/python nba_pipeline/scripts/run_mechanism_rapm_registry.py \
  --current-td550 --workers 2 --cores-per-rapm 2
```

Each solve writes its result, `*_alpha_cv.csv` receipt, and log beneath
`nba_pipeline/results/mechanism_td550_current/`. The top-level run manifest
records selected offense/defense penalties, CV split types, and whether either
selection touched the alpha-grid boundary.

Build the parallel research board using the forward-screen recommendations:

```bash
.venv/bin/python nba_pipeline/scripts/run_mechanism_rapm_registry.py \
  --future-selected --workers 4 --cores-per-rapm 2
```

This writes one organized 2021-2026 zero-prior TD550 result for each of the 15
future-screened mechanisms under
`nba_pipeline/results/mechanism_td550_future_selected/`, plus a run manifest.
The completed 2026-09-05 run has 15/15 outputs and zero skipped. It is a research
comparison board and does not silently replace the reader's existing current
view.

## Guardrails

- Net actual, Net SQ explanation, and TD550 Net SQ VPM use distinct series IDs,
  filenames, output roots, and alpha metadata.
- Required CSV and parquet counts are validated independently; one format can
  no longer mask a missing counterpart.
- Provider-era gaps are recorded as missing inputs. They are never backfilled
  by pretending a one-season file covers a multi-season window.
- Specialized role models use their dedicated runners and common-sample
  anchoring rules.
- Raw licensed/provider inputs never enter the registry artifact payload.
- The registry is an inventory and future API contract, not authorization to
  publish a table or promote a research model.

See [RAPM Table Catalog](./RAPM_TABLE_CATALOG.md) for the reader-facing
taxonomy and [Rolling Net DECOMP and Quarterly TD](./ROLLING_NET_DECOMP_AND_QUARTERLY_TD.md)
for the detailed computation contract.
