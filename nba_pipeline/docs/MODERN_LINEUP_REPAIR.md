# Modern Raw-Lineup Repair (2014+)

`scripts/repair_modern_raw_lineups.py` audits and repairs corrupt
`home_player1..5` / `away_player1..5` values in modern raw PBP parquets.

## Why this exists

NBA GameRotation responses are not uniformly complete. A corrupt game can have
missing stints, overlapping stints, reversed interval endpoints, or fewer than
five active players. Directly attaching those intervals can create missing
slots, duplicate players, cross-team overlap, and hundreds of bad event rows.

The GameRotation archive at
<https://github.com/gabriel1200/shot_data/tree/master/rotations> is useful
evidence, but it contains the same incomplete intervals for some affected
games. The repair therefore treats an archived active stint as positive
evidence only. It does not treat absence from the archive as proof that a
player was off the court.

The repair solves each team and period as a constrained substitution replay:

- substitution descriptions define lineup state changes;
- non-admin event actors are strong positive on-court evidence;
- archived GameRotation intervals are positive evidence;
- surviving unique raw-lineup players are weak positive evidence;
- every emitted state must have five unique positive players;
- home and away states may not overlap.

The name resolver handles NBA display-name collisions such as `Jay. Williams`
versus `Jal. Williams`, initials such as `M. Brown`, and suffixes such as
`Brown Jr.`. Corrected same-clock events can be placed before an outgoing
substitution when the event actor proves that ordering. Bench technical fouls
are excluded from the actor-on-court requirement.

## Canonical event order

Modern raw PBP must be sorted by:

1. `game_id`;
2. `period` ascending;
3. quarter clock descending;
4. `event_num` ascending only within the same clock.

NBA V3 can append a corrected action with a high event number at an earlier
clock. Event-number-first sorting can move time backward, inflate minutes,
misattach lineups, and alter FT/rebound/possession sequencing.

The raw fetcher, shared RAPM processor, career-teammate minute builder, and
repair script now use clock-first ordering.

The shared processor's non-partial finalizer also checks every offense and
defense slot, same-side uniqueness, and O/D overlap. The previous `D5`-only
guard could allow a row with `O5 = 0` to survive.

The fetcher and processor no longer apply slotwise in-game `ffill/bfill` to a
sparse lineup. That operation can combine different lineup states and create
duplicate players. Sparse rows remain visible for this repair tool; an
unrepaired non-partial row is dropped by processing rather than guessed.

## Commands

Audit and write repaired validation copies without touching production:

```bash
.venv/bin/python nba_pipeline/scripts/repair_modern_raw_lineups.py \
  --start-year 2014 --end-year 2026 --season-types all
```

Audit only:

```bash
.venv/bin/python nba_pipeline/scripts/repair_modern_raw_lineups.py \
  --start-year 2014 --end-year 2026 --season-types all --audit-only
```

Limit reconstruction to selected games:

```bash
.venv/bin/python nba_pipeline/scripts/repair_modern_raw_lineups.py \
  --start-year 2017 --end-year 2017 --season-types rs \
  --games 0021600148,0021600712
```

Validation outputs default to:

- repaired season parquets: `validation/modern_lineup_repairs/raw/`;
- JSON reports: `validation/modern_lineup_repairs/reports/`;
- downloaded rotation cache: `validation/modern_lineup_repairs/rotation_cache/`.

Use `--promote` only after reviewing the report. Promotion is refused when any
lineup issue row, unresolved substitution, or required event-actor mismatch
remains. Each replaced production parquet is backed up under a timestamped
`production_backup_*` report directory first.

## 2026-07-21 validation and promotion

The full local 2014-2026 RS+PS audit covered:

- 26 raw season files;
- 7,784,893 event rows;
- 31 corrupt games;
- 10,623 corrupt lineup rows before repair;
- 0 corrupt lineup rows after repair;
- 0 unresolved games.

The original validation report is
`validation/modern_lineup_repairs/reports/modern_lineup_repair_20260721_142907.json`.
The repaired files were subsequently promoted with production backups. The
post-promotion audit report is
`validation/modern_lineup_repairs/reports/modern_lineup_repair_20260721_235717.json`;
it reports zero unresolved games across all 26 production files.

The LeBron-relevant corrupt game is `0021600712` (OKC at Cleveland,
2016-17). The other two originally diagnosed 2014-2018 games were
`0021500588` and `0021600148`; neither contains LeBron.

## Processed rebuild validation

After promotion, the 2014-2026 RS+PS processed surface was rebuilt for every
locally supported registered metric. The validation report at
`validation/processed_rebuild_2014_2026/processed_rebuild_validation.json`
scans 644 parquets and 66,359,329 rows with zero missing/unreadable artifacts
and zero invalid saved lineups. The exact five-child `FIRST_CHANCE` identity
has zero row failures above `1e-10` and a maximum floating-point gap of
`5.69e-14`.

The ten licensed ShotQuality-derived families were rebuilt for 2026 RS+PS.
The local NBA24/NBA25 raw files do not contain `sq_descriptor_bundle` or
`initial_ev`, and the external enrichment CSV directory is absent, so those
families cannot be truthfully regenerated for 2024-25 from this checkout.
