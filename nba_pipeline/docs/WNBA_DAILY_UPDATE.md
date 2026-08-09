# WNBA Daily Metric Update

`wnba_test/update_wnba_metrics.py` is the one-command current-season update for
the preferred factor-specific box-prior WNBA impact metric.

## Operator Command

From the repository root:

```bash
.venv/bin/python wnba_test/update_wnba_metrics.py --as-of "$(date +%F)"
```

Inspect the intended stages without fetching or writing:

```bash
.venv/bin/python wnba_test/update_wnba_metrics.py \
  --as-of "$(date +%F)" \
  --plan-only
```

The executable update accepts only today's date because the WNBA CDN scan
represents the current completed-game state, not a historical as-of snapshot.
The frozen daily lane currently supports the 2026 regular season.

## What It Does

1. Copies the current normalized season into staging.
2. Scans the official WNBA liveData CDN and appends newly completed games.
3. Reconstructs lineups from boxscore starters and substitutions, then rejects
   promotion if any 2026 game has an incomplete five, duplicate/cross-side
   player, or live actor absent from the recorded team lineup.
4. Force-refreshes the current WNBA PBP Stats player-totals cache used for
   public box features and expected FT/3P inputs.
5. Rebuilds `RAPM`, `TS`, `REB`, `TOV`, `FIRST_CHANCE`, and
   `SECOND_CHANCE_CLEAN` current-season CSVs.
6. Trains each factor-specific box model only on 2022-2025 and scores the
   current 2026 box rows. A player is eligible for the current-season box prior
   when both PBP Stats offensive and defensive possessions are positive.
   Reported minutes are diagnostic only because the 2026 totals feed contains
   nonpositive minutes for players with substantial valid possession exposure.
7. Solves the single-season prior-centered factor board with the frozen
   pre-2026 selection: offense alpha `5000`, defense alpha `40000`, prior scale
   `1.5`.
8. Validates player uniqueness, finite values, offense/defense/net closure,
   prior and lineup-update closure, the six shot-zone atoms, FT frequency plus
   severity, and turnover-loss children.
9. Promotes the normalized source, clean source, and stable current board only
   after every gate passes.

The update never reruns the 2024-2025 selection grid and refuses a frozen
selection report whose tuning folds contain 2026 or later.

## Outputs

- Stable board:
  `wnba_test/Results/current/wnba_current_metrics.csv`
- Stable latest manifest:
  `wnba_test/Results/current/latest_manifest.json`
- Self-contained results review:
  `wnba_test/Results/current/wnba-results-review.html`
- Dated manifest and command logs:
  `wnba_test/Results/daily_updates/YYYY-MM-DD/`
- Dated solver output:
  `wnba_test/Results/clean_decomp_26_rs_a5000_40000_prior1.5_box_trueprior_daily_YYYYMMDD/`

Generated CSV, JSON, and log artifacts remain local and gitignored.

Build or refresh the sortable review table after a current-board update:

```bash
.venv/bin/python wnba_test/build_wnba_results_review.py
```

The page keeps player, impact, offense, and positive-good defense together,
then shows the raw box-prior net/offense/defense predictions before the frozen
`1.5` scale. A local audit toggle adds the centered effective prior and 2026
lineup update.

On a processing or validation failure, the previously promoted current board
is left intact and the six processed CSVs are restored from the run-local
backup. The dated failure manifest records the failed command and retained log.

## July 29, 2026 Run

The first end-to-end run promoted five new completed games:

- source: `202` to `207` games, `99,389` to `101,799` event rows;
- latest game id: `1022600207`;
- lineup gate: all `207` games clean, with zero missing-player,
  duplicate-side, cross-side, or live-actor issues;
- processed standard possessions: `34,586`;
- board: `222` players;
- box-prior sources after repairing the minutes gate: `221` current-season and
  `1` zero fallback;
- PBP Stats exposure audit: `55` players have nonpositive normalized minutes
  despite positive offensive and defensive possessions; they remain eligible
  from their possession exposure and are listed in the manifest;
- maximum published additive gap: `0.001`, the expected three-decimal CSV
  rounding tolerance.

Before this repair, the bad minutes gate misclassified `31` active players as
previous-season fallbacks and `24` as zero fallbacks; one player remains a
legitimate zero fallback. Napheesa Collier was one of the affected players: the
repaired build uses her 2026 box prediction
(`+0.588` net), producing an effective prior of `+0.915` and a final impact of
`+0.847` over `205` lineup possessions.
