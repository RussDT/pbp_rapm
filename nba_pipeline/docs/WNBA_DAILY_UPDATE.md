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
- Databallr publish files (see below):
  `wnba_test/Results/current/publish/`

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

## Publishing The Board To databallr (WNBA VPM)

Databallr labels this board **WNBA VPM** and serves it from the legacy
`/api/supabase/wnba_rapm` route (the route name is preserved for consumer
compatibility). Prepare the publish files after a complete daily update:

```bash
.venv/bin/python wnba_test/prepare_wnba_vpm_publish.py \
  --stage-databallr /Users/russ/Documents/Codex/basketball/nba/products/databallr
```

The script refuses to run unless `latest_manifest.json` reports
`status: complete`, re-validates the promoted board, and writes to
`wnba_test/Results/current/publish/`:

- `wnba_vpm_boxprior_current.csv` — the drop-in databallr file: the 61 legacy
  `wnba_weighted_factors_alt3_efg_value_*` columns in their original order
  (headline `off` / `def` / `net_rapm`, zone freq/FG atoms, FT freq/severity,
  point-valued TOV, `SC`/`FC`, display sums and residuals, possessions), then
  the box-prior audit columns (`prior_*`, `delta_*`, `has_prior`, `box_*`,
  `box_prior_source`), then `team` (PBP Stats current abbreviation,
  informational), `season`, `season_type`, `model_variant`, `qualified`
  (`possessions >= 1000`), and `as_of`. Exactly one row per player with
  `Latest_Year = season`, sorted by `net_rapm`, because the databallr ticker,
  dashboard (`limit=500`), stats table, and profile join all assume one row
  per player id. Every additive identity is re-checked to the three-decimal
  tolerance (`0.0021`) before writing.
- `wnba_vpm_boxprior_history_2022_2026.csv` — the 2022-2025 rows from
  `build_wnba_metric_history.py` plus the refreshed current-season rows,
  with per-season ranks. It has multiple rows per player and is **not**
  drop-in compatible with the current consumers; it is staged for a future
  season-aware endpoint only.
- `publish_manifest.json` — source paths and sha256s, board / publish /
  history validation summaries, top-10, the databallr checklist, and the
  consumer contract.

- `wnba_vpm_rack_payload.json` — the payload databallr's `/wnba/vpm` page
  fetches: board meta, one row per player with the published factor and zone
  fields, and the `profile` block (column keys, per-season league means, and
  each player's 2022-2026 box + VPM rows) built from the same box cache and
  history file the standalone rack reader uses. `--skip-rack-payload` omits it.

`--stage-databallr <root>` copies the current CSV, the season-history CSV,
**and the rack payload** into databallr `public/` and
`new next js migration/public/`. It never commits or deploys. Databallr's
season-aware `/wnba/stats` request reads the history artifact when `season` is
explicit; callers that omit `season` retain the one-row-per-current-player
contract. The databallr route constants, URL fallbacks, versioned Flask cache
key, and the VPM labels were switched on
August 18, 2026; the old
`wnba_weighted_factors_alt3_efg_value_21_26_all_td700_a1750_4000.csv` stays in
databallr `public/` only as the RAPM Galaxy / RAPM Twins source.

Production cutover order (databallr `main` → Netlify auto-deploys the SPA and
the `public/` CSV; the Flask API is a manual `eb deploy production-py312` from
`backend/`): (1) commit and push the CSV plus the route change so Netlify
publishes `https://databallr.com/wnba_vpm_boxprior_current.csv` — on Elastic
Beanstalk the bundle is `backend/` only, so the route reads that URL; (2) deploy
the backend from a clean worktree of the pushed commit (`backend/.ebignore`
makes the EB CLI zip the working directory, untracked files included); (3) only
then push the VPM label changes, so "VPM" never sits over the old time-decay
payload. Browser caches can hold the previous `/wnba_rapm` payload for up to an
hour (`max-age=3600`). The data + route commit is databallr `ba567e73`
(August 19, 2026); see `backend/DEPLOYMENT.md` there for the worktree recipe.

Two caches sit in front of that route, and the server one bit on this cutover.
Flask-Caching ignores `key_prefix` when `query_string=True` — the key is only
`request.path` plus an MD5 of the sorted args — so the `wnba_rapm_v2` to `_v3`
bump retired nothing, and 26 entries cached from the old CSV kept being served
per query string. A plain `?limit=5000` returned the new board while the exact
`?limit=5000&fields=...` URL the stats table requests returned the old 243-row
one, which is what "still loading normal RAPM" looked like. databallr `2cfd9471`
adds `versioned_query_cache_key()` and uses it on `/wnba_rapm`; bump
`wnba_rapm_vpm_boxprior_v1` if the published CSV's source or schema changes
again. To invalidate without deploying, SCAN-delete
`databallr:/api/supabase/wnba_rapm*` — the provider rejects `KEYS`, so
`POST /api/cache/clear/supabase` errors. `REDIS_URL` in databallr `backend/.env`
is the production Redis, so a local delete clears production too. The second
cache is the browser's own `max-age=3600` copy, per URL; a hard reload
(⌘⇧R) clears it and everyone else self-heals within the hour.

This is the box-only lane: no RTSS / Second Spectrum priors are involved, so
missing tracking games do not affect the published numbers. The RTSS-augmented
board (`build_wnba_rtss_augmented_board.py`) remains research-only.

The live reader is databallr `/wnba/vpm` (page `src/pages/WNBAVpm.tsx`,
renderer `src/pages/WnbaVpm/`, doc `fresh_docs/WNBAVpm.md`). It is a port of the
rack below, defaulting to the Arena theme, and it reads only
`public/wnba_vpm_rack_payload.json` — it never queries Flask, so the page and
the `/api/supabase/wnba_rapm` board always describe the same solve as long as
both files are staged and committed together. The payload is a mutable pointer
file: it must keep Netlify's revalidating default and must never be given
`immutable` cache headers.

Local reader / candidate work: `wnba_test/build_wnba_vpm_rack.py` renders the published CSV
into the self-contained dark "rack" page
`wnba_test/Results/current/publish/wnba_vpm_rack.html` from
`wnba_test/wnba_vpm_rack_template.html`. It embeds published fields only
(eFG value = `*ALT_EFG_VALUE`, FT, TOV value, `*SC`; four factors close to
`off` / `def` within `0.0021`), and it refuses a manifest whose `status` is not
`complete` or whose `as_of` differs from the CSV. `--emit-payload PATH` writes
the same JSON the databallr page consumes (add `--no-html` to skip the page),
which is how the publish script builds it.

The page carries a three-way colour theme switch (`Rack` default, `Arena`,
`NBA RAPM`) whose Arena / NBA RAPM token blocks are databallr's own palettes
(`DuelLobbyCards.css` / `DuelLobbyTheme.tsx`); every colour resolves through
CSS tokens, the SVG knobs and canvases read the live tokens at draw time, and
the choice persists in `localStorage`. The databallr themes use Ballr Sans,
embedded as base64 `@font-face` rules from
`nba/products/databallr/public/fonts/ballr-sans/` (`--fonts-dir`; `--no-fonts`
falls back to the system stack); the Rack theme keeps the system UI stack.
Below the rack, a **player profile** section re-sets databallr's profile
modules in the rack idiom and wires them to the board selection: pick a player
on the leaderboard and the profile follows. It replaces the earlier mock-data
"table studies" block, and every number in it is real — nothing is sample data.

- **Head** — identity, a season dropdown, a regular/playoffs split (disabled
  when the player has no playoff rows), a PG / P36 / P75 / P100 / TOT rate
  basis, and a four-group stat strip (Production, Two-way, Shooting, Impact)
  with percentile ticks.
- **Shot profile** — FT / RIM / MID / 3P by volume and efficiency, diverging
  from the league slot. Volume is attempts per 100 of the player's own on-floor
  offensive possessions; the league column pools that same ratio over every
  player, which is team attempts per 100 divided by five. Volume rows share one
  axis and efficiency rows share another, so bar length is comparable across
  zones; the span is the 5th–95th percentile gap that season, so a
  three-attempt mid-range line cannot set the axis.
- **Role & playstyle** — usage, rim rate, 3PA rate, self-created share, assist
  points, offensive rebounds, fouls drawn and lost-ball turnovers over 100
  offensive possessions; stops, steals, blocks and defensive rebounds over 100
  defensive possessions. These are this repo's own box definitions. They are
  *not* databallr's Usage / On-Ball / Shot Quality fields, which come from a
  different pipeline — only rim rate happens to coincide.
- **Career log** — one row per season with a possession-weighted career line,
  carrying the published VPM off/def/net beside the box columns.

Data comes from the repo's own WNBA box cache (`--box-dir`, default
`nba/research/wnba/data/YYYY[ps]_pbp.csv` — the same pbpstats player-season
totals the box priors read) joined to `--history`
(`wnba_vpm_boxprior_history_2022_2026.csv`). Percentiles and the shot-profile
axis are computed from that cache at render time against player-seasons
clearing 200 offensive possessions (60 in the playoffs), so they are real
league percentiles rather than illustrative ranges; the tint uses each theme's
accent above median and its secondary hue (`--cool-rgb`) below.

Two labelling rules matter here. `TS+` is `TS% − league TS%` in percentage
points — it is deliberately **not** called rTS, because databallr's `RTS` is
role-adjusted and this repo's `rTS` belongs to the `zTS = rTS + playtype_diff`
chain. And the VPM column carries the solve that season actually ran: 2022–2025
are RS+PS solves shown on the regular-season row, 2026 is regular season to
date, and playoff rows have no separate solve. The career log spans 2022–2026,
the reach of the box cache; earlier seasons are absent, not zero.

`--no-profile` strips the whole section (markup and payload alike).

## August 18, 2026 Run

- source: `242` to `266` completed games (`24` new), latest game id
  `1022600266`; lineup gate clean;
- board: `226` players, `148` with at least `1000` possessions;
- box-prior sources: `225` current-season and `1` zero fallback
  (Darianna Littlepage-Buggs, one possession);
- maximum published additive gap: `0.002` (three-decimal rounding);
- publish: `wnba_vpm_boxprior_current.csv` (226 rows) and the 887-row
  2022-2026 history file were built and staged into both databallr public
  folders, together with the `wnba_vpm_rack_payload.json` (311 KB, 226 players,
  226 profile rows) that backs databallr `/wnba/vpm`; the top of the board is A'ja Wilson `+7.24`, Olivia Miles `+4.93`,
  Jackie Young `+4.56`, Aliyah Boston `+4.44`, Veronica Burton `+3.92`.

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
