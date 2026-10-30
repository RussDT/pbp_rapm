# Fouls drawn: next-game per-100 filters

Offensive fouls drawn (charges plus non-charge offensive fouls such as illegal
screens and push-offs), charges drawn, and personal fouls drawn (PFD) now have
measured per-player-game lines from 2019-20 through 2025-26 and next-game
OU-versus-random-walk filters fitted with the same harness and protocol as
`NBA_TRACKING_NEXT_GAME.md`. Research/forecast-lane only: nothing here enters
RAPM, VPM, a prior, or a public table.

## Why a new source was needed

The repo's stats-feed play-by-play (`raw_data/NBA{YY}.parquet`) names only the
fouler on every foul row; the trailing parenthetical is the referee, and
`player2_id`/`player3_id` are null on 100% of foul rows (see the former
BLOCKED section in `NBA_ESTIMATED_SKILLS.md`). The retired v2 stats endpoint
that carried the fouled player returns empty bodies. The static NBA live CDN
feed, `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json`,
carries `foulDrawnPersonId` and `foulDrawnPlayerName` on every non-technical
foul action from 2019-20 on (2018-19 and earlier objects return a permanent
S3 `AccessDenied` 403), plus `subType` (personal / offensive / technical) and a
`descriptor` (shooting, charge, loose ball, take, double, flagrant, off-the-ball,
...). No proxy is needed; plain browser headers work.

## Ledger (`fetch_fouls_drawn_cdn.py`)

- Universe: every game in `player_game_lines.parquet` with season >= 2020
  (8,879 games, 2019-20 regular season through the 2025-26 playoffs). Eight
  threads, retries, resumable `game_status.csv`; a 403/404 is recorded as
  `unavailable`, never as zero fouls.
- Result: 8,613 games fetched, 266 unavailable (all 2019-20: 266 of its
  1,142 games are missing at the CDN). Two 2019-20 objects (0021900777,
  0021900795) are stored as Python-repr dicts with single quotes rather than
  JSON; the fetcher falls back to `ast.literal_eval` for those.
- Ledger: 347,534 foul actions with game, period, clock, fouler, drawer,
  sub-type, descriptor, qualifiers, official, and description. Descriptor
  vocabulary drifts (`loose ball` / `loose-ball` / `looseball`,
  `off-the-ball` / `offtheball`) and is normalized downstream.
- Definitions: **offensive fouls drawn** = fouls with `subType == offensive`
  (charges carry `descriptor == charge`; illegal screens, push-offs, and other
  non-charge offensive fouls carry no descriptor or `off-the-ball`; the feed
  cannot split the non-charge kinds further). **Charges drawn** = the
  `charge` subset. **PFD** = every non-technical foul drawn. Technicals name no
  drawer and are not PFD. Overall counts: offensive/no-descriptor 18,597,
  offensive/charge 7,565, offensive/off-the-ball 2,611, personal/shooting
  170,340, personal/no-descriptor 103,863, personal/loose ball 20,641,
  personal/take 10,949, personal/double 268, technicals about 10,000.

## Reconciliation (`build_fouls_drawn_game_lines.py`)

Every game is reconciled before any zero is written. Three independent parses
of the same official feed are compared:

1. **Internal identity.** Every non-technical foul names exactly one drawer,
   the game's drawn total equals its committed total, and no drawer lies
   outside the game's PBP player universe.
2. **Committed side versus repo PBP.** The CDN per-player committed count
   must equal `pf_committed` in `player_game_lines.parquet` (an independent
   parse of the stats feed). The repo counter reads the `(Pn)` ledger tag,
   which the modern neutral row `Foul : Double Personal - A (n PF), B (m PF)`
   does not carry, so a player's CDN count may exceed the PBP count by at
   most their CDN double fouls; any other gap fails this check.
3. **Official box.** The local leaguedash game-report archive
   (`player_sheets/game_report/{season}/{game_id}.csv`) must agree with the
   CDN on both `PF` and `PFD` for every player it shares with the game's PBP
   universe. The archive files traded players under their later team, so a
   file can carry a player from another game that day and omit one who
   played; only shared players are compared, and every fetched game has an
   archive file.

A game is **measured** when identity holds and either the PBP committed check
passes with the archive agreeing, or the PBP check fails but the archive
agrees on every player with any CDN foul activity (the box arbitrates a
repo-PBP parse gap such as a missing foul event; this happened once, in
2025-26). A game that was not fetched or fails any check is unmeasured for
every player in it (values NaN, never zero). Players in a measured game who
drew no foul are genuine zeros. The rule string is written to
`reconciliation.json`.

| Season | Games | Measured | Unavailable | Box mismatch | Identity fail | PFD / team-game | Off. fouls drawn | Charges | Non-charge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019-20 | 1,142 | 859 | 266 | 10 | 7 | 20.97 | 2.11 | 0.64 | 1.47 |
| 2020-21 | 1,165 | 1,162 | 0 | 3 | 0 | 19.38 | 1.68 | 0.52 | 1.16 |
| 2021-22 | 1,317 | 1,309 | 0 | 8 | 0 | 19.76 | 1.69 | 0.46 | 1.23 |
| 2022-23 | 1,314 | 1,302 | 0 | 12 | 0 | 19.94 | 1.91 | 0.50 | 1.41 |
| 2023-24 | 1,312 | 1,299 | 0 | 13 | 0 | 18.73 | 1.56 | 0.40 | 1.16 |
| 2024-25 | 1,314 | 1,307 | 0 | 6 | 1 | 18.73 | 1.40 | 0.33 | 1.08 |
| 2025-26 | 1,315 | 1,305 | 0 | 10 | 0 | 20.00 | 1.48 | 0.30 | 1.18 |

Totals: 8,543 of 8,879 games measured (96.2%); 266 unavailable; 62
`archive_box_mismatch` (post-game stat corrections where the official box and
the repo PBP agree with each other against the frozen live feed; left
unmeasured); 8 `drawer_identity_mismatch` (2020 bubble games with offensive
fouls that name no drawer, and one 2024-25 "TEAM foul" row). Committed-side
agreement with the repo PBP on offensive fouls is >= 0.9999 and on charges
>= 0.9993 in every season. Nine March 2020 games with truncated CDN feeds are
rejected by check 2. Output: `fouls_drawn_game_lines.parquet` (207,178 rows,
199,479 measured) with `pfd`, `off_fouls_drawn`, `charges_drawn`,
`off_fouls_drawn_noncharge`, shooting / loose-ball / off-ball splits, the CDN
committed side, `fouls_drawn_is_measured`, and `fouls_drawn_reason`; plus
`game_reconciliation.csv` and `reconciliation.json`.

## Protocol (registered before fitting)

**Panel.** `build_tracking_next_game.py prepare --family fouls_drawn` joins
the game lines exactly on player, season, season type, and game date to the
resolved PBP `player_game_lines.parquet` for exposure (offensive possessions
for PFD, defensive possessions for offensive fouls and charges), with the
same minutes-agreement and unique-key gates as the tracking family (0
ambiguous rows here). Rows in unmeasured games are unscored, never zero.
Coverage by measured minutes: 0.752 in 2019-20 (the CDN gap), 0.990-0.997 in
every later season; valid rows per season 18,124 / 24,828 / 27,727 / 27,317 /
27,783 / 27,910 / 28,330 for 2020-2026 (`source_audit.json`).

**Objective, models, tuning, selection, uncertainty.** Identical to
`NBA_TRACKING_NEXT_GAME.md`: next-game exposure-weighted per-100 MSE with
equal season weights; next-game random walk (5 parameters) versus next-game
two-component OU with a common attractor (6 parameters), log-space DE with
seed 20260912, 60 iterations, population multiplier 15; predictions floored at
zero; `next_ou` only if its development gain over the random walk exceeds one
paired game-cluster bootstrap SE, frozen before stress; 2,000 paired
game-cluster bootstrap draws, player-cluster and >=200-prior-possession
sensitivities, per-season wins; strictly prior naive baselines.

**Split.** Data start in 2019-20, so the split shifts one season later than
the tracking family: 2020 warm-up (mean rate, warm-up variance, mean exposure
from all 2020 rows); tune and select on 2021-2022; 2023-2025 frozen
historical stress (fresh for these stats); 2026 filtering and export only;
2027 untouched. The harness's 2026/2027 guards apply unchanged.

## Completed run (September 14, 2026)

Development windows hold 52,305 rows (offensive fouls, charges) and 52,285
rows (PFD) across 2,471 games. Mean 2020 rates per 100: offensive fouls drawn
0.418, charges 0.127, PFD 4.167.

| Stat (per 100) | Dev OU vs RW | Gate (delta / SE) | Stress rows / games | Stress MSE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Offensive fouls drawn (def) | +0.062% [-0.007, +0.131], 2/2 | 1.78 SE | 82,717 / 3,908 | 0.6800 | 0.8246 |
| Charges drawn (def) | +0.032% [-0.024, +0.091], 2/2 | 1.08 SE | 82,717 / 3,908 | 0.1803 | 0.4246 |
| PFD (off) | +0.287% [+0.171, +0.400], 2/2 | 4.79 SE | 82,708 / 3,908 | 7.7456 | 2.7831 |

All three selected `next_ou`. PFD cleared the gate comfortably; offensive
fouls drawn and charges drawn cleared it narrowly, with development intervals
that cross zero. The selection was frozen before stress and is not switched.

Frozen 2023-2025 stress, selected OU versus each comparator on identical rows
and exposure weights (percent next-game MSE improvement, game-clustered 95%
interval, per-season wins):

| Stat | vs next-game RW | vs trailing ten | vs season-to-date | vs career |
| --- | ---: | ---: | ---: | ---: |
| Offensive fouls drawn | +0.116% [+0.038, +0.194], 3/3 | +7.77% [+7.20, +8.37], 3/3 | +7.99% [+7.18, +8.84], 3/3 | +1.75% [+1.41, +2.11], 3/3 |
| Charges drawn | +0.116% [+0.040, +0.202], 3/3 | +8.47% [+7.43, +9.53], 3/3 | +7.94% [+6.68, +9.37], 3/3 | +1.93% [+1.15, +2.79], 3/3 |
| PFD | +0.348% [+0.244, +0.446], 3/3 | +6.55% [+6.16, +6.96], 3/3 | +6.08% [+5.62, +6.59], 3/3 | +5.86% [+5.44, +6.25], 3/3 |

Reading the table:

- **All three OU selections hold on stress** with pooled game-clustered
  intervals above zero and three season wins out of three. Player-clustered
  intervals agree (offensive fouls +0.116% [+0.037, +0.200]; charges +0.116%
  [+0.007, +0.255]; PFD +0.348% [+0.252, +0.446]), and the >=200 prior
  possession sensitivity moves each by less than 0.01 points.
- **The narrow development gates for offensive fouls and charges did not
  reverse**, unlike contested shots in the tracking family, but the
  per-season stress intervals are individually weak: offensive fouls
  2023 +0.087% [-0.043, +0.219], 2024 +0.138% [-0.008, +0.281], 2025 +0.128%
  [+0.000, +0.259]; charges 2023 +0.024% [-0.099, +0.152], 2024 +0.215%
  [+0.069, +0.369], 2025 +0.125% [-0.006, +0.256]. Treat OU and the random
  walk as near-equivalent for these two stats; the evidence that matters is
  that both beat the naive baselines by 7-8% (trailing ten, season-to-date)
  and 1.4-3.4% (career) in every stress season.
- **PFD is the strong result**: +0.335%, +0.337%, +0.373% over the random
  walk by season, all with intervals above zero, and +5.1% to +6.5% over
  career in every season. It is a high-count stat (about 4.2 per 100
  offensive possessions) driven by rim pressure and drives, so it shares
  information with FTA, rim attempts, and drives rather than adding a new
  dimension.
- **Charges are rare** (0.13 per 100 in 2020, declining to about 0.30 per
  team-game by 2025-26) and career is a strong baseline for them (+1.9%
  gain only). Offensive fouls drawn including illegal screens are about
  three times as common and behave similarly.
- Gains versus the random walk are small in percent for the usual reason:
  single-game rows are mostly sampling noise. Paired season wins and
  intervals are the evidence.

All six optimizers hit the 60-iteration budget without formal convergence
(random walk about 12 s per stat, OU about 37 s, single-threaded BLAS); as in
the tracking and PBP lanes the `(s_mu, s_d, q_off, r)` scale is unidentified
for predictions, so the flag is cosmetic. Selected OU parameters are in each
stat's `frozen.json`.

## Current states

`current_2026.csv` per stat holds 582 players with a 2026 observation (last
measured date June 13, 2026): last measured date, the selected model's
postgame per-100 state, observed games, and career exposure. Leaders with at
least 1,000 career possessions of the relevant kind:

- Offensive fouls drawn per 100 defensive possessions: Toumani Camara 1.62,
  Jamal Shead 1.45, Sidy Cissoko 1.33, Garrison Mathews 1.30, Elijah Harkless
  1.28 (34 games), Nickeil Alexander-Walker 1.15.
- Charges drawn per 100 defensive possessions: Jaylin Williams 0.75,
  Garrison Mathews 0.66, Moritz Wagner 0.65, Brandin Podziemski 0.54, Marcus
  Smart 0.47, Jalen Brunson 0.45.
- PFD per 100 offensive possessions: Giannis Antetokounmpo 13.09, Victor
  Wembanyama 10.09, Joel Embiid 10.00, Shai Gilgeous-Alexander 9.26, Deni
  Avdija 9.18, Devin Booker 9.06.

These are last-game filtered estimates, not projections advanced to a later
date, and the latent variances are not calibrated intervals.

## Deliverables and reproduction

```bash
.venv/bin/python nba_pipeline/scripts/fetch_fouls_drawn_cdn.py --workers 8   # ~9 min for 8,839 games; resumable
.venv/bin/python nba_pipeline/scripts/build_fouls_drawn_game_lines.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  nba_pipeline/scripts/build_tracking_next_game.py prepare --family fouls_drawn
for s in off_fouls_drawn charges_drawn pfd; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
    nba_pipeline/scripts/build_tracking_next_game.py tune --stat "$s" &
done; wait
for s in off_fouls_drawn charges_drawn pfd; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
    nba_pipeline/scripts/build_tracking_next_game.py evaluate --stat "$s" &
done; wait
.venv/bin/python nba_pipeline/scripts/build_tracking_next_game.py summary --family fouls_drawn
.venv/bin/python -m pytest nba_pipeline/tests/test_fouls_drawn_game_lines.py nba_pipeline/tests/test_tracking_next_game.py -q
```

Run the stats as separate single-threaded-BLAS processes, never a fork pool.
A named `--stat` resolves its own family and output root; `--family` picks
the root for `prepare` and `summary`.

Generated files (gitignored):

- `results/estimated_skills/fouls_drawn/`: `foul_ledger.parquet`,
  `game_status.csv`, `fetch_summary.json`, `fouls_drawn_game_lines.parquet`,
  `game_reconciliation.csv`, `reconciliation.json`.
- `results/estimated_skills/fouls_drawn_next_game/`: `panel.parquet`,
  `source_audit.json` (input hashes, coverage by stat and season), and per
  stat `frozen.json`, `stress_report.json`, `development_cells.parquet`,
  `stress_cells.parquet`, `predictions.parquet`, `current_states.csv`,
  `current_2026.csv`, plus `summary.md` / `summary.json`. Each `frozen.json`
  records SHA-256 hashes of the harness, both engines, the deflections
  harness, and the panel; `evaluate` refuses to run if any changed.

Tests: `tests/test_fouls_drawn_game_lines.py` (five tests) checks that
technicals never count, that non-charge offensive fouls and charges split
correctly, that a game is zero-filled only when identity, the repo-PBP
committed check (with the double-foul tolerance), and the shared-player box
check pass, that the archive arbitrates a repo parse gap only when it covers
every active player, and that misfiled archive players are ignored.
`tests/test_tracking_next_game.py` (nine tests) covers the shared harness,
including the fouls-drawn registry (exposure bases, measured flag, shifted
split) and that reconciled zeros are observations while unmeasured games are
never scored.

## Caveats and next steps

- **2019-20 is thin** (859 of 1,142 games; 75% of minutes). It is only
  warm-up here, but any lane that wants 2018-19 or the missing 2019-20 games
  needs another source (pbpstats game logs carry offensive fouls drawn; not
  started, owner call).
- **The 62 stat-correction games and 8 identity failures stay unmeasured.**
  They could be recovered from the official box for PFD alone, but not for
  the offensive-foul and charge splits, so they are left out of all three
  stats for consistency.
- **Non-charge offensive fouls cannot be split** into illegal screens versus
  push-offs versus other; the feed has no descriptor for them.
- **PFD overlaps** FTA, rim attempts, and drives; it is a next-game count
  filter, not an independent skill. Offensive fouls drawn and charges are the
  new information the owner asked for.
- Plain OU with no context. The lab winners (multiplicative opponent
  context, the playoff term, team-change jump, cross-skill form) are the next
  step, and the opponent term is the obvious first candidate for a stat
  where the fouling team matters; but 2023-2025 is now spent for these stats,
  so a fouls-drawn lab must declare 2021-2025 as development and lean on 2027.
- 2026 has been filtered and exported but never scored; 2027 is untouched.
