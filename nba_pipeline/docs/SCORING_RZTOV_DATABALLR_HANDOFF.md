# Scoring Impact + rZTOV: Databallr Integration Handoff

**Status (2026-09-29):** ready for Databallr integration. Every number below
reconciles with the other artifacts and with live Databallr `player_zts`.
Nothing has been published. The Supabase write and any Databallr commit or
deploy stay deliberate owner steps.

This page is the contract for moving two research readers into Databallr:

| Reader | Page (self-contained) | Method doc |
| --- | --- | --- |
| rZTOV explorer | [`rztov-explorer.html`](./rztov-explorer.html) | [RZTOV.md](./RZTOV.md) |
| Scoring impact | `historical-scoring-impact.html` (generated, untracked, 3.4 MB) | [HISTORICAL_SCORING_SECURITY.md](./HISTORICAL_SCORING_SECURITY.md) |

It also covers one research column, **Playtype mix RAPM**
(`playtype_mix_rapm`; method in
[TD_OVERALL_PLAYTYPE_MIX_OTS.md](./TD_OVERALL_PLAYTYPE_MIX_OTS.md)). It
answers the objection that zTS rewards a high-usage creator for hard shots
while the diet of the lineup around them also gets harder (the "DeRozan
effect").

A second research group, **Lineup TOV** (`lineup_*tov*`; method in
[LINEUP_TOV_BOARD.md](./LINEUP_TOV_BOARD.md)), was added 2026-09-29 for
transparency. The box lane prices (`0.90` scoring, `1.02` bad pass, 2022-26;
corrected 2026-10-19 from the ridge-shrunk `0.54` / `0.77`) price an excess
turnover, but they do not say whose turnover the lineup avoided. This group shows
what time-decayed lineup data implies:

- **The solve.** Two-season TD550 on-court team turnovers with the player on
  offense.
- **The split.** Own, teammates, and unassigned team turnovers, per lane.
- **The price.** A kept possession's net first-chance value.

It sits beside the box turnover values and never replaces or adjusts them.

## Owner decision, 2026-09-26: one mix column

The scoring board carries a single mix column. It is the full lineup value,
own diet plus teammate diet, not a split:

```
playtype_mix_rapm = mix_value100
                  = own_mix_value100 + teammate_mix_value100
                  = −mix_rapm × league_fga_per100 / 50        (pts/100, + easier)
```

It comes from a **two-season, 550-day half-life** lineup ridge solved for
**every season from 1997 through 2026**. Each feature season fits itself plus
the prior season, so the prior season carries about `2^(−365/550) ≈ 0.63`
weight. This replaces the eight-season TD700 2015-2026 sidecar and its ten
split columns. The own/teammate split stays available on the standalone mix
board for audit.

## Datasets

All four are registered in `config/supabase_publications.json` with exact
column order, row keys `(nba_id, year)`, and minimum row counts. All four pass
`publish_supabase_results.py` in dry-run mode.

| Slug | Rows | Cols | Seasons | What it is |
| --- | ---: | ---: | --- | --- |
| `historical_scoring_security_1997_2026` (schema v4) | 11,605 | 41 | 1997-2026 | scoring value, turnover lanes, `playtype_mix_rapm`, and the lineup TOV view |
| `nba_rztov_explorer_seasons_2014_2026` | 5,370 | 25 | 2014-2026 | observed rZTOV ledger with lane point prices |
| `td_playtype_mix_board_1997_2026` (schema v2) | 14,565 | 17 | 1997-2026 | standalone two-season TD550 mix board, with the exact own/teammate split |
| `lineup_tov_board_1997_2026` (schema v1) | 14,583 | 30 | 1997-2026 | standalone two-season TD550 lineup turnover board: saved/100 by lane and piece, raw rates, price |

Schema v3 of the scoring board drops the ten v2 mix columns and appends one,
`playtype_mix_rapm`. The first 27 columns are unchanged from v2. The mix board
slug was renamed from `td_playtype_mix_board_2015_2026`; its 17 columns keep
their names, but every value changed with the new spec.

Schema v4 keeps the first 28 v3 columns unchanged and appends 13 lineup
columns. The eleven values are priced in pts/100:
`lineup_{scoring,badpass}_tov_{own,teammate,team,total}_value100`, the two
`lineup_{scoring,badpass}_tov_raw_value100`, and `lineup_tov_value100`. The
other two are `lineup_tov_point_price` and the `lineup_team_tov_coverage`
label.

### Scoring board columns a reader needs

| Column | Meaning | Units / sign |
| --- | --- | --- |
| `tsa100`, `rts`, `role_difficulty`, `zts` | production zTS chain inputs; `zts = rts + role_difficulty` | TS points |
| `scoring_value100` (SV/100) | `-1.211295 + 0.800618 * 2*tsa100*(zts+4)/100`; equals Databallr `player_zts.pred_ots_m4_tsa25fit` | pts/100, + good |
| `rztov`, `sc_excess`, `bp_excess` | mix-adjusted turnover rate and lane excess per 100 | − good |
| `scoring_tov_point_price`, `badpass_tov_point_price` | per-lane, per-window net-continuation price of one excess turnover | pts per TOV |
| `scoring_tov_value100` | `-sc_excess * scoring price` | pts/100, + good |
| `badpass_tov_value100` | `-bp_excess * bad-pass price` | pts/100, + good |
| `scoring_impact_no_badpass100` | SV + scoring-TOV value (**default sort**) | pts/100, + good |
| `turnover_value100` | scoring-TOV + bad-pass-TOV value | pts/100, + good |
| `scoring_security_value100` | SV + turnover value | pts/100, + good |
| `role_source`, `turnover_source` | provenance: observed vs backcast role; `observed` (2014+) vs `box_estimated` TOV | label |
| `qualified_scoring_security` | 1 when OffPoss ≥ 1,000 | flag |
| `playtype_mix_rapm` | full lineup shot-diet value with the player on offense (own + teammate diet), two-season TD550 | pts/100, + easier, − harder |

| `lineup_{lane}_tov_{own,teammate,team}_value100` | lineup turnovers avoided with the player on offense, split into own, teammates, and unassigned team turnovers | pts/100, + good |
| `lineup_{lane}_tov_total_value100`, `lineup_tov_value100` | lane total = own + teammate + team; both lanes summed | pts/100, + good |
| `lineup_{lane}_tov_raw_value100` | unadjusted on-court turnover rate against league average, same price | pts/100, + good |
| `lineup_tov_point_price`, `lineup_team_tov_coverage` | points per kept possession; `absent` ≤2013, `partial` 2014, `full` 2015+ | pts per TOV; label |

Every current scoring-board row has a mix value, and 11,583 of 11,605 have a
lineup TOV view. If a future row has no solve, leave it blank; never zero-fill
it.

## Display contract

1. **Label it research.** Put Playtype mix RAPM under a visible
   "Research" header, visually separate from the ledger. It is descriptive
   only and ungated at this spec; the TD700 predecessor failed the future-oTS
   increment test (`−0.0040 [−0.0233,+0.0136]`).
2. **Never add it.** It is not added to zTS, SV/100, scoring impact, or
   scoring security, and nothing in the ledger closes through it. Do not
   describe any sum with it as a corrected SV.
3. **Never the default rank.** The default sort stays scoring impact. Sorting
   by the mix column on demand is fine.
4. **Signs.** rZTOV is negative-good. Every `*_value100` column and
   `playtype_mix_rapm` are positive-good (positive = easier diet, more
   expected points). Use U+2212 minus signs in display.
5. **Early-era caveats.** 1997 is priced on itself and solved on one season.
   Before 2009, step-back and pull-up threes are almost never tagged in
   play-by-play, so early values mostly separate rim, other twos, threes, and
   transition. Say so in the row detail.
   **Complement effect.** Most of the spread comes from the teammate half.
   Centers who finish easy rim chances can read hard (Valančiūnas −0.75,
   Ayton −0.67), and playmakers read easy (Draymond Green +1.20, Jokić +1.04).
   A row detail that interprets a value should show the mix board's
   own/teammate split, but the table carries only the one column.
6. **Qualification.** Default floors are OffPoss ≥ 1,000 for the scoring
   board. The rZTOV explorer uses ≥ 1,000 for seasons and ≥ 2,000 for its
   2022-26 board.
7. **Never name the tracking vendor** behind the rZTOV involvement volumes, in
   any label, tooltip, payload key, or network response.
8. **Missing stays missing.** Observed-era rows without Synergy role coverage
   (`databallr_synergy_observed_missing`) have no substitute value.
9. **Lineup TOV is a comparison, not a correction.** Show it under the same
   visible "Research" header, next to (not inside) the box turnover values. Do
   not add it to anything or make it the default rank.
   - **Own column.** Say that lineup own is against a flat one-fifth share, not
     the box workload allowance. That is why creators read negative on own
     and positive on teammates. Compare lane totals with box values.
   - **Raw column.** Label it as unadjusted for teammates and opponents.
   - **Before 2014.** Flag `lineup_team_tov_coverage = absent`: team
     turnovers are missing from the source there.
10. **Databallr projection gotcha.** A route that honours a `fields=` projection
   drops any key missing from the client field list, so the column renders blank
   even though the response is 200. Add every new key to the client field list,
   and verify with the client's own query string.

## Reconciliation receipt

Before any integration or publish, run:

```bash
cd nba_pipeline/scripts
../../.venv/bin/python audit_scoring_rztov_reconciliation.py --fetch-live
```

It exits nonzero on any miss and writes
`results/scoring_rztov_integration/reconciliation_receipt.json`. The
`--fetch-live` refresh is a read-only `player_zts` select that reads
credentials from `nba/pipelines/core/.env`. Current result: **all 34 checks
pass.**

| Comparison | Rows | Max gap | Why that size |
| --- | ---: | ---: | --- |
| SV/100 vs Databallr `pred_ots_m4_tsa25fit` | 8,410 | 7.4e-4 | Databallr stores 3 dp |
| zTS / rTS / role vs Databallr | 8,410 | 9.9e-4 / 5e-4 / 0 | 3 dp |
| board rZTOV, lane excess vs rZTOV CSV | 5,370 | 0 | same ledger |
| board lane prices vs rZTOV CSV | 5,370 | 5e-5 | 4 dp CSV |
| board turnover value vs rZTOV points saved | 5,370 | 1.1e-4 | 4 dp prices × excess |
| rZTOV page JS arithmetic vs CSV (rate / saved / points) | 5,370 | 7.9e-4 / 2.4e-4 / 1.9e-4 | 4 dp payload inputs |
| scoring page payload vs board | 11,583 | 5e-5 | 4 dp payload |
| scoring board identities (4) | 11,605 | 1e-6 | 6 dp CSV |
| mix board identities (parent split, share, value split) | 14,565 | 6.7e-16 | exact |
| mix board spec: 550-day half-life, two-season window, every season 1997-2026, `history_seasons` 1 in 1997 and 2 after | 30 seasons | 0 misses | manifest + board |
| board `playtype_mix_rapm` vs mix board `mix_value100` | 11,605 | 5e-7 | 6 dp CSV |
| lineup board identities (lane splits, raw splits, lane sum, value sum) | 14,583 | 1e-6 CSV / 7e-15 manifest | exact in builder |
| lineup board spec: 550-day half-life, two-season window, every season 1997-2026 | 30 seasons | 0 misses | manifest + board |
| board `lineup_*` vs lineup board saved/100 × price | 11,583 | 5e-7 | 6 dp CSV |
| board lineup split closure; coverage labels; pass-through seasons | 11,583 / 30 / 30 | 3e-6 / 0 / 0 | 6 dp CSV |

Every observed row matched (no unmatched observed rows). The rZTOV page's
5,370 embedded seasons equal the CSV's row set. Every scoring-board mix value
has a mix-board row, and every scoring-board lineup value has a lineup-board
row.

### Stability

`audit_playtype_mix_stability.py` solves disjoint two-season windows three
years apart at the same spec. With 187-236 players per pair above 1,500
weighted FGA of exposure in both windows, `mix_rapm` correlates
`0.47`-`0.62` across pairs from 2003-04 → 2006-07 to 2022-23 → 2025-26. It
holds for movers too, meaning players whose frequent-teammate sets barely
overlap (Jaccard < 0.15): `0.45`-`0.62`. The diet travels with the player,
not the roster.

## Rebuild order

```bash
cd nba_pipeline/scripts
../../.venv/bin/python build_rztov_explorer.py --out-dir <scratch> --html ../docs/rztov-explorer.html
../../.venv/bin/python build_td_playtype_mix_board.py            # add --rebuild-attempts after raw PBP changes (~7 min)
../../.venv/bin/python build_lineup_tov_board.py                 # ~25 s
../../.venv/bin/python build_historical_scoring_security.py --reuse-role-cache   # add --refresh-anchor-from-databallr to pull a fresh role snapshot
../../.venv/bin/python build_historical_scoring_impact_html.py
../../.venv/bin/python audit_scoring_rztov_reconciliation.py --fetch-live
../../.venv/bin/python audit_playtype_mix_stability.py            # optional stability receipt
cd ../.. && .venv/bin/python nba_pipeline/scripts/publish_supabase_results.py \
  --dataset historical_scoring_security_1997_2026 \
  --dataset nba_rztov_explorer_seasons_2014_2026 \
  --dataset td_playtype_mix_board_1997_2026 \
  --dataset lineup_tov_board_1997_2026               # dry run; --publish is the owner's step
```

The rZTOV build must run before the scoring board, which reads its CSV. The mix
and lineup boards must also run before the scoring board, which joins them.
They read the previous scoring board only for display names.

## Worked example: DeMar DeRozan, 2026

| Step | Value |
| --- | ---: |
| rTS + role = zTS | 1.446 + 2.399 = 3.845 |
| zTS add−4 = 2 × 23.72 × (3.845 + 4) / 100 | 3.721 |
| SV/100 = −1.211295 + 0.800618 × 3.721 | +1.77 |
| scoring TOV: −(−1.174) × 0.902 | +1.06 |
| scoring impact (no bad pass) | +2.83 |
| bad-pass TOV: −(−0.751) × 1.020 | +0.77 |
| scoring security | +3.59 |
| mix RAPM = 0.239 × own diet 0.936 + 0.761 × teammate diet 0.354 | 0.224 + 0.269 = +0.493 |
| **Playtype mix RAPM** = −0.493 × 91.25 / 50 | **−0.90** (own −0.41, teammate −0.49) |
| **Lineup TOV** scoring lane: own + teammates + team | +0.37 − 0.30 − 0.04 = **+0.04** (raw +0.62) |
| **Lineup TOV** bad-pass lane: own + teammates + team | +0.26 + 0.52 + 0.04 = **+0.81** (raw +0.74) |
| **Lineup TOV** total (box turnover value +1.82) | **+0.85** (raw +1.36), at 1.325 pts per kept possession |

rZTOV is −5.12, second-best of 364 qualified 2026 player-seasons. Playtype
mix RAPM is −0.90, the hardest lineup diet among 385 qualified 2026
player-seasons. With DeRozan on offense, his lineup's shots are priced about
0.9 expected points per 100 possessions below an average diet, a little more
than half of it from teammates' shots. That is the size of the skeptic's
objection. The page shows it beside the ledger instead of folding it in.

On turnovers, the lineup data backs the direction of the box ledger but about half its size after adjustment:

- **Totals.** With DeRozan on offense his lineups avoid `+0.85` points per 100
  of turnovers after ridge adjustment, against `+1.82` on the box ledger and
  `+1.36` raw on-court.
- **Bad-pass lane.** The bad-pass edge carries over and extends to
  teammates.
- **Scoring lane.** His own scoring-turnover edge is largely offset by
  teammates.
