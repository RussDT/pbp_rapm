# Lineup Turnover Board (two-season TD550, 1997-2026)

**Status (2026-09-29):** descriptive research, ungated. It is a transparency
view shown beside the box rZTOV turnover values on the scoring-impact board.
It is not rZTOV and does not refit the lane prices. It is never added to
scoring value, turnover value, or scoring security, and it is not a prior or
VPM input.

- Builder: `scripts/build_lineup_tov_board.py`
- Output: `results/lineup_tov_board/` (gitignored)
- Tests: `tests/test_lineup_tov_board.py`

## Why it exists

The box ledger prices a player's **own** turnovers against a workload
allowance. It converts the excess to points with lane prices of about `0.90`
per scoring turnover and `1.02` per bad pass (2022-26). Those prices are
regressions of near-unpenalized lane turnover RAPM on the box excess; before
2026-10-19 they used the production `2000/4000` solve and read a shrunk `0.54`
/ `0.77` (see [rZTOV points conversion](./RZTOV.md#points-conversion)). A
price says what an excess turnover costs. It does not say whose turnover the
lineup avoided.

The question is: with the player on offense, how many team turnovers per 100
does the lineup avoid, and whose are they? This board answers it directly
from lineup data instead of leaving the answer inside a price.

## Method

**Window.** Each feature season is solved over itself plus the prior regular
season, using the reader spec shared with `playtype_mix_rapm`:

- 550-day half-life, as of the day after the feature season's last game;
- about `0.63` weight on the prior season;
- one standard regular-season possession per row;
- 1997 is solved on one season, because there is no NBA96 raw file.

**Targets.** Four 0/1 targets per possession, all season-centred on their
time-decay-weighted mean, with positive = saved:

| Target | Meaning |
| --- | --- |
| `scoring_att` | scoring turnover committed by one of the five offensive players |
| `scoring_un` | scoring turnover with no on-court offensive actor (team turnovers: shot clock, 5 and 8 seconds, plus rare actor-lineup mismatches) |
| `badpass_att` | bad-pass turnover committed by one of the five |
| `badpass_un` | bad-pass turnover with no on-court actor |

**Parent.** A joint offense/defense ridge with alphas `3000/5000` (the
Daniel-reader turnover parent) is solved on every target.

**Split.** Each attributed lane also gets a centred focal own-turnover
contrast `c`:

- There are five focal rows per possession, with target `−I(focal committed)`
  and the focal indicator centred within the lineup.
- The contrast alpha is chosen by pooled, time-decay-weighted, grouped-game
  CV. It uses three folds in each of the 2006, 2016, and 2026 windows, and
  picks the largest alpha within one paired SE of the minimum.
- The selected alpha is `300` for both lanes. The contrast lowers CV loss by
  `1.3e-4` (scoring) and `2.2e-4` (bad pass) against no contrast.

The pieces are:

```
own        = (parent_attributed + 4c) / 5
teammates  = 4 (parent_attributed − c) / 5
team       = parent_unassigned
lane total = own + teammates + team
lineup TOV saved = scoring total + bad-pass total
```

Ridge solutions are linear in the target, so on the identical design every
identity is exact. The manifest records the largest gap, `7e-15`.

**Price.** One averted turnover is one possession kept. Its value is the
window's time-decay-weighted net-continuation first-chance value of a
non-turnover possession: the `FC_TS_NET_Diff` baseline, the same price as
`ALT_TOV_NET_VALUE`. That runs from `1.21` to `1.32` points and rises over the
era (`1.325` in 2026). Every piece is saved per 100 × price, in pts/100, with
positive = good.

**Raw.** `*_raw_*` columns are the unadjusted time-decay on-court rates
against league average. They are not adjusted for teammates or opponents.

## Reading it

- **Own is not the box excess.** Lineup own is measured against a flat
  one-fifth share of the lineup's turnovers. Box excess is measured against a
  workload allowance.
  - High-usage creators therefore read negative on lineup own even when their
    box excess is good: they commit more than one fifth of the lineup's
    turnovers because they handle the ball more.
  - Their teammates then read strongly positive.
  - Compare the lane totals, not own versus box.
- **Ridge shrinkage.** Adjusted values are shrunk toward zero. Raw values are
  not shrunk, but they are confounded by teammates and opponents.
- **Team turnovers before 2014.** The 1997-2013 Gabriel-converted TOV
  possessions omit team-actor turnovers, so totals there miss about 4% of
  turnovers and `team` holds only lineup mismatches. `team_tov_coverage` is
  `absent` through 2013, `partial` for the 2014 window, and `full` from 2015.

## Pass-through: how much of the box value the lineup shows

`build_historical_scoring_security.py` writes
`results/historical_scoring_security/lineup_tov_pass_through_1997_2026.csv`.
For each season it fits, on qualified rows (OffPoss ≥ 1,000), an
OffPoss-weighted slope of each lineup value on the matching box value.

| Mean over seasons | 1997-2013 | 2015-2026 | 2026 |
| --- | ---: | ---: | ---: |
| lineup turnover total on box turnover value: slope | 0.39 | 0.50 | 0.58 |
| … correlation | 0.68 | 0.64 | 0.67 |
| scoring lane total: slope | 0.35 | 0.37 | 0.45 |
| bad-pass lane total: slope | 0.31 | 0.43 | 0.47 |
| scoring raw on-court: slope | 0.74 | 0.76 | 0.92 |
| bad-pass raw on-court: slope | 0.71 | 1.01 | 1.08 |
| bad-pass own: slope | −0.15 | 0.31 | 0.38 |

The slopes are against the corrected box prices of 2026-10-19. Those prices are
about 1.7 times the old shrunk ones, so every slope fell by roughly that factor;
the correlations did not change.

How to read these rows:

- **Totals.** Adjusted lineup totals recover about half of the box turnover
  value. Most of the shortfall is this board's own ridge shrinkage
  (`3000/5000`, two seasons).
- **Raw rates.** Raw on-court rates recover roughly three quarters to all of
  it. They are unshrunk but confounded: good ball-security players share the
  floor with good teammates.
- **Bad-pass own.** The negative pre-2016 bad-pass own slope is the
  allowance-versus-flat-share construct at work: potential-assist creators
  carry most of the bad-pass allowance. It is not a data error.

## Examples (2026, pts/100, + good)

| Player | Box TOV value | Lineup own (sc + bp) | Teammates | Team | Lineup total | Raw (sc + bp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeMar DeRozan | +1.82 | +0.63 | +0.22 | +0.00 | +0.85 | +1.36 |
| Shai Gilgeous-Alexander | +2.60 | −1.00 | +3.09 | +0.28 | +2.37 | +4.87 |
| Nikola Jokić | +0.14 | −2.72 | +2.93 | +0.32 | +0.53 | +2.24 |
| Luka Dončić | +0.41 | −3.20 | +4.06 | +0.22 | +1.08 | +1.28 |

**DeRozan.** About half of his box ball-security edge holds up on the floor after adjustment:

- **Scoring lane.** Own `+0.37` is offset by teammates `−0.30`, for a total of
  `+0.04`.
- **Bad-pass lane.** Own `+0.26` and teammates `+0.52` give a total of
  `+0.81`.

**Creators.** Jokić and Dončić show the flat-share construct:

- Own is heavily negative and teammates are heavily positive.
- The lineup total is positive: their lineups avoid more turnovers than their
  box lanes credit.

The 2026 qualified spread (SD) is `0.59` for the lineup total and `0.70` for
box turnover value.

## Rebuild

```bash
cd nba_pipeline/scripts
../../.venv/bin/python build_lineup_tov_board.py                  # ~25 s; runs role-alpha CV
../../.venv/bin/python build_historical_scoring_security.py --reuse-role-cache
../../.venv/bin/python build_historical_scoring_impact_html.py
../../.venv/bin/python audit_scoring_rztov_reconciliation.py --fetch-live
```

**Frozen alphas.** Use `--role-alpha-scoring 300 --role-alpha-badpass 300` to
skip the CV. The builder reads the observed rZTOV CSV and the previous scoring
board only for display names.

**Outputs:**

- `lineup_tov_board_1997_2026.csv` (14,583 rows)
- `coverage.csv`: per-season possession and turnover accounting
- `role_alpha_cv.csv`
- `league_summary.csv`: price and league rates
- `manifest.json`: spec and identity gaps

The Supabase registry carries the board as `lineup_tov_board_1997_2026`
(schema v1). Its eleven priced columns are joined onto
`historical_scoring_security_1997_2026` in schema v4.
