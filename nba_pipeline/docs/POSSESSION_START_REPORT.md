# Possession Start Report

`scripts/build_possession_start_report.py` recreates a possession-start offensive-rating chart from local raw NBA play-by-play.

The report uses the standard RAPM possession parser, so the possession denominator matches the repo's processed `RAPM{YY}.parquet` surface. Each possession is scored with actual offensive points, then labeled by the terminal event that created the possession. Categories are not mutually exclusive: for example, an at-rim missed field goal is also a missed 2PT and a missed FG.

## Command

```bash
python nba_pipeline/scripts/build_possession_start_report.py --year 26 --season-type RS
```

Outputs default to `nba_pipeline/results/possession_start_report/`:

- `possession_start_league_26_rs.csv`
- `possession_start_teams_26_rs.csv`
- `possession_start_league_26_rs_league.png`
- `possession_start_league_26_rs_league.svg`

To render a team chart:

```bash
python nba_pipeline/scripts/build_possession_start_report.py --year 26 --season-type RS --team OKC
```

Use `--compare scope` when a team chart should compare categories to that team's own average offensive possession instead of league average.

## Category Notes

- `Live Ball Turnover` is a previous terminal turnover with a steal tag in the event text.
- `Deadball` is a previous terminal turnover without a steal tag.
- Miss categories are based on the field-goal miss that immediately precedes the terminal defensive rebound.
- `At Rim Miss (Unblocked)` uses rim action-type codes, layup/dunk/tip text, or a `<=4` foot distance fallback, excluding shots with a block tag. `At Rim Block` contains the blocked subset separately. Broader missed-FG and missed-2PT aggregates still include blocks.
- `FT Miss` includes both previous terminal missed free throws and defensive rebounds that terminate a possession after a missed free throw. Offensive rebounds after a missed free throw keep the same possession alive, so they should not create a new possession-start row.
- `Corner 3 Miss` and `Above The Break 3 Miss` are distance approximations because this raw PBP source does not contain shot coordinates or explicit corner labels. Missed threes at `<=24` feet are labeled corner-range; missed threes at `>=25` feet are labeled above-break-range; threes without a parsed distance remain only in `Missed 3PT`.
- `Timeout` is applied when a timeout appears before the first live action in the new possession.

## Reader chart

The chart sorts categories by PPP relative to the selected average, labels each bar with that PPP difference, and shows actual PPP and possession counts in adjacent columns. Missed threes appear as one `Missed 3PT` bar; distance-based three subcategories remain in CSVs for audit only. `Made FG` remains an aggregate CSV row but is replaced in the chart by `Made Rim FG`, `Made Mid FG`, and `Made 3FG`.

Made categories describe the previous terminal MAKE event: threes use the 3PT description, rim makes use the same action-code/text/at-most-four-foot rule as rim misses, and mid means all remaining made two-pointers. The three made categories partition `Made FG` exactly. Sequences ending with a subsequent free throw are classified by that terminal free throw, not the preceding made basket.

## Own missed-shot continuation companion

Run `python nba_pipeline/scripts/build_possession_start_continuation.py` after the 2026 RS report has been written with `--write-possession-detail`. It writes `own_miss_continuation_26_rs.csv`, an event-detail parquet, and the combined `possession_start_with_continuation_26_rs` CSV/PNG/SVG under the report output directory.

Own continuation is actual later offensive points in the same standard possession after each missed FGA, averaged over **all** eligible misses in the category, including unrecovered misses with zero continuation. It is not points conditional on securing an offensive rebound. The eligible possession universe matches the possession-start report. Repeated misses in one possession each define a remaining-value node; their downstream points overlap and cannot be summed as separate possession credit.

The opponent column retains its existing defensive-rebound-conditioned denominator. The own-continuation and opponent-PPP columns therefore must not be directly added or subtracted. Made-shot, turnover, timeout, and FT rows show a dash for the FGA-only continuation estimate. Raw actual FT makes after the miss are included in continuation; this companion does not use the DECOMP expected-FT target or its first-miss-only grain.

## Shooter-position filters

`build_possession_start_position_filters.py` reads the raw 2026 RS events plus the saved possession/continuation detail and produces aggregated `possession_start_position_filters.json` and `possession_start_filters.html`. The checked-in fragment template is `scripts/templates/possession_start_filters.html`.

The local generated position snapshot, `shooter_positions_2026.json`, is read-only `player_stats` data from Databallr: exact `nba_id`, `player_name`, `Pos2 AS position`, and `Minutes AS minutes`, with `year=2026 AND playoffs=0`. Choose the highest-minute row if a player has multiple rows. PG/SG map to Guards, SF/PF to Forwards, C to Bigs. Missing labels remain Unknown (available as Unclassified), and All includes them. Current coverage is 217,131 of 218,296 FGA (99.466%), with 82 unclassified shooters. Never infer their positions from names.

The filter belongs to the shooter of the preceding shot, not the rebounder or lineup. For misses leading to a new possession, use the raw shot immediately preceding the terminal rebound, exactly matching the existing report. Make and final-FT categories use the preceding terminal shooter; non-shot turnover/timeout rows appear only in All. Own-continuation values use the missed-FGA shooter's group. Make-rate counts include all eligible FGA nodes, including repeated shots and and-one makes; terminal made-FG rows exclude sequences ending in free throws, so those denominators differ deliberately.

Every filter retains the same 1.1579515769 league PPP reference. The All/Guards/Forwards/Bigs plot scale is shared; the sparse Unclassified view may expand it to avoid clipping. The existing opponent PPP and continuation denominators remain distinct. Current unblocked-rim opponent PPP is 1.208369 Guards, 1.198077 Forwards, and 1.223270 Bigs; this is broad rim classification, not an easy-layup difficulty filter.

Validation checks exact All reproduction, exhaustive group count partitions, unique live-event keys, and preservation of prior-terminal text. Raw duplicate keys here are 28 pairs of non-live heave/block annotations: interval lookup uses first start / last end occurrence but retains the raw rows. Verified selection changes, tooltip values, and responsive light/dark display at 736 and 360 pixels.

## Paired comparisons with separate baselines (2026-09-10)

The `possession_miss_balance.html` companion separates the outcomes into two panels with differences expressed per 100. The main view pools short/long midrange using their respective miss and opponent-possession counts, keeps rim misses unblocked, and shows one three-point miss category. A detail checkbox restores blocks and short/long midrange. Made rim/mid/three appear only in the opponent panel. The make-rate table expresses makes per 100 attempts. Own continuation is actual remaining points per all misses minus the fixed league all-FG-miss average (`0.3775391182`). Opponent scoring is the original full ensuing-possession PPP minus overall league PPP (`1.1579515769`), without rebound-probability weighting or a sign flip. Positive therefore means more own scoring for the own bar and more opponent scoring for the opponent bar. These have different denominators and must not be stacked or summed.

Made rim, midrange, and three-point FGs are included in the opponent comparison. Their own missed-FG continuation is unavailable, displayed as a dash rather than zero. Shooter filters retain these made-FG categories; non-shot categories appear only in All. Baselines remain fixed across filters. The chart retains actual rates/counts in tooltips and the all-attempt make-rate table.

All-player unblocked rim misses show own continuation `+0.1225887` points/miss and opponent PPP `+0.0506750`. Long-mid misses show own `-0.0756803` and opponent `+0.0209440`. Made rim FGs show opponent PPP `-0.0302011`. This is descriptive comparison, not causal shot-selection value.

### Audit-only common-denominator calculation

`miss_branch_balance.py` retains a separate per-original-miss calculation in the data, but the reader chart does not use it. It defines `C` as own remaining points/all misses, `q` as the fraction with an exactly matched immediate defensive-rebound branch, `P` as opponent PPP on those matched branches, and `L` as league PPP. Its opponent premium is `T=q*(P-L)`; centered own is `C-C0`, centered opponent benefit is `T0-T`, and the audit sum is their sum. The fixed reference `T0=0.0116894390` comes from all FG misses.

The matching gate excludes 49 of 80,093 opponent miss-start rows from this audit calculation because their preceding shot is absent from eligible own-miss detail: 41 rim blocks, six short-mid misses, and two threes. Unblocked rim and long-mid have no exclusions. The displayed original `n`, `ppp`, and `delta` remain unchanged; only audit weighting uses `paired_n` and `paired_ppp`. Do not interpret unmatched rows as an OREB rate or the audit sum as complete per-attempt value.

### Make/miss-weighted opponent scoring

The reader now defaults to separate short (5–14 feet, excluding rim actions) and long (15+ feet, non-three) midrange. A new panel computes exactly `FG% * opponent_PPP_after_make + (1-FG%) * opponent_PPP_after_miss`, then compares that blend with league PPP. Each zone uses its own made-shot split and observed FGA make rate; rim includes blocked attempts in this full-mix panel. The own-continuation rim row remains explicitly unblocked, with blocked rim available as detail.

2025–26 All: rim 1.161080, short mid 1.153209, long mid 1.145546, three 1.147873, versus league 1.157952. This is a make/miss-standardized blend, not the observed unconditional mean across every subsequent opponent possession: missed-shot PPP remains DREB-conditioned. It does not follow the ensuing opponent possession after an offensive recovery. 41 midrange attempts lack a qualifying distance and remain outside short/long rows. Full branch rates remain available under the expandable shot-result view.

### First-chance-only filter

Run `build_possession_start_position_filters.py --first-chance-only` to write separate `_first_chance` JSON/HTML artifacts. Within the saved eligible possession intervals, shots qualify only before the first own rebound of a live missed shot/FT; administrative rebounds after nonfinal, technical, or flagrant FTs do not start a second chance. Opponent branch rows are filtered by the exact source-shot event's first-chance flag. Own continuation keeps all remaining actual own points after an eligible first-chance miss. FG rates and made/missed opponent branch means use the same first-chance restriction; original all-shot artifacts remain available.

The filter retains 190,132 of 218,296 attempts and excludes 28,164 later-chance attempts. Own centering uses the league first-chance-miss continuation mean (0.3698859002); opponent centering remains full league possession PPP (1.1579515769). Opponent scoring still covers the full ensuing possession, including opponent second chances. Rim weighted mix includes blocks. Counts partition exactly and weighted means reproduce the make/miss identity.

### Consistent rim definition

Both reader panels now include blocked rim misses. Own rim continuation pools blocked and unblocked remaining-point totals divided by their combined miss count; opponent miss PPP pools their opponent point totals divided by combined ensuing-possession count. The rim breakdown checkbox adds unblocked and blocked rows beneath the inclusive rim row. The first-chance restriction remains active.

### Season comparison: 2024–25 and 2025–26

The position-filter builder accepts `--year 25 --first-chance-only`; the default remains 26. Each season requires its matching possession detail, missed-FGA continuation detail, and exact season-specific Databallr Pos2 snapshot. Do not reuse 2026 positions for 2025. The combined reader embeds both generated payloads, defaults to the newly requested 2024–25 season, and retains shooter group and rim breakdown controls. Every season uses its own full-possession PPP baseline and first-chance-miss continuation baseline. The make/miss formula, source-shot first-chance restriction, midrange distance boundaries, and inclusive rim definition are unchanged.

### Reader net tables

Season/group-linked tables show miss-conditioned `SC - (miss_PPP - league_PPP)` and all-attempt `shot_value*FG% + (1-FG%)*SC - (weighted_PPP-league_PPP)`. These are explicitly estimated adjusted shot values with mixed branch conditioning, not full unconditional accounting; made-FG value excludes associated free throws. Rim SC combines blocked/unblocked misses by miss counts. Negative opponent cost is a benefit.

### Databallr team export

`--team-filters` exports all 30 shooting teams crossed with the five position groups, while preserving the league cohort. Assign source opponent-possession rows from the exact preceding shot event’s `off_team_abbr`; assert team attempt, miss, and opponent-branch count closure. Product `/shot-aftermath` consumes only the curated aggregates, generated by Databallr `scripts/import-shot-aftermath.py`. The Vite page has fixed seasonal league references, first-chance-only samples, inclusive rim misses, both net tables, URL filters and CSV export.

Publication status (2026-09-11 03:03 UTC): Databallr PR410 merged through the required PR workflow after direct main push was rejected. Local page, data invariants, typecheck and production build passed. Immediate live crawler checks still saw the previous deployment; production delivery was not yet verified. Local preview is available at http://localhost:8080/shot-aftermath.

### Current Databallr per-miss correction (2026-09-10)

The local product now uses the previously audit-only matched branch calculation: `T = paired_n / misses * (paired_ppp - league_ppp)`. Own continuation already includes unrecovered misses as zero and receives no further OREB weighting. Relative charts use season-wide all-miss references `SC0` and `T0`; net is `(SC-SC0)-(T-T0)`. Raw net tables use `SC-T`; all-attempt cost is `FG%*(made_ppp-L)+(1-FG%)*T`. The old mix-weighted PPP remains a diagnostic, not the cost. This supersedes the earlier mixed-denominator product formulas above. It still excludes the later opponent possession following an own recovery and does not constitute complete sequence accounting.

Team filtering now uses the exact source shot event’s shooting team, fixing one 2025 sparse-team mismatch that could otherwise yield matched branch counts above misses. Both seasons were rebuilt and the product importer enforces `0 <= paired_n <= misses` without clipping. Five product invariant tests and frontend typecheck pass. Latest browser verification was blocked by a stale error-page URL policy; local preview was restarted and this correction has not been released.
