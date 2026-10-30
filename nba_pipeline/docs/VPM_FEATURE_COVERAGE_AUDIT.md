# Current SQ-parent VPM feature audit

**Repair follow-up:** the original frozen package audited below remains
unchanged. The cumulative rim counts have now been recovered with appearance
and trade gates; see [repair receipt](RIM_DEFENSE_MATCHUP_SKILLS.md).
`vpm_neutral_rim_recovered_v3` is the new versioned feature package, not a silent
replacement of the v18 baseline. Other chronological-fit and entrant gaps remain.


Audited 2026-09-20 against v18 config, exporter, own-parent mapper, SQ-parent
solver and saved feature/source artifacts. No player ratings changed.

## Actual input inventory

The extended own-defense arm feeding SQ Parent has 19 numeric prior features,
plus a missing indicator for each. This is distinct from the larger 28-stat
research registry and from the 25 fields exported in the neutral package.

| Side | Features | Units |
| --- | --- | --- |
| Offense | Rim makes, rim attempts | Each per 100 offensive possessions |
| Offense | Midrange makes, midrange attempts | Same |
| Offense | Three-point makes, three-point attempts | Same |
| Offense | FT makes, FT attempts | Same |
| Offense | Assists, player turnovers, offensive rebounds | Each per 100 offensive possessions |
| Offense | Usage events: FGA + .44 FTA + TOV | Per 100 offensive possessions, not conventional share of team usage |
| Defense | Blocks, steals, defensive rebounds | Each per 100 defensive possessions |
| Defense | Six-foot rim points saved, six-foot contest attempts | Each per 100 defensive possessions |
| Defense | Defended FGA | Per 100 defensive possessions |
| Defense | Personal fouls committed | Per 100 combined offensive + defensive possessions |

Makes are attempt-rate times estimated percentage, not separately estimated
make-rate filters. Percentages are underlying inputs even though they do not
enter the O/D mapping as separate columns. Rim savings is league-relative
six-foot contest savings, not PBP rim taxonomy or opponent-normalized savings.

NOT in this current prior: deflections, potential assists, drives, possession
time, contested shots, contested defensive rebounds, fouls drawn, charges
drawn, college/G-League priors or direct height/wingspan. Age and context enter
upstream skill estimation; they are not independent O/D mapping columns.
Complete EPM is a benchmark, not an own-model feature. The three-feature
blocks/steals/DREB defensive arm is a retained core control.

The separate SQ layer changes the parent target and adds zero-prior mid/three
residuals. It does not add all of the research-registry skills to the prior.

## Source coverage and chronology findings

1. **Current skill history begins 2018-10-16 (ending-year season 2019).**
   `player_game_lines.parquet` has 237,292 rows through 2026-06-13. Earlier raw
   PBP availability is not the same as earlier built/filter-ready skill history.
   The own mapper requires at least three matured label cuts and 100 rows per
   side, and the export begins at 2019-10-01. This cannot directly support a
   2020 preseason own-prior fit. Backfill earlier core history and build
   chronologically trained models; do not remove the guard to force a score.
2. **Deflections are not currently a VPM feature.** The built next-game panel
   has 205,519 rows and starts 2018-10-16. The earlier source audit documents
   raw deflections from ending-year 2017, but 2017–2018 were not integrated
   because this exposure panel begins in 2019. Distinguish calendar date,
   ending-year season, raw availability and validated joined availability.
3. **Rim RS history is rejected for seasons 2020–2022.** The saved prediction
   file has 126,474 rows: RS rows in 2019 and 2023–2026, plus valid playoff
   rows in every season 2019–2026. Do not call these three seasons wholly absent.
   See the precise archive diagnosis below.
4. **DFGA is measured-only**, with 208,005 prediction rows from 2018-10-16.
   The exporter checks measured validity and never consumes the tracking
   backfill's estimated or inferred-zero counts.
5. **Observation timing passes, parameter timing does not.** Saved provenance
   has zero observations on/after the emission cutoff, but 90,600 direct
   feature-provenance rows have parameter fit end on/after the cut. This is
   already labeled development replay, not a newly discovered hidden leak.
   PBP parameters are selected through 2025; rim through 2023; DFGA through
   2022. Historical normalizers/initialization also require cutoff-safe refits:
   `prepare()` computes pooled pre-2026 means/variance from its supplied panel,
   while exporter fixes context normalization through 2025. Splitting final
   snapshots does not fix those upstream dependencies.

## Where the rim problem comes from

Source chain:

`player_sheets/game_report/year_files/{year}[ps]_games.csv`
→ `build_rim_defense_matchup_skills.py`
→ `rim_defense_matchups/player_games.parquet`
→ `build_predictive_rim_saved.py`
→ `predictive_rim_saved/dated_predictions.parquet`
→ `export_vpm_own_skills.py` → own O/D priors.

The measurement builder accepts complete `lt6ft_totals_` or `less_6ft_def_`
field families, but explicitly replaces unprefixed `FGA_LT_06`/`FGM_LT_06`
fields with missing values. 2020–2022 RS files only have that unprefixed family.
The September14 investigation documented implausible aggregate contest rates
when those values were interpreted as daily counts.

Direct raw inspection in this audit: Giannis (203507) has 2020 rows on
2019-10-24/26/28 with attempt counts 2/9/14 and made counts 1/5/7; his maximum
attempt count is 254 across 63 rows. 2021 starts 3/18/24 (maximum223), and
2022 starts 6/15/22 (maximum344). These examples are **consistent with cumulative
season-to-date snapshots**, not single-game counts. They do not establish that
all players/rows are recoverable. The older wording “repeated season totals”
should not be interpreted as the same final total repeated on every date.

Next source-repair audit: verify endpoint query semantics and date windows;
check monotonicity, first snapshots, missing appearances, season/team resets,
trades, duplicates and revisions; difference only validated consecutive
snapshots and reconcile to authoritative totals. Gaps may represent multiple
games and must not be assigned to a single game. Prefer a verified per-game
re-fetch where available. Do not silently enable the rejected raw fields or
claim the underlying NBA data is unavailable. No reconstruction attempted yet.

## Current missing-data behavior and its limits

- Export holds the latest covered state before each date. It does not advance
  states to the cut, and sets values missing only after 730 days.
- Mapper uses training-weighted mean imputation plus per-feature missing
  indicators. This avoids treating an unobserved measurement as measured zero.
- A feature needs at least25 observed training rows; this does not establish
  multiple-season reliability or proper era treatment.
- A player with no features on a side is excluded from mapping and gets zero
  prior through the solve fallback. No entrant translation is currently wired.
- A two-year-old carried rim value is neither missing nor accompanied by
  freshness/uncertainty in the mapper: adapter returns only numeric features.
- Feature absence due to an era-wide outage, no attempts, no source coverage,
  stale state and no NBA history are not separate mapping categories.
- v18 imputes makes and attempts independently. Its coherent-shooting option
  defaults false. Missing products can therefore lose makes=attempts×percentage
  coherence after imputation. Some neutral Gaussian rate estimates are negative:
  six negative three-attempt states at the 2020-12-01 and 2021-10-01 cuts; one
  negative rim-attempt state at 2026-09-14. These are exported-state counts, not
  verified active-roster counts. Preserve v18 and test repairs as named changes.

Core-covered universe below means players with a nonmissing assist-rate state,
not the target season roster. Do not advertise these as roster/minutes coverage.

| Preseason target | Cut | Core-covered players | With rim state | Rim state older than365 days |
| --- | --- | ---: | ---: | ---: |
| 2020 | 2019-10-01 | 530 | 517 | 0 |
| 2021 | 2020-12-01 | 653 | 538 | 343 |
| 2022 | 2021-10-01 | 634 | 305 | 86 |
| 2023 | 2022-10-01 | 697 | 311 | 109 |
| 2024 | 2023-10-01 | 712 | 546 | 16 |
| 2025 | 2024-10-01 | 661 | 647 | 85 |
| 2026 | 2025-10-01 | 694 | 682 | 121 |

In particular, surviving 2021–2023 rim coverage is partly conditioned on playoff
participation. A single missing indicator does not demonstrate that this
selection/measurement change is harmless.

## Required handling for the historical rebuild

Preserve the simple O/D SQ-parent architecture and existing controls. First
repair/audit the rim source and backfill core skill history before 2019. Build
an availability table recording first usable observation, last observation,
coverage, provenance, parameter-fit date and label-maturity date per feature/cut.

Use a historical core prior where new features have insufficient earlier
training evidence; admit a richer feature set only when earlier chronological
validation supports it. Keep all model versions and common-feature controls
explicit. Do not fit a late feature's coefficient using later years and insert
it into an earlier rating. A missingness flag alone is not an era strategy.

Carry measured-zero, never-measured, stale, era-outage, no-shot-opportunity and
entrant-no-history statuses separately. Propagate reliability rather than
presenting carried values as fresh. Build dated entrant/G-League O/D priors
and a learned no-source fallback. Repair shooting imputation coherently without
silently changing the saved v18 baseline. Audit derived-feature clocks as the
maximum clock of their inputs.

This audit covers the feature path and documented controls. It is not a full
certification of SQ target timing, upstream source vintages, lineup targets or
2020–2026 forecast validity. Those remain required before claiming no leakage.

## Reproduction and receipts

Run `.venv/bin/python nba_pipeline/scripts/audit_vpm_feature_coverage.py`.
Outputs: `results/estimated_skills/vpm_feature_audit_20260920/feature_coverage.csv`
and `summary.json`, including input hashes. Generated outputs remain artifacts.

Sources: [exporter](../scripts/export_vpm_own_skills.py),
[own mapper](../../../vpm/vpm/harness/own_parent.py),
[adapter](../../../vpm/vpm/harness/own_skills.py),
[rim measurement builder](../scripts/build_rim_defense_matchup_skills.py),
[rim source contract](RIM_DEFENSE_MATCHUP_SKILLS.md),
[deflections source contract](NBA_ESTIMATED_DEFLECTIONS.md),
[xPM/owner benchmark scope](XPM_RESEARCH_NOTES.md).

## Expanded research challenger, September 20

The v23 SQ board now adds deflections and contested DREB on defense and potential assists, drives and possession time on offense (15 O / 9 D). Original v18 remains frozen. All five measured-state panels begin October 16, 2018; absent measurements remain missing. Fixed 2020–2022 tracking tuning is not historical as-of fitting. On matched actual-point 2025 rows, gain over the repaired original-feature control is +10.29e-6 MSE, paired 95% CI [-14.53,+33.99]e-6: inconclusive. Full receipt: [expanded parent](../../../vpm/vpm/docs/OWN_TRACKING_PARENT.md).
