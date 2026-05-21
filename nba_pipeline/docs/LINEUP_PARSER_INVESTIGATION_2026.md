# 2026 Lineup Parser Investigation

This document records the full investigation into the 2026 raw lineup / on-off parser in `nba_pipeline/scripts/lineup_stats.py`.

It is intended to preserve context on:

- what the parser is trying to do
- where its outputs diverged from `pbpstats`
- what bugs were found and fixed
- what was measured along the way
- what still appears unresolved as of March 25, 2026

## Purpose

The goal of the parser is to materialize 2026 lineup and player on/off data directly from raw play-by-play instead of relying on `RAPM26.parquet` or any other derived possession surface.

The intended use case is:

- query lineups by NBA ID
- query player on/off by NBA ID
- store raw-count lineup rows first
- derive ORTG / DRTG / TS% / TOV / FGORB% / rim-mid-3 frequencies and accuracies after aggregation

The intended semantics are close to `pbpstats`, not the repo's older RAPM possession semantics.

## Scope

This investigation only covers:

- NBA regular season `2025-26`
- raw data in `nba_pipeline/raw_data/NBA26.parquet`
- lineup/on-off build logic in `nba_pipeline/scripts/lineup_stats.py`
- shared raw preprocessing in `nba_pipeline/scripts/process_rapm_blocks/common.py`

This document does not cover:

- the standard RAPM parquet build
- DRL / Shapley work
- the alternate clean 3-factor alias path

## Source Of Truth During Investigation

The external comparison target was `pbpstats`.

The key reference endpoint is:

```text
GET https://api.pbpstats.com/get-wowy-stats/nba
```

Important parameter patterns used during the investigation:

- Team totals:
  - `Season=2025-26`
  - `SeasonType=Regular Season`
  - `TeamId=<team id>`
  - `Type=Team`
- Player on:
  - `0Exactly1OnFloor=<nba player id>`
- Player off:
  - `0Exactly1OffFloor=<nba player id>`

Supabase table `public.wowy_stats_two` was also checked. For the current published 2026 season snapshot, `leverage = 0` matched live `pbpstats` totals exactly.

## Original Parser Goals

The requested raw-count outputs were:

- `OffPoss`
- `DefPoss`
- `Points`
- `OpponentPoints`
- `FGA`
- `FGM`
- `FGMiss`
- `FTA`
- `FTM`
- `ORB`
- `TOV`
- `RimAtt`
- `RimMade`
- `MidAtt`
- `MidMade`
- `ThreeAtt`
- `ThreeMade`

Then derive:

- `ORTG = 100 * Points / OffPoss`
- `DRTG = 100 * OpponentPoints / DefPoss`
- `TS% = Points / (2 * TSA)` where `TSA = FGA + 0.44 * FTA`
- `FGORB% = ORB / FGMiss`
- rim / mid / three frequency and accuracy

## Current Parser Flow

### 1. Raw preprocessing

`_base_processing()` in [common.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/process_rapm_blocks/common.py) does the shared raw read and cleanup.

Important steps:

- read `NBA26.parquet`
- normalize `game_id` to stable 10-character strings
- defensively dedupe exact raw rows
- sort by `game_id`, `period`, `event_num`
- repair missing score fields with in-game forward fill
- repair missing lineup slots with in-game lineup fill
- map event type numbers to strings
- compute `Home_Action_Score` and `Away_Action_Score`
- create lag/lead description columns
- alias lineup columns to `a1-a5` and `h1-h5`

### 2. Raw possession preparation

`prepare_raw_possessions()` in [lineup_stats.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/lineup_stats.py) does:

- infer home and away team abbreviations for each game
- compute `offensive_FT_Rebound`
- compute `End_of_Possession`
- compute `TeamOnOffense`
- repair blank FT-side offense assignment
- mark `PotentialFoul`
- run lineup propagation through FT / substitution sequences
- run `ft_off_check_py()` to insert extra `End_of_Possession` corrections
- build `poss_group`
- fill `poss_offense` within each `poss_group`
- drop rows whose `poss_offense` is still unresolved

### 3. Possession summary materialization

`build_possession_summary()` in [lineup_stats.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/lineup_stats.py) does:

- classify shot attempts and zones
- count FGA / FGM / FTM / ORB / TOV by possession owner
- split contiguous non-terminal offense segments into carry rows
- preserve foreign-side technical/admin FT rows as zero-possession scoring rows
- attach terminal lineup rows to possession groups
- create counted possession rows with `OffPoss = 1` and `DefPoss = 1`

### 4. Lineup aggregation

`build_lineup_stats()` in [lineup_stats.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/lineup_stats.py) canonicalizes the five-man units, aggregates offense and defense separately, merges them, and derives rates.

## Investigation Timeline

## Phase 1: Early mismatch surfaced

The first strong signal was that local Jokic and Denver on/off outputs were materially off from trusted public numbers.

Examples discussed during the investigation:

- Basketball Reference showed Denver with Jokic on-court offense around the high-120s and off-court offense around the mid-110s.
- `pbpstats` Denver season team total was confirmed at:
  - `8207` points
  - `6753` offensive possessions

At that stage, local parser totals were materially higher in points and possessions, which was too large to be explained by possession-definition philosophy alone.

## Phase 2: Duplicate raw rows

One of the earliest large bugs found was raw event duplication.

Concrete example:

- Denver vs Spurs game `0022500076`
- raw rows were effectively duplicated
- this produced obviously impossible game totals such as Denver scoring `276`

This led to defensive duplicate-row normalization:

- normalize `game_id`
- dedupe exact raw PBP rows on read
- later also dedupe at raw fetch/write time

Durable rule captured in instructions:

- exact duplicate raw PBP rows must be removed before downstream parsing

## Phase 3: Missing lineup-slot rows

We found that many raw rows were being dropped because one of the 10 lineup slots was blank or null.

This originally cost:

- `3117` raw rows

The fix was:

- in-game `ffill().bfill()` across the lineup slots before dropping unresolved rows

After that change:

- those missing-lineup drops were fully recoverable in the current sample

This ruled out lineup-slot filtering as the main remaining denominator issue.

## Phase 4: Free throw side assignment

We found scoring FT rows where `TeamOnOffense` and later `poss_offense` stayed blank.

The fix was:

- if a row is an FT and the description clearly shows the FT is on the home or away side, assign offense directly from the shooting side as a fallback

This was a real bug, but it was not the final explanation for the season-total mismatch.

## Phase 5: And-1 made-FG ownership

We found that made baskets followed by `S.FOUL` could fail to get offensive ownership on the made-shot row.

The bad pattern was:

- made FG row
- next row is `S.FOUL`
- `TeamOnOffense` stayed blank on the make
- later group fill could attach the basket to the wrong side

The fix was:

- made FG rows always keep the shooting side as `TeamOnOffense`, even when the next event is a shooting foul

This fixed a real row-level attribution issue, but it did not solve the full possession mismatch.

## Phase 6: Foreign-side technical/admin FT contamination

We found possession groups where:

- a live-ball possession belonged to one side
- a technical/admin FT by the other side happened inside the same broad sequence
- the whole group was being rolled up onto the terminal `poss_offense`

The fix was:

- preserve foreign-side technical/admin FT rows as separate zero-possession scoring rows
- do not let them contaminate the live-ball possession owner's scoring

This improved point attribution materially.

Important consequence:

- points survive
- but those admin rows do not add `OffPoss` or `DefPoss`

That design choice is now one known denominator suppressor.

## Phase 7: Non-terminal offense segments

We found possession groups where:

- one offense appeared first
- later the group transitioned to another offense
- the terminal owner inherited the earlier scoring

The fix was:

- split each contiguous non-terminal offense segment into its own possession row

This reduced the remaining point mismatch further.

## Phase 8: Missing end-of-possession triggers

Two additional possession-end rules were identified from an external engine comparison and then added to the local parser family.

### Offensive foul / charge without turnover row

The comparison engine treats an offensive foul or charge row as a possession end when:

- the row is a `Foul`
- the description contains `offensive` or `charge`
- the next action is not a `Turnover`

A league-wide `2026` raw scan found:

- `28` such rows

These were previously missing from the local standard possession definition because the parser only ended those possessions if a companion turnover row appeared.

### Missed last free throw without live rebound continuation

The comparison engine also treats a missed last FT as a possession end when:

- the row is a final FT (`1 of 1`, `2 of 2`, `3 of 3`)
- the FT is missed
- the FT is not technical
- the next row is not a live defensive rebound by the other team
- the next row is not a live offensive rebound by the shooting team

A league-wide `2026` raw scan found:

- `5442` missed final FT rows total
- `4477` followed by live defensive rebounds
- `569` followed by same-team live offensive rebounds
- `396` remaining rows that would become new possession ends under this rule

This made missed last FTs a much larger undercount candidate than the offensive-foul-only bucket.

### Implementation outcome

Both rules were added to:

- `nba_pipeline/scripts/lineup_stats.py`
- `nba_pipeline/scripts/process_rapm_blocks/common.py`
- `nba_pipeline/scripts/process_rapm_blocks/process_rapm.py`

Targeted validation:

- offensive-charge examples in games `0022500040` and `0022500043` now end possessions on the foul row
- missed-final-FT example `0022500009` `Q4 00:08` (`MISS Robinson Free Throw 2 of 2`) now ends the possession because the next event is a violation, not a live rebound

## What Was Ruled Out

### Raw row loss is not the main issue anymore

On the current March 25, 2026 raw file:

- raw rows read: `516,585`
- rows after player filtering in `_base_processing()`: `516,585`

So the parser is no longer losing a meaningful number of raw rows before possession logic starts.

### Missing terminal `End_of_Possession` rows are not the main issue

Audit on the current raw file after `prepare_raw_possessions()`:

- total possession groups: `212,872`
- total groups without terminal `EOP`: `4`
- Denver groups without terminal `EOP`: `1`

That means the `counts -> eop_rows` inner merge is not the main reason possessions are low.

## Current Live Comparison Snapshot

All numbers below are from live `pbpstats` checks and the current local raw-file parser state as of March 25, 2026.

## Denver

Live `pbpstats` team total:

- Points: `8698`
- OpponentPoints: `8381`
- OffPoss: `7158`
- DefPoss: `7209`

Current local raw parser:

- Points: `8697`
- OpponentPoints: `8381`
- OffPoss: `7114`
- DefPoss: `7109`

Delta:

- Points: `-1`
- OpponentPoints: `0`
- OffPoss: `-44`
- DefPoss: `-100`

Interpretation:

- point parity is now excellent
- denominator parity is still not there
- the defensive denominator miss is much larger than the offensive one

## Jokic On, Denver

Live `pbpstats`:

- Points: `5219`
- OpponentPoints: `4755`
- OffPoss: `4117`
- DefPoss: `4106`

Current local raw parser:

- Points: `5220`
- OpponentPoints: `4763`
- OffPoss: `4094`
- DefPoss: `4059`

Delta:

- Points: `+1`
- OpponentPoints: `+8`
- OffPoss: `-23`
- DefPoss: `-47`

Interpretation:

- same general pattern as team-level Denver
- undercount on both sides
- bigger miss on defense

## Luka On, Lakers

Live `pbpstats` for Lakers `2025-26`, Luka (`1629029`) on:

- Points: `5369`
- OpponentPoints: `5203`
- OffPoss: `4454`
- DefPoss: `4471`

Earlier local parsed table snapshot:

- Points: `5153`
- OpponentPoints: `5027`
- OffPoss: `4263`
- DefPoss: `4264`

Later live raw-file audit in the current parser path:

- Points: `5220`
- OpponentPoints: `4763`
- OffPoss: `4094`
- DefPoss: `4059`

The exact raw-file snapshot moved as the season data advanced, but the structural pattern stayed the same:

- local offense possessions are low
- local defense possessions are also low
- defense undercount is at least as large as offense and often larger

This shows the denominator issue is structural, not just a Denver-only quirk.

## Denominator Suppressors Found

## 1. Explicit zero-possession admin rows

The parser currently emits foreign-side technical/admin FT rows as separate zero-possession scoring rows.

On the current Denver sample:

- Denver offense:
  - `14` zero-possession rows
  - `13` points
  - `14` FTA
- Denver defense:
  - `5` zero-possession rows
  - `5` points
  - `5` FTA
- Jokic on offense:
  - `12` zero-possession rows
  - `11` points
- Jokic on defense:
  - `3` zero-possession rows
  - `3` points

These rows help point parity but directly suppress possessions.

They explain part of the gap, but not most of it.

### If those points were removed instead of preserved

That would worsen parity:

- Denver points would move from `8697` to `8684`
- Jokic on points would move from `5220` to `5209`

So the right fix is not to delete those points.

The real question is whether some of those admin sequences should count as their own possessions instead of remaining zero-possession scoring rows.

## 2. Merged dead-ball continuation groups

This appears to be the dominant unresolved issue.

Denver grouped-possession audit on the current raw file found:

- `75` groups containing technical FT events
- `19` groups containing flagrant events
- `12` groups containing transition-FT / take-foul style events
- `351` groups containing `and-1` patterns
- `14` groups ending on turnover while still carrying points
- `12` groups where both teams scored
- `1279` groups with multiple scoring rows

This does not mean all of these are wrong.

It does mean these are exactly the kinds of groups where a single parser `poss_group` can contain multiple possession-like segments under a finer possession model.

Representative high-noise Denver groups included:

- long `and-1` groups ending on `FreeThrow`
- mixed-scoring groups where both teams scored inside one `poss_group`
- technical-FT groups ending on `FreeThrow`, `Rebound`, or even `MAKE`
- turnover-terminal groups that still had earlier scoring in the same group

The highest-signal conclusion from this audit is:

- we are not mostly dropping possessions after grouping
- we are merging some sequences that `pbpstats` likely counts as more than one possession

## Why defensive possessions can be more undercounted

One repeated question during the investigation was whether it makes sense for a player's defensive possessions to be consistently lower.

The answer is:

- yes, but only because the same underlying missing possession is also disappearing from the opponent offensive side
- it is not a one-sided truth; it is a structural parser issue

This happens when:

- a weird dead-ball sequence is merged into a prior or later offensive group
- the counted possession is attached to the terminal owner
- the tracked team's defensive sample loses a denominator event that `pbpstats` is still counting

So a low defensive denominator on Denver or on Luka does not mean the parser is only hurting defense.

It means the parser is mishandling a shared possession boundary, and the tracked player's defensive sample is where the miss becomes visible.

## What We Thought Might Be Happening, But Mostly Ruled Out

### End-of-period possessions being dropped while points stayed

This does not appear to be the main issue.

Current evidence:

- only `1` Denver group lacked terminal `EOP`
- the main suppressor bucket is explicit zero-possession admin rows and merged dead-ball continuation groups

Normal end-of-period handling may still differ from `pbpstats` in edge cases, but it does not explain the bulk of the current denominator gap.

### Massive unresolved `poss_offense` loss

This also does not appear to be the main issue now.

The bigger issue is not rows with no offense owner.

The bigger issue is broad possession grouping around foul/admin/dead-ball sequences.

## Current Working Theory

As of March 25, 2026, the best working theory is:

- the parser is now close on points
- the parser is still low on possessions because some dead-ball foul/admin continuation sequences are being counted as too few possessions
- explicit zero-possession admin FT rows explain a small but real slice of that
- the larger remainder is in merged continuation groups involving:
  - technical FTs
  - flagrants
  - transition/take-foul FTs
  - `and-1` continuations
  - turnover-terminal groups with earlier scoring

This means the unresolved problem is a segmentation problem, not mostly a scoring problem.

## Current Questions For A PBP Parsing Expert

If asking an expert such as Kostya Medvedovsky for help, the most useful questions are:

- Which dead-ball FT/admin sequences in `pbpstats` are promoted into their own counted possessions?
- When a technical FT happens during another live-ball possession sequence, is it always split into its own possession?
- For flagrants with retained possession, how many counted possessions are created and under what exceptions?
- For take fouls / transition FTs with retained possession, what exact possession splitting rule is used?
- Do any technical/admin FT sequences count as zero-possession scoring rows in `pbpstats`, or are they always possession-bearing?
- When a possession group contains scoring by both teams, what exact event boundary forces a split in `pbpstats`?
- How are turnover-terminal groups with earlier scoring handled?

## Most Likely Next Fix Direction

The likely next implementation direction is not to delete points.

The likely next implementation direction is:

- add more synthetic possession splits before `poss_group` is finalized
- especially around:
  - technical/admin FT starts
  - flagrant FT starts
  - transition/take-foul FT starts
  - offense-side changes inside dead-ball continuation sequences
  - turnover-terminal groups that already contain scoring

The objective would be:

- preserve the current point parity
- increase possession counts by splitting more dead-ball continuation groups into separately counted possessions

## Files Touched During This Investigation

- [lineup_stats.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/lineup_stats.py)
- [common.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/process_rapm_blocks/common.py)
- [PIPELINE_QUICKSTART.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/PIPELINE_QUICKSTART.md)
- [README.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/README.md)
- [AGENTS.md](/Users/russellthomas/Docs/pbp_rapm/AGENTS.md)
- [CLAUDE.md](/Users/russellthomas/Docs/pbp_rapm/CLAUDE.md)

## Bottom Line

The investigation started with a broad suspicion that the parser was simply “wrong.”

The current state is much narrower and clearer:

- duplicate raw rows were real and were fixed
- missing lineup-slot row loss was real and was fixed
- blank FT-side ownership was real and was fixed
- `and-1` made-basket ownership was real and was fixed
- foreign-side technical/admin FT contamination was real and was fixed
- contiguous non-terminal offense segments were real and were fixed

What remains is mostly denominator structure:

- not a large numerator problem
- not a large raw-row loss problem
- not a large open-group problem

The remaining parser mismatch is primarily that some dead-ball continuation sequences are still being counted as too few possessions relative to `pbpstats`.
