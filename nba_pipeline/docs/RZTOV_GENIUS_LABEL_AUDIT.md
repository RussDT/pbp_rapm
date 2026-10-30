# Genius source and putback-label audit

2026-09-05. Source inspection only; no labels, fits, or public ratings were
changed in this audit. Current rZTOV role labels and their fitted prices require
correction/revalidation before team-style interpretation.

## Original source versus transformed labels

The `geniussports/shot_fixed_*_rs.csv` exports count foul-inclusive shooting
opportunities. `shot_*_rs.csv` are FGA-only. `passes_more_rs.csv` exposes Passes
and % Potential Assist; their product recovers integer potential-assist counts
(the per-possession-rate reconstruction differs by at most 0.026 due to export
rounding). All source rows used here are Season `2025-26` from RS filenames.
Files have SSI player IDs but no team or game columns. Team allocation therefore
uses local player-season team possession shares and is explicitly approximate
for traded players, not an exact team-specific tracking extract.

Exact normalized-name matching to local NBA-ID/full-name maps covers 99.28% of
Boston offensive player-slot exposure; Dalano Banton, John Tonje, and Max Shulga
are unmatched in that map and excluded, not guessed. SSI IDs join the tracking
files to each other. Matched rows, allocated over all Boston team possessions:

| Opportunities /100 | Original Genius | Current estimate on same players |
| --- | ---: | ---: |
| Self-created foul-inclusive shots | 55.81 | 47.89 |
| Assisted foul-inclusive shots | 45.90 | 55.45 |
| Potential assists | 45.96 | 47.75 |

This is a source diagnostic, not exact team tracking: trade allocation and
provider possession coverage differ slightly. For White the source FGA count
is 1,108, matching raw PBP; inferred passing exposure is about 5,171 versus
5,157 local resolved possessions. Do not multiply provider rates by local
exposure and claim exact counts. Source tables are saved under
`results/rztov_genius_actuals/`.

## Concrete bug

`build_rztov_explorer.py::putback_counts` labels a shot as a putback whenever the
immediately preceding logged event is a rebound by the shooter in the same
period/game. It never checks offensive versus defensive rebound. Its assertion
that this must be an offensive rebound is false: intervening dribbles and
unrecorded passes need not create logged PBP events.

The explorer then subtracts these counts from Genius self-created volume and
adds them to the assisted bucket. These modified roles become the observed
labels used by the estimated-role model. Thus the downstream assisted label
already includes non-passer opportunities and erroneous defensive-rebound shots.

2026 classifications verified against changes in cumulative `Off:N Def:N`
rebound descriptions:

| Player | Detector putbacks | Following offensive rebound | Following defensive rebound |
| --- | ---: | ---: | ---: |
| Jaylen Brown | 160 | 30 | 130 |
| Derrick White | 78 | 25 | 53 |
| Payton Pritchard | 77 | 29 | 48 |
| Neemias Queta | 155 | 121 | 34 |

Across the season, 25,740 detector matches include 12,983 offensive-rebound
matches, 12,661 defensive-rebound matches, and 96 without resolved counter changes.
These counters establish the defensive false positives; they do not establish
that every offensive-rebound match meets a narrow time-limited putback definition.

Example: original Brown Genius rates are 29.038 self-created and 6.444 assisted.
The current observed training label is 25.776 self-created and 9.706 assisted.
His original potential-assist rate stays 13.397. The recategorization creates
assisted volume without a paired pass.

## Consequence

At fixed shooting volume and matched passer/receiver opportunities, the user's
intuition is correct: an assisted shot plus potential assist has more combined
allowance than a self-created shot under current fitted prices. The team's
apparent reversal cannot be justified by pretending the transformed assisted
bucket is pure potential-assisted volume. Current potast price is approximately
13.43%, not 9%; this does not change the direction of that comparison.

Next correction must separate actual offensive-rebound putbacks from defensive
rebound attacks, retain putbacks as explicitly unassisted/non-passer opportunities
(or give them a separately named lane), rebuild observed/estimated roles, refit
prices, and rerun the team checks. Do not silently call assisted-plus-putback
volume pure assisted shots. All existing learned prices are conditional on the
old label contract and must not be treated as validated constants after repair.
