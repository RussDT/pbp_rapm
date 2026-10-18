# Per-game matchups and defensive VPM feature research

Owner authorized October18,2026: recover all available per-game NBA matchup
evidence for ending seasons2018–2026, engineer defensive features, and test
whether they improve predictive VPM. This extends the existing NBA VPM
research portfolio slot. Implementation and backfill are in progress; no
feature has been admitted and no performance result is claimed here.

## Source contract

`scripts/fetch_matchup_game_archive.py` builds an official `LeagueGameFinder`
census and checks canonical PBP game IDs, dates, phases and ending seasons.
The initial census contains12,214 completed games:11,503 regular-season and
playoff games plus preseason, play-in and Cup-final games. All-Star games
are excluded. Extra phases remain archived separately and do not alter the
existing field benchmark's regular-season game universe. Ending2027 is
forbidden. Census includes known PBP-only games if official history omits one.

Private, generated outputs under `cache/matchup_games_v1/`:

- `game_census.parquet`, `census_sources/official_games_{year}.csv`;
- `raw/{game_id}.json`: exact provider response before normalization;
- `metadata/{game_id}.json`: request/check time, SHA256, coverage/status;
- `pairs/{game_id}.parquet`: native statistics with exact attacker/defender IDs;
- `download_audit.parquet`, `download_counts.csv`, `progress.json`.

Every native JSON's outer team/player is the attacker; the nested opponent
is the defender. Verify exact official teams and flag opposing native-roster
absence; defense-only players can be absent from the offense-oriented outer
roster. Feature admission verifies exact PBP game/player/team and positive side
exposure instead of rejecting those players automatically. Never copy the
legacy BAM ID reversal. Preserve fractional partial possessions,
nulls, source defects and per-field coverage. An empty response is unavailable,
not a zero-skill observation. Source makes exceeding attempts remain flagged
and excluded from accepted count measurements. Tracking totals need not close
to the scoreboard: early FT fields can be unmeasured, shooting assignments can
overlap, and potential assists are zero in every initial live pilot. Each lane
requires its own measurement gate. Do not infer exclusive possession credit.

Backfill is restartable, rate limited, checksum protected and uses bounded
retries. An HTTP403/429 stops the batch for cooldown. Downloading historical
data today does not establish original publication or extraction availability.
Network retries replace failed HTTP connection pools while retaining the
same public endpoint, headers and request-rate limit. A bounded rolling queue
keeps other workers moving while one game is slow. This recovery follows
observed2021 request timeouts that succeeded on fresh ordinary connections;
it does not convert a timeout into an empty or zero-valued observation.

## Candidate features and chronology

RAPM owns raw pairs and measurement engineering. VPM receives only stabilized,
dated player summaries from `results/estimated_skills/matchup_defense_v1/`,
with exposure, coverage, latest observation date and pricing-training dates.
No raw provider pairs or event rows cross the seam.

1. Assignment difficulty: exposure-weighted opponent prior-season BPM/OBPM,
   prior-day scoring, assist and shot/turnover load, concentration
   and role mix. Compare raw burden with position-adjusted burden. Include
   assignment difficulty as an explicit candidate and report its incremental
   value; do not force a favorable coefficient or equate harder assignments
   with causal points saved.
2. Execution: raw points and field-goal points allowed, two/three shot frequency and
   accuracy, assists, turnovers, blocks and fouls, with coherent denominators.
   Compare actual production with expectations based on the opponent's strictly
   earlier completed game-days. Keep assignment strength separate from results.
3. BAM-like residual summaries: opponent-adjusted production allowed rather
   than importing annual full-sample BAM. Distinguish native team-point rates
   from individual production; neither is automatically total defensive impact.
4. Position/context and missingness: exact prior-season position, opponent
   position mix, coverage and count exposure. Missing source fields remain
   missing; an absent historical player classification is not an inferred role.

The initial bank has27 fields including coverage, which is a reliability
diagnostic and never a player-quality input. Use fixed TD550 count decay and
exposure stabilization K1000; conditional shooting percentages use K100 shot
opportunities. Opponent rates use fixed-anchor decayed past counts; all
current-game and same-day results are excluded. Full-season2026 means never
price an earlier historical game. Basketball-Reference BPM2.0 was introduced
in2020 and retrospectively recalculated: even prior-season BPM retrieved today
has a historical-definition vintage caveat. BPM/OBPM matchup measurements
are unavailable before a conservative March1,2020 knowledge floor; position
information remains available. Even later retrieved historical values are
not certified as-was publications. Position BPM means use prior-season
minutes, and traded-player metadata uses the unique whole-season aggregate.
The staged metadata has5,489 exact NBA-ID player-seasons2017–2026; three
unmatched identities stay missing.

### Feature dictionary

All names below have the prefix `matchup_`. Unless marked as a rating or
fraction, values are per100 tracked assignment partial possessions. Expected
production uses the attacker's strictly earlier game-days; each lane keeps its
own exposure and measurement mask. Values are TD550/K1000 stabilized except
coverage. These are candidate predictors, not independent additive credits.

|Feature suffix|Meaning|Direction before model fitting|
|---|---|---|
|`assignment_difficulty`|Expected opponent FG+FT scoring load|Assignment burden|
|`assignment_bpm`|Prior-season opponent BPM, assignment weighted|Assignment burden; rating units|
|`assignment_obpm`|Prior-season opponent offensive BPM|Assignment burden; rating units|
|`assignment_pos_difficulty`|Opponent BPM minus the prior-minute-weighted mean at his position|Position-adjusted burden; rating units|
|`assignment_assist_load100`|Expected opponent assists|Passing burden|
|`assignment_usage_load100`|Expected opponent FGA +0.44FTA +TOV|Shot/turnover workload burden|
|`points_saved100`|Expected minus allowed FG+FT points|Positive means less scoring allowed|
|`fg_points_saved100`|Expected minus allowed FG points|Positive means less FG scoring allowed|
|`ft_points_saved100`|Expected minus allowed FT points|Positive means less FT scoring allowed|
|`fg_points_allowed100`|Raw tracked FG points allowed|Higher means more scoring allowed|
|`points_allowed100`|Raw tracked FG+FT points allowed|Higher means more scoring allowed|
|`shot_frequency_saved100`|Expected minus tracked opponent FGA|Positive means fewer shots allowed|
|`three_frequency_saved100`|Expected minus tracked opponent three attempts|Positive means fewer threes allowed|
|`assists_saved100`|Expected minus tracked opponent assists|Positive means fewer assists allowed|
|`turnovers_above_expected100`|Tracked minus expected opponent turnovers|Positive means more turnovers|
|`bam`|Expected minus actual FG points at the observed two/three attempt mix|Conditional shotmaking residual; distinct from legacy BAM|
|`blocks100`|Tracked matchup blocks|Measured defensive event rate|
|`help_blocks100`|Tracked help blocks, where measurement is established|Measured defensive event rate|
|`shooting_fouls100`|Tracked shooting fouls committed|Higher means more fouls|
|`potential_assists_allowed100`|Tracked opponent potential assists, where measured|Higher means more passing chances allowed|
|`ft_frequency_saved100`|Expected minus tracked opponent FTA|Positive means fewer FT attempts allowed|
|`primary_share`|Assignment-weighted average of each game's largest opponent share|Concentration; fraction|
|`concentration`|Assignment-weighted average within-game opponent-share HHI|Concentration; fraction|
|`coverage`|Accepted assignment exposure divided by all official defensive exposure|Acquisition diagnostic; excluded from ability inputs|
|`guard_share`|Exposure against opponents listed as guards last season|Position mix; fraction|
|`wing_share`|Exposure against opponents listed as forwards last season|Position mix; fraction|
|`big_share`|Exposure against opponents listed as centers last season|Position mix; fraction|

The direction column describes the measurement. All fitted coefficients,
including assignment difficulty, remain free-signed. The raw-allowed and
adjusted-saved columns deliberately share information; ridge and the saved
correlation/leave-one-out diagnostics address that dependence without treating
each coefficient as a separately identified basketball effect.

## Predictive selection and evaluation contract

Keep frozen v44 offense, parent4000/8000, TD550, repaired SQ residuals and
entrant treatment unchanged for the first defense-only increments. Preserve
v39, v44, true-zero-prior and entrant-only controls. Coefficients fit only
earlier completed future-game packets through the exact fixed lineup solve;
no saved shrunken player rating is treated as an error-free target. The2019
warm-up may use saved v39 forecasts because v44 has no2018 emission; report
that switch and a sensitivity excluding the2019 warm-up from every coefficient
training prefix explicitly (hence the2020 forecast has exact zero increment).

Development seasons2020–2022 alone select between null, assignment-only,
execution-only, BAM-like and all-feature candidates. Differential evolution
uses a compact continuous search over log10 ridge[-3,3], overall increment
scale[0,2] and group weights[0,1], seed20261018, popsize8, maxiter30, no polish.
Feature slopes remain free-signed. Objective is equal-season actual game
margin MSE. A paired one-SE rule prefers the simpler eligible model, including
the exact null. Record the full search and the limited three-year development
sample. Neither stress rankings nor individual player aesthetics select models.

Freeze all choices before2023–2026 report-only stress. Run the identical seven
field tests, direct actual-point diagnostics where feasible, and paired error
comparisons against VPM controls, dated EPM and lagged DARKO/LEBRON/MAMBA.
Keep squared correlation distinct from direct forecast R²; RMSE is lower-is-
better. Shared actual minutes/pace make this conditional player-quality
evidence. These historical seasons are already spent research evidence;
2026–27 remains untouched and nothing is promoted automatically.

Audit source hashes/coverage, exact IDs/dates/teams, same-day exclusion,
future-append invariance, defensive signs, frozen-control parity, exact affine
response and independent seven-test arithmetic. A negative or unresolved
feature result is a valid outcome; a backfill alone does not complete this
research goal.

### Literal BAM follow-up within this goal

Source inspection established that `matchup_bam` is a conditional FG residual
and does not test the original BAM formula's turnover penalty, Synergy
creation load, nonlinear fallback and positional multiplier. A separate
immutable siblingv48 is being prepared to test that actual hypothesis. It
uses completed2018–2025 sources only, fresh native matchup counts,
verified same-season TS/creation inputs and completed-season BRef positions.
The archived BAM values remain a separately named retrospective diagnostic:
their positions were static profiles, and2026 is incomplete. No new model
outcomes have been inspected to choose this addition.

The original2018 annual archive matches the native correct attacker/defender
orientation exactly across78,602 pairs for integer shots, points and turnovers.
Before v48 fits, require resolved RS requests and native/source checks for
every needed2018–2025 source year. The partial source audit now verifies2018–2021; a partial
identity proof does not certify the remaining years. Source-version tolerances are checked before any model score.
The dated reconstruction preserves the legacy1.05 turnover/FGA-equivalent
penalty and0.44 FT coefficient, with the all-defender FGA denominator before
unknown-pair masking, exposure-weighted annual normalization, exact RS-end
TD550, at most four completed source years and one K1000 shrink. Its source
defects remain disclosed. Independent partial-pilot reconstruction checks
311,861 native pairs and4,744 dated snapshots within1.82e-12; all77 combined
source/feature/BAM tests pass. This remains partial, and neither the pilot nor
the stored annual BAM is a predictive performance result. v47's
frozen feature and model recipe remains separate.

## Commands

Run the complete restartable chain with the bundled Python environment:

```bash
.venv/bin/python -u nba_pipeline/scripts/run_matchup_defense_research.py
```

This ordinary process runs download, archive audit, full feature export,
feature audit, immutable v47 evaluation, independent score/QR audit and the
existing field audit in order. It stops on any failure or code change. Stage
logs and progress are private under `cache/matchup_games_v1/research_run/`.
It does not create a scheduled automation. Individual stage commands:

```bash
.venv/bin/python nba_pipeline/scripts/fetch_matchup_game_archive.py --stage all
.venv/bin/python nba_pipeline/scripts/fetch_matchup_game_archive.py --stage fetch
.venv/bin/python nba_pipeline/scripts/fetch_matchup_game_archive.py --stage audit
.venv/bin/python nba_pipeline/scripts/fetch_matchup_bpm.py
.venv/bin/python nba_pipeline/scripts/build_matchup_defensive_features.py
.venv/bin/python nba_pipeline/scripts/audit_matchup_game_archive.py
.venv/bin/python nba_pipeline/scripts/audit_matchup_defensive_features.py \
  nba_pipeline/results/estimated_skills/matchup_defense_v1
.venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v47_matchup_defense.json \
  matchup-defense --stage all
.venv/bin/python nba_pipeline/scripts/audit_matchup_vpm_scorecard.py
.venv/bin/python -m pytest -q nba_pipeline/tests/test_matchup_game_archive.py
```

Immutable sibling v47 reads the final feature manifest dynamically, rejects
partial-download pilot exports, saves per-feature single/leave-one-out
development diagnostics, and emits selected, DE, simple and no-warm-up arms.
A report-only common-native future actual-possession diagnostic supplements
the seven full-game tests. The independent score auditor reconstructs game
forecasts from player minutes, team quality and logistic wins, all seven
scorecards/paired points, and coefficients via augmented QR least squares.
Initial arithmetic verification reproduces693 frozen-control scorecard rows
within3.6e−13. This validates the auditor, not the unrun new model.

The final27-field pilot(v4) independently verifies238,275 feature cells and
12,561 common-mask point identities, maximum gap3.55e−15. The combined source,
BPM, feature and independent-audit suite passes48 tests. The full download
and subsequent model stages are running; pilot validation is not a substitute
for their eventual receipts or a predictive performance result.

The source pilot and rejected legacy dated checkpoint are documented in
[matchup source inventory](RIM_DEFENSE_MATCHUP_SKILLS.md). Existing controls:
[long-SPM training](VPM_LONG_SPM_TRAINING.md), [field tests](VPM_FIELD_BENCHMARK.md),
[old/new blends](VPM_LONG_SPM_BLENDS.md).
