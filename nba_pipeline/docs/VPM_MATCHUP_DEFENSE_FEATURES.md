# Per-game matchups and defensive VPM feature research

Owner authorized October18,2026: recover all available per-game NBA matchup
evidence for ending seasons2018–2026, engineer defensive features, and test
whether they improve predictive VPM. This extends the existing NBA VPM
research portfolio slot. The full backfill, dated feature export and v47 seven-test evaluation are
complete. The separate fresh-source literal-BAM test and its independent audits are
also complete.
Everything remains research; no feature is promoted.

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
The official census resolves home/away from exact matchup abbreviation tokens,
not from the row's team ID. Neutral-site official rows can both contain the
same `AWAY @ HOME` text. Source checks caught this on2024–25 games before any
model run;19 census records across2025–2026 needed corrected venue slots.
Cached raw responses remain unchanged. Exact token/ID ambiguity fails, and
unknown international preseason opponents keep unknown IDs. This source
repair does not change the frozen field benchmark's venue rules.

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
immutable siblingv48 was prepared to test that actual hypothesis. It
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

### Fresh-source v49 correction

The completed v48 gate passes2018–2024 but rejects2025: archived/native FGA
normalized L1 is25.93% and turnover L1 is55.14%, versus the fixed1% threshold.
Adding playoffs does not explain it; participation counts mostly match while
action attribution differs. The original raw annual response is absent, so its
exact origin is unproven. V48 remains unfitted. This is a source failure, not a
negative predictive BAM result; its gate was not widened.

Separate immutable v49 discards every legacy annual count and metadata input.
Fresh native RS counts, actual own-PBP RS TS rounded to the provider's three
places,24 newly retrieved official Synergy PerGame RS responses2018–2025,
and completed source-year BRef positions reconstruct the same formula. The
fresh source has5,914 playtype rows. The original unweighted sum of
`PPP * POSS` across player/team/playtype rows remains the primary creation
load, with its full-cohort median and fifth-percentile fallback. This original
convention overstates traded-player season load; a GP-weighted diagnostic is
kept separate and never enters the feature or selection family. No BAM model
outcome has been inspected to choose this repair or its settings.

The fresh source, feature and independent formula/date gates all passed
before fitting v49. Its numerical model/DE/selection/seven-test settings are
identical to v48. New source tools are `fetch_fresh_bam_inputs.py`,
`build_fresh_bam_features.py` and the independent fresh-source/BAM auditors;
the output seam is `results/estimated_skills/fresh_bam_v1`. The sibling
configuration is `research_harness_v49_fresh_bam_defense.json`.

## V47 completed results

The archive resolves12,214 requests:11,521 contain2,169,558 native pairs,
and693 are unavailable (22 regular-season games plus671 preseason games).
The full native identity/fidelity audit passes with no count defects. The full
feature export has8,825 dated summaries; independent reconstruction checks
238,275 cells within7.11e-15. Help blocks and potential assists are wholly
unmeasured and are not observed zeros.

Development2020–2022 selects assignment-only. Its game-MSE gain over null is
0.021735, with unadjusted paired95% interval[-0.229193,+0.272663]. The
registered one-SE family preference places null first, individual groups next,
all features next and DE last; equal-preference group ties use assignment-first
order rather than physical feature count. Null misses eligibility by only
0.006895MSE. Removing2019 warm-up training selects the exact null. The emitted
`null_first` sensitivity instead keeps the registered assignment parameters
and removes2019 training; do not confuse that board with the sensitivity's
selection decision.

Frozen2023–2026 stress contains4,920 games and120 team-seasons. The seven
headline quantities below are squared correlations except team-net RMSE.
Higher squared correlation is better; lower RMSE is better.

| Test | v44 | +assignment | EPM | lagged DARKO | lagged LEBRON | lagged MAMBA |
|---|---:|---:|---:|---:|---:|---:|
| All games R² | 0.1866 | 0.1877 | 0.1889 | 0.1752 | 0.1577 | 0.1903 |
| Low-continuity games R² | 0.2957 | 0.2951 | 0.2981 | 0.3044 | 0.2988 | 0.3113 |
| Team win% R² | 0.6691 | 0.6697 | 0.6890 | 0.6490 | 0.5992 | 0.6967 |
| Low-continuity team win% R² | 0.6201 | 0.6177 | 0.4860 | 0.5936 | 0.4687 | 0.5225 |
| Team net RMSE | 3.2352 | 3.2225 | 3.1200 | 3.4102 | 3.6404 | 3.2431 |
| Veteran games R² | 0.1763 | 0.1775 | 0.1795 | 0.1653 | 0.1474 | 0.1852 |
| Veteran team win% R² | 0.6457 | 0.6469 | 0.6665 | 0.6215 | 0.5698 | 0.6834 |

All-game RMSE falls13.9604→13.9510, versus EPM13.9415. Assignment's paired
game-MSE gain over v44 is+0.262909[-0.003996,+0.530044], unresolved. Veteran
game gain is+0.318686[+0.061167,+0.573544]; five of seven headline point
estimates improve. EPM still has better team-net error, with a paired loss
for assignment of-0.650083[-0.863152,-0.454109]. These overlap and are
unadjusted historical research contrasts, not fresh confirmation.

DE ran1,984 evaluations and reached its fixed30-iteration budget without
convergence. Its game RMSE13.9588 and main game gain remain unresolved. No
positive standalone feature gain excludes zero. Concentration, primary share,
assignment difficulty and usage load have the leading positive point estimates;
conditional blocks helps only inside the full fixed model, which itself loses
to null. Highly correlated features prevent unique importance or causal claims.
BPM has no observable development exposure in2020 and its slopes first learn
for2022, leaving only one informative forward development season.

The independent score audit reproduces1,078 score rows,126 paired points,
8,289 game forecasts and210 team-seasons within3.56e-13, including exact veteran
overrides and independent QR coefficients. Supplemental checks reproduce all
seven no-warm-up boards and680,469 actual-point rows with eight paired intervals.
The general field validator verifies source/artifact hashes, clock/score and win
accounting. A scalar/vector missingness comparison bug in the independent
auditor was fixed with three regression tests; the saved VPM inputs, model,
forecasts and scores did not change. The failed receipt and explicit audit
recovery are retained under `research_run/score_audit_recovery/`.

Full controls, per-feature diagnostics and curated evidence are in the sibling
`vpm/docs/MATCHUP_DEFENSE_RESEARCH.md` and
`vpm/evidence/matchup_defense_v47_summary.json`.98 focused RAPM tests pass.

## V49 completed fresh-BAM result

The independent source and formula audits pass646,089 native annual pair
rows,4,389 defender-seasons and6,336 dated feature/provenance rows, maximum
difference1.82e-12. The24 fresh Synergy responses contain5,914 rows. No legacy
annual counts or metadata enter this feature.

Development selects the exact null, also after removing2019 warm-up training.
Fixedridge1 BAM loses0.170959 game-MSE relative to baseline, with paired95%
interval[-0.449508,+0.107591] expressed as gain. DE converges after120
evaluations/four iterations to a numerically negligible increment; selected
boards therefore preserve v44 exactly on all seven tests.

For transparency, the already-registered fixed BAM arm is reported below
**despite not being selected**. These are spent2023–2026 stress outcomes;
none changes selection or warrants promotion.

| Test | v44 / selected null | Forced fixed BAM | EPM |
|---|---:|---:|---:|
| All games R² | 0.1866 | 0.1876 | 0.1889 |
| Low-continuity games R² | 0.2957 | 0.2976 | 0.2981 |
| Team win% R² | 0.6691 | 0.6700 | 0.6890 |
| Low-continuity team win% R² | 0.6201 | 0.6283 | 0.4860 |
| Team net RMSE | 3.2352 | 3.2257 | 3.1200 |
| Veteran games R² | 0.1763 | 0.1773 | 0.1795 |
| Veteran team win% R² | 0.6457 | 0.6466 | 0.6665 |

Forced BAM game RMSE13.9544 is below baseline13.9604 and above EPM13.9415.
Its paired game-MSE gain is+0.166913[-0.028475,+0.362350]. All seven headline
point estimates improve, but all seven paired **error** gains versus baseline
cross zero; team-win and veteran-team-win RMSE actually worsen slightly even
though their squared correlations increase. Actual-possession gain is
+14.660e-6[-3.483,+32.270]e-6, unresolved. The DE possession gain around1e-11
is numerically negligible, not meaningful evidence of a basketball improvement.

All independent score/forecast/QR, no-warm-up and actual-point bootstrap audits
pass for v49, with the same1,078 score rows/126 paired points/8,289 games and
210 team-seasons as v47. Supplemental report-only fixed-BAM uncertainty has
14 comparisons; independent paired-point arithmetic agrees within9.89e-17.
135 focused RAPM source/feature/auditor tests pass. The sibling frozen model
suite previously passes52 focused tests. Configs and incumbent outputs remain
unchanged; neither feature family is promoted and2026–27 is untouched.

Full fresh-BAM controls and curated evidence live in sibling
`vpm/docs/FRESH_BAM_DEFENSE_RESEARCH.md` and
`vpm/evidence/fresh_bam_defense_v49_summary.json`.

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

The archived-input literal-BAM chain can wait for this ordinary job, then run its
full2018–2025 source concordance, required-complete export, independent
formula/date audit, v48 fit and both score audits:

```bash
.venv/bin/python -u nba_pipeline/scripts/run_lagged_bam_research.py --wait-main
```

Its stage logs, immutable code hashes and preserved receipt history live at
`cache/matchup_games_v1/lagged_bam_run/`. It never starts a model from a
partial feature manifest. Six admission and wait-failure tests pass.

Immutable sibling v47 reads the final feature manifest dynamically, rejects
partial-download pilot exports, saves per-feature single/leave-one-out
development diagnostics, and emits selected, DE, simple and no-warm-up arms.
A report-only common-native future actual-possession diagnostic supplements
the seven full-game tests. The independent score auditor reconstructs game
forecasts from player minutes, team quality and logistic wins, all seven
scorecards/paired points, and coefficients via augmented QR least squares.
A separate `audit_matchup_vpm_supplement.py --root <experiment>` verifies
the no-warm-up QR coefficients/boards and independently reconstructs actual
point errors plus the frozen paired game bootstrap; eight focused tests pass.
Initial arithmetic verification reproduces693 frozen-control scorecard rows
within3.6e−13. This validates the auditor, not the unrun new model.

The final27-field pilot(v4) independently verifies238,275 feature cells and
12,561 common-mask point identities, maximum gap3.55e−15. The combined source,
BPM, feature and independent-audit suite passes48 tests. The full v47 download and model stages have now completed as reported above;
pilot validation alone was not their admission receipt.

The source pilot and rejected legacy dated checkpoint are documented in
[matchup source inventory](RIM_DEFENSE_MATCHUP_SKILLS.md). Existing controls:
[long-SPM training](VPM_LONG_SPM_TRAINING.md), [field tests](VPM_FIELD_BENCHMARK.md),
[old/new blends](VPM_LONG_SPM_BLENDS.md).

Fresh-source reconstruction and final audit commands (run in order):

```bash
.venv/bin/python nba_pipeline/scripts/fetch_fresh_bam_inputs.py
.venv/bin/python nba_pipeline/scripts/audit_fresh_bam_sources.py
.venv/bin/python nba_pipeline/scripts/build_fresh_bam_features.py --require-admitted
.venv/bin/python nba_pipeline/scripts/audit_fresh_bam_features.py
.venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v49_fresh_bam_defense.json \
  matchup-defense --stage all
.venv/bin/python nba_pipeline/scripts/audit_matchup_vpm_scorecard.py \
  --root ../vpm/vpm/results/harness/vpm_fresh_bam_defense_v1
.venv/bin/python nba_pipeline/scripts/audit_matchup_vpm_supplement.py \
  --root ../vpm/vpm/results/harness/vpm_fresh_bam_defense_v1
.venv/bin/python ../vpm/vpm/scripts/validate_field_benchmark.py \
  ../vpm/vpm/results/harness/vpm_fresh_bam_defense_v1
```
