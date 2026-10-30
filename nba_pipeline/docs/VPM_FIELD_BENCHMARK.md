# Predictive VPM field benchmark

October15 follow-up: the [new SQ temporal repair and paired rerun](VPM_SQ_TEMPORAL_REPAIR.md)
is complete in sibling v39/v40. The original artifacts described here remain
preserved; their identified leakage warning is not retroactively erased.

The [player-cohort audit](VPM_EPM_COHORT_AUDIT.md) uses that repaired v40
control to investigate VPM-versus-EPM errors without changing the model.

**Leakage audit, October 15, 2026:** this receipt is **not certified leakage-free**.
The surviving v2 ShotQuality builders compute zone and jumper league shooting
means over the full 2015–2026 panel before their seasonal training split
(`build_russell_sq_v2.py:86,97` and `_hist.py:86,97`). Those means enter
shrunk shooter features, so later outcomes can influence earlier EV inputs.
Filtering possession rows before the rating cut does not remove this upstream
path. The numerical effect on saved ratings is unquantified; the archived EV
lineage and original missing transition sidecar are not fully certified.
Keep the scores as retrospective diagnostics. Rebuild and rerun in a new
version before making clean out-of-sample claims. Saved ratings, scores,
configuration and manifests are preserved; `leakage_audit.json` records this
finding separately.


Owner request: October 15, 2026. Evaluate the existing predictive VPM on the
seven LOKI-style season/game tests against dated local competitors. This is
the active NBA research receipt; WNBA v6 remains waiting for confirmation.

## Frozen protocol

Immutable sibling configuration:
`vpm/vpm/configs/research_harness_v37_field_benchmark.json`, experiment
`vpm_field_benchmark_v3`. Rules recorded before the first scoring run. v35/v1 was superseded to
repair a left-censored veteran classification (Jeremy Pargo, extension only);
v36/v2 is retained as the pre-MAMBA receipt. v37/v3 adds the subsequently
recovered MAMBA source. No performance-based rule or model tuning occurred.
No impact-model parameters are selected or changed in this experiment.

Main period: 2020–21 through 2025–26; extension: 2019–20. Ratings are frozen at
October 1 before the target season, except December 1, 2020. VPM uses the
saved v34 monthly own-SQ boards, not the older fixed-parameter v18 replay.
The saved no-entrant-prior solve is an ablation. Complete EPM uses the exact
requested-date API file, preserving provider-effective dates. DARKO uses
only the previous season's endpoint observation dated before the cut. Local
LEBRON uses the previous season and exact NBA IDs. MAMBA uses the recovered 2015–2025 endpoint archive, matched to exact
normalized same-season names in the canonical NBA-ID box/LEBRON identity
fields. Normalize accents, case and whitespace only. Reject names ambiguous
in either provider, retain unmatched names as missing, and save the ID audit.
This supports the complete requested 2021–2026 overlap. LOKI player-level
boards remain unavailable; its published aggregate scores are not imported.

The reference is the owner's supplied LOKI excerpt and the
[first-party page](https://court-share.com/luma/loki/about). Its public page
states 7,194 games / 180 team-seasons over 2020–21 through 2025–26. Our
census, cuts, fill rules, continuity definition and scoring formulas are
explicit below; this is a comparable test family, not an exact replication.

### Shared population and forecast inputs

RAPM owns `scripts/export_vpm_benchmark_games.py`, the canonical source adapter.
It reconstructs minutes from sorted `time_quarter` MM:SS deltas, assigning each
interval to the post-event home/away lineups. Duplicate raw rows are removed;
last nonmissing scoreboard values are the targets. Require complete clock
mass (including overtime), five distinct valid players on each side, disjoint
sides, a final zero clock, two distinct scoring-actor team IDs, one processed
game date, a nontied final score and plausible possession pace. Quarantine
failures rather than silently evaluating partial scores. The pace denominator
is half the canonical resolved RAPM row count, so it is an observed resolved
possession proxy, not a newly certified official pace measure.

Every system uses the same accepted games, minutes and pace. A team's
minute-weighted rating is `sum(player_rating * minutes) / game_duration`,
which is the sum of five average on-court ratings. Game margin is the home
minus away rating difference times actual pace / 100, plus a shared fixed
3-point home term (zero for the 2020 bubble). This is a conditional player
quality test using realized workload, not an available-before-tipoff minutes
or injury forecast.

Team-season quality uses all accepted actual minutes. Its unweighted
league-team mean is removed separately for each system/season using ratings
and minutes only. Actual net rating is total scoreboard margin / summed
pace * 100. Team wins use actual full scores. A shared fixed logistic
`k=0.14` converts game margins to probabilities; average probabilities over
the realized schedule yield the direct season win-percentage forecast.
Neither the home term nor the logistic coefficient is fit on scored outcomes.

### Coverage and requested subgroups

Native values are retained when available. Missing veterans use −2. Missing
players under 250 prior NBA minutes use the same fixed draft-slot fill:
pick 1: −1; 2–5: −1.5; 6–14: −2; 15–30: −2.5; 31–60: −3. Unknown pick/status
uses −3 without inferring that the player was undrafted. Draft metadata must
have a draft year no later than the cut. These are transparent benchmark
assumptions, not fitted talent or the unpublished LOKI fill scheme.

Prior minutes come from the saved 2014–2026 raw-PBP player-game source,
strictly before the cut, including playoffs, supplemented with pre-2014
minute accounting from the legacy LEBRON file (no impact values enter).
The historical NBA exposure ledger identifies earlier players; fail rather
than classifying an earlier NBA player as a rookie when the available minute
accounting remains below 250. The supplementary early counts are a sufficient
veteran-classification bound, not a claimed complete career-minute history.
The veterans-only test overrides every system's value to −2 for all players
under 250 minutes, even if the system has a native rating.

Continuity is the fraction of current game/team minutes supplied by players
who appeared for that same franchise in the preceding regular season.
Game continuity averages the two sides. Low continuity is fixed at <50%.
Membership uses prior-year raw clock minutes; full-season team/game
classification uses realized workload and is diagnostic. It does not use
scores or impact ratings. This is our declared continuity measure; the
excerpt does not provide LOKI's complete definition.

### Seven tests and uncertainty

1. All-game margin R².
2. Low-continuity game margin R².
3. Team win-percentage R² from rating × actual minutes.
4. Low-continuity team win-percentage R².
5. Team net-rating RMSE, points per 100.
6. Veterans-only game margin R².
7. Veterans-only team win-percentage R².

Headline explained-variation R² is squared correlation; it describes a fitted
association. Direct predictive R² (`1−SSE/SST`), RMSE and bias are saved
separately. For team wins, explained variation correlates minute-weighted
quality with actual wins; direct errors use the fixed schedule probabilities.
Do not present squared correlation as calibrated forecast accuracy.

Seasons receive equal weight. Save pooled and individual-season tests plus an
all-systems-native game sensitivity. Pair VPM with each competitor/ablation
on identical rows. Game bootstrap resamples games within seasons. Team
bootstrap resamples whole seasons, preserving opponent/team dependence; the
six available season blocks limit precision. Use 4,000 draws, frozen seed
20261015. Positive MSE gain means the reference has larger error than VPM;
an interval crossing zero is inconclusive. No opponent-strength, calibration,
synergy or position correction is trained on these scored seasons.

## Reproduction and artifacts

From RAPM:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v37_field_benchmark.json \
  preseason-benchmark --stage prepare

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v37_field_benchmark.json \
  preseason-benchmark --stage evaluate
```

Outputs: sibling `vpm/vpm/results/harness/vpm_field_benchmark_v3/`.
`benchmark_report.md` is the reader receipt; `scorecard.csv` contains pooled,
per-season and common-native scores. `paired_comparisons.csv` gives paired
MSE/R² differences and intervals. Game/team/player inputs, coverage,
quarantine, requested/effective dates and source/code/artifact hashes are
retained. Source/config/code locks reject changed inputs; use a new experiment
for changes. All generated outputs remain gitignored and private.

## Interpretation boundary

All 2020–2026 evidence is retrospective development. V34's chronological
refits fence observations, but its architecture and sources were assembled
later. EPM was retrieved retrospectively; annual DARKO, LEBRON and MAMBA files do
not certify what their live sites displayed before opening night. Compare
these labeled artifacts honestly; no claim of a certified archived live-DARKO
victory follows. Source coverage and identity limitations of the underlying
monthly VPM history remain applicable.

2026–27 is forbidden by the benchmark runner, including via an arbitrary
season override. A future confirmation needs its own frozen model and
pre-opening-night board hashes, certified rookie/roster identities, dated
competitor snapshots and a predeclared evaluation window. This work does
not change v18/v34 or publish a leaderboard.

## Results

### October 15 owner-requested scale diagnostic

The owner suspected the current monthly board was compressed and suggested
a 1.10 multiplier. A separate read-only diagnostic uses the saved 7,230
games / 180 team-seasons, without changing the frozen benchmark or boards.
Sibling `vpm/scripts/audit_field_benchmark_scale.py` verifies the original
artifact hashes and writes `vpm/results/harness/vpm_scale_audit_20261015/`.
This is post-hoc investigation on spent outcomes, still subject to the SQ
leakage finding above; it selects or promotes no replacement model.

Calibration regresses actual margin or net rating on predicted quality, with
a separate diagnostic intercept per season and equal season weights. A slope
above one suggests expansion of the aggregate rating signal. VPM's game
slope is **1.046** (game-cluster 95% interval **[0.994, 1.096]**;
whole-season-block sensitivity **[0.943, 1.125]**). Its team-net slope is
**1.099 [0.978, 1.195]**, resampling whole seasons. Both remain consistent
with unit scale; a team-level slope does not identify separate offense,
defense, star-tail or individual-player corrections.

For a fixed uniform 1.10 expansion of all player quality including missing
fills, the home term remains exactly three points. Game RMSE changes from
**13.98964 to 13.99122**; team-net RMSE changes from **3.17923 to 3.15642**.
The team-net MSE gain is **+0.1445 [-0.1909, +0.4305]** with whole-season
blocks. Scaling only native VPM ratings while preserving the benchmark's
common missing-player fill gives game RMSE **14.00468** and team-net RMSE
**3.17443**. These two policies must not be conflated: native-only game MSE
worsens by **0.4210** (game-cluster gain interval **[-0.7561, -0.0957]**;
whole-season interval crosses zero). Neither sensitivity refits a forecast
intercept or changes the logistic curve.

The September 1 board's 585 recent players have unweighted net SD **2.076**;
this is a descriptive selected-player spread, not the exposure-weighted
reference or an estimate of true talent variance. Jokić's +7.539 would become
+8.293 under uniform 1.10 scaling; SGA's +6.851 would become +7.536.
Original ratings remain unchanged. Repair SQ first, then reassess scale on
the repaired pipeline before freezing any future confirmation candidate.

The owner's follow-up **1.20×** sensitivity is saved separately under
`vpm/results/harness/vpm_scale_audit_1_2_20261015/`; reproduce with the same
command below, adding `--multiplier 1.2` and changing the output directory.
Uniform expansion gives game RMSE **14.02043** and team-net RMSE **3.18005**.
Game MSE gain is **-0.8623 [-1.6262, -0.0708]** with within-season game
resampling, but the six-season-block interval is **[-2.4335, +0.3729]**.
Team-net gain is **-0.0052 [-0.6785, +0.5608]**. Preserving common missing
fills while scaling native ratings gives game RMSE **14.04000** and team-net
RMSE **3.20279**. The current evidence favors 1.10 over 1.20 descriptively;
the follow-up does not create an untouched scale-selection test. Jokić would
be +9.047, SGA +8.221 and Wembanyama +7.608. No board values changed.

Reproduce from RAPM:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python ../vpm/vpm/scripts/audit_field_benchmark_scale.py \
  --source ../vpm/vpm/results/harness/vpm_field_benchmark_v3 \
  --monthly ../vpm/vpm/results/harness/vpm_monthly_2019_v1/monthly_ratings.parquet \
  --output ../vpm/vpm/results/harness/vpm_scale_audit_20261015
```

### Frozen benchmark scores

Completed frozen v37/v3 run: **7,230 games and 180 team-seasons** over
2020–21 through 2025–26. The optional 2019–20 extension adds 1,059 games.
All 9,519 games in the 2019–2026 source census pass the clock/score/lineup
checks; the scored 2020–2026 set contains 8,289 unique games. There are
853 low-continuity headline games and 20 low-continuity team-seasons.
No historically observed pre-2014 player remains ambiguously below the
veteran threshold after the early-minute supplement.

All rows below use squared correlation except team net RMSE:

| Test | VPM | EPM | Lagged DARKO | Lagged LEBRON | Lagged MAMBA |
|---|---:|---:|---:|---:|---:|
| Game margins R² | 0.179 | 0.179 | 0.166 | 0.152 | 0.181 |
| Low-continuity games R² | 0.285 | 0.285 | 0.281 | 0.272 | 0.294 |
| Team win percentage R² | 0.639 | 0.654 | 0.621 | 0.593 | 0.671 |
| Low-continuity teams R² | 0.377 | 0.409 | 0.479 | 0.453 | 0.528 |
| Team net RMSE (pts/100) | 3.179 | 3.084 | 3.344 | 3.465 | 3.204 |
| Veterans-only games R² | 0.168 | 0.168 | 0.157 | 0.146 | 0.175 |
| Veterans-only team wins R² | 0.614 | 0.632 | 0.596 | 0.570 | 0.655 |

Direct game-margin RMSE is **13.990 VPM, 14.008 EPM, 14.102 lagged DARKO,
14.217 lagged LEBRON, 14.151 lagged MAMBA**. Direct forecast R² is respectively
0.1720, 0.1698, 0.1587, 0.1449 and 0.1529. The shared fixed home term produces
about +1.26 to +1.31 points of bias; it has not been retuned on these outcomes.

Paired game MSE gains (reference error minus VPM error), with game-clustered
95% intervals:

- EPM: **+0.527 [−0.725, +1.744]** — unresolved; not an EPM win.
- Lagged DARKO: **+3.157 [+1.907, +4.383]**.
- Lagged LEBRON: **+6.412 [+4.655, +8.150]**.
- Lagged MAMBA: **+4.537 [+2.221, +6.845]**.
- Same-solve no-entrant ablation: **+1.821 [+1.121, +2.531]**.

MAMBA's slightly higher game squared correlation does not translate into
smaller direct errors at its archived scale. Its team-win association is
stronger. EPM also has the better season-win association and lower team-net
error. VPM-minus-EPM team-net MSE gain is **−0.597 [−1.331, +0.160]**:
the error point estimate favors EPM, with an inconclusive six-season interval.
Veterans-only game VPM versus EPM is unresolved (**+0.233 [−0.917, +1.425]**
MSE gain). On low-continuity teams, VPM's direct win-percentage forecast is
worse than EPM (**−0.00181 [−0.00326, −0.00013]** MSE gain), but this is only
20 teams over six season blocks. Do not turn this into player-level labels
or fit a correction on the scored years.

Mean native minute coverage across the six headline seasons: **90.46% VPM,
90.11% EPM, 90.11% DARKO, 88.07% LEBRON, 86.20% MAMBA**. All remaining minutes
receive the frozen common fill. The MAMBA identity audit accepts 3,167
player-season names across source seasons 2019–2025, rejects no ambiguous
matches, and leaves ten 2025 names unmatched rather than guessing IDs.
The all-systems-native full-game sensitivity has just 250 games and is
strongly selected by bench/rookie participation; it is a coverage diagnostic,
not a replacement headline sample.

Validation: **60 focused tests pass**, covering the existing harness/monthly
contracts plus new snapshot, source-lock, shared replacement, pre-2014 minute,
identity ambiguity, continuity chronology, clock/overtime, sparse score,
quarantine and R²-definition checks. Both repositories pass `git diff --check`.
An independent saved-artifact audit also verifies every hash, game/player
uniqueness, five-player clock mass, season win/point accounting, the veteran
override and weighted-OLS versus direct-error definitions. It passes; see
`independent_validation.json`. Reproduce with:

```bash
.venv/bin/python ../vpm/vpm/scripts/validate_field_benchmark.py \
  ../vpm/vpm/results/harness/vpm_field_benchmark_v3
```

The final manifest hashes ten generated CSV/parquet artifacts and all rating,
raw/processed, code and configuration sources. Original v18/v34 and all 2027
outcomes remain unchanged. Nothing was published.


## Leakage audit details (October 15 follow-up)

The direct feature construction problem is visible in both surviving SQ
builders: `lg_by_zone = df.groupby('zone').made.mean()` and
`lgj = df.made.mean()` occur before `tri = z[z.season_end_year < test_season]`.
These are outcome-derived priors, not harmless category inventories.
They enter `sh_zone_rate` and `sh_jumper_rate`. In a simple formula-only
prefix exhibit, holding the earlier shooter's 1/2 count fixed while appending
four made shots to a 2/4 league panel changes the earlier smoothed feature
from 0.5000 to 0.7451. This proves failure of prefix invariance in the formula;
it does not quantify the effect in the archived EVs or the final scoreboard.

What passed the narrower audit: all 2,627,535 saved own-skill provenance rows
have observations and parameter fit ends before their cuts, and parameter
fit dates no later than their cuts. Saved benchmark boards' last training
games precede their preseason cuts. Own O/D mapper code admits only labels
whose complete forward windows have matured; checked receipts end no later
than their emission cuts. These checks do not inspect or repair upstream SQ
feature construction.

Additional distinctions: architecture/penalties/features were chosen with
historical development evidence, so the scorecard is not a fresh holdout;
actual future minutes/pace are oracle inputs shared by all systems; squared
correlation measures association rather than calibrated forecast accuracy;
retrospectively retrieved provider files do not prove their historical model
publication vintages. None of those points substitutes for fixing the
confirmed full-panel-prior construction.

Required clean rerun: preserve existing v18/v34 and v37 evidence, generate a
new past-only SQ version with date-audited transition inputs, demonstrate that
appending/changing post-cut observations cannot change earlier EVs or ratings,
then rebuild the dated VPM boards and rerun all seven comparisons. Historical
scores still remain development evidence after that repair; untouched
2026–27 confirmation requires a separately frozen protocol.
