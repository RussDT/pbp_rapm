# NBA Analytics Project Portfolio

Last reviewed: 2026-10-21 (owner requested MPG decision and quarterly predictive VPM history from2014, with Net lanes)

This is the portfolio-level source of truth for deciding what to work on. It is
not a methodology document and it does not replace each project's canonical
handoff. Its job is to keep finished work, release work, research, and ideas
from competing for attention as if they were all equally urgent.

## Portfolio call

The constraint is not a shortage of good work. It is too much simultaneous
work in progress.

Many projects that feel unfinished are actually one of three different things:

- built but not integrated into the main public product;
- correctly waiting for untouched 2026-27 evidence;
- useful research that already produced a decision and should now stay parked.

Use a hard work-in-progress limit of three:

1. one **SHIP** project;
2. one **RESEARCH** project;
3. one **MAINTENANCE** project.

Everything else is **WAITING**, **PARKED**, or **DONE**. A new idea enters the
portfolio before it becomes code. Starting it requires pausing one of the three
active projects.

## The three active projects

### 1. SHIP — nbarapm.com release train

**Outcome:** make nbarapm.com the coherent home for the public RAPM family and
give the already-built specialty products a clear route from the main site.

Owner decision 2026-09-03: Net DECOMP is the forward explanatory product. The
release should make these distinctions obvious:

- RAPM is the parent impact estimate;
- Net DECOMP explains RAPM with continuation-priced shot, turnover, and
  second-chance accounting;
- gross DECOMP is a compatibility benchmark during migration;
- VPM / Impact Monster is the ten-factor predictive research breakdown;
- rZTOV, zTS, title lift, Five-Man Fit, and teammate help are separate tools,
  not extra components to stack onto RAPM or VPM.

**Next physical actions:**

1. Migrate the completed 140-window rolling 1Y-5Y grids for actual Net and
   SQ-v2 plus residual Net into `csvs/scposs`, site, Supabase, and downstream
   consumers with a gross rollback bundle; keep TD550 SQ VPM research-labeled.
2. Use the generated six-series/36-mechanism registry as the release inventory;
   require CSV/parquet pairs for all 140 cuts in each Net series.
3. In `products/nbarapm-next`, finish or explicitly shelve the in-progress
   player-v3 / net-DECOMP work, then build and typecheck it.
4. Add one public “Research & tools” index that links to the existing VPM,
   rZTOV/zTS/scoring, title-lift/help, and Five-Man Fit readers.
5. Write one release checklist covering source artifact, validation, ingest,
   public label, and rollback for every published metric.
6. Register every reader-facing result in the Supabase results manifest or
   record its explicit product-specific publication replacement.

**Done means:** the current player surface reads from audited active DECOMP,
every specialty metric has one canonical public URL and status label, and the
release can be repeated from documented commands. Every published result is
also discoverable through the Supabase results manifest or a named dedicated
table/workflow.

Canonical docs: [RAPM Table Catalog](./RAPM_TABLE_CATALOG.md),
[Net DECOMP](./NET_CONTINUATION_DECOMP.md),
[Gross compatibility DECOMP](./PUBLIC_DECOMP.md),
[VPM reproduction](./VPM_REPRODUCTION.md), [rZTOV](./RZTOV.md), and
[Pipeline quickstart](./PIPELINE_QUICKSTART.md), and
[Supabase results publishing](./SUPABASE_RESULTS_PUBLISHING.md).

### 2. RESEARCH — NBA predictive VPM and historical coverage

**Pregame prior implementation, October21:** owner-authorized [implementation](VPM_PREGAME_BUILD_STATUS.md), same RESEARCH slot. Complete455-date/3068-game2020–22 timing/context then decay comparison freezes annual-snapshot + season/home +365-day calibration-game half-life: equal-season RMSE13.435545 versus null13.441051. Conditional paired decay MSE gain+.147994, game95%CI[.011405,.286076], is small post-selection development evidence. Timing difference remains unresolved; all arms refresh daily inputs/ratings. Independent6538pin/1700JSON audit passes and original null predictions are bit-identical. Existing adjoint controller runs2023–26calibrations before frozen rolling/annual/native/currentNet diagnostics and recommendation. Preserve scientific controls/quarantine, finite-budget qualification and failed roots. No selectedv61/2027 or publication change.

**Quarterly history completed, October21:** same VPM slot. Matched mapper100 ablation retains MPG. Audited v65 emits52 quarterly cuts /105268 player-date rows from2014 through2026 with exact prior/final Net lanes. Missing tracking uses model-aware aggregate fallbacks until replacements are admitted and observed; rejected v64 remains archived. Seven stress point estimates improve, overall game gain remains unresolved (13.888098→13.881618). Native forward gain versus matched zero-impact priors is positive;71 focused tests and independent source/model/native/browser/export audits pass. Reader localhost8768/quarterly-history.html. Existing selected model stays unchanged; no2027 outcomes/public promotion. See [quarterly history](VPM_QUARTERLY_HISTORY.md) and [MPG decision](VPM_MPG_DECISION.md).

**Coherent view, October20:** same selected mapper100, unchanged data. ratings.html and net-decomposition.html now alias one O/D/total leaderboard with grouped Net summary and three-column player lanes; priors are separate values.1556/582 universes and CSV/source identities verified. No new model/project; team-win inputs still pending.

**Win-projection/display follow-up, October20:** same VPM slot. Unified season machinery inspected read-only; selected-VPM minutes/team/win bridge implemented and7 tests pass, but current rosters/minutes/identities missing and owner assumption choice pending. No current win forecasts claimed. Net reader now defaults to midrange-reference frequency; fresh canonical five-shot audit passes and selected ratings remain identical. See VPM_TEAM_WIN_PROJECTIONS.md and VPM_NET_LANE_DECOMPOSITION.md.

**Owner selection, October20:** owner chooses already-audited clean mapper100 as local VPM default. Ratings/skills/Net and exports aligned through sibling current_model.json; JokicD+2.595/Reed+3.430. Preserve historical evidence and previous mapper10; no new fit, prospective-confirmation claim or public deployment.

**Defense and exact Net follow-up completed, October20:** [nine defensive calibration arms](VPM_DEFENSIVE_CALIBRATION.md) select mapper100, retain4000/8000/TD550; game improvement unresolved and low-continuity team win percentage worse. [Parent now fully decomposes into learned Net lanes](VPM_NET_LANE_DECOMPOSITION.md), including own/entrant/combined priors and SQ children. Clean entrant source excludes two quarantined2015 games; eight direct clean parents and component sums audited, historical forecast parity1.2e-12. Reader8767/net-decomposition.html exposes582/1556 players. Same RESEARCH slot, no automatic promotion/public deployment/2027 outcomes.

**Daily-skill organization, October20:** [one operational inventory and preflight](ESTIMATED_SKILLS_DAILY.md) routes current-neutral/standalone/retained skill outputs, source requirements and pinned recipe files.16 groups/88 routed outputs,20 research lanes;16 tests and browser inventory pass. Live2027 adapters and scheduling remain unwired; no live-ready claim, new fits,2027 outcomes or extra portfolio project.

**Offensive creation follow-up completed, October20:** [three separate predictive blocks](VPM_OFFENSIVE_CREATION_TEST.md), same Parent research slot. Rim/potential assists game13.894258 vs current13.895289 is unresolved and low-continuity worsens; assistedness13.907115 worsens; playtype points wins early development but later13.897795 and native/prior-only errors worsen. Retain current. Difficulty already uses point-valued2*TSA*difficulty/100, with approximate FGA+.44FTA denominator explicitly disclosed. Source/model/582-player reader audits and6 tests pass;8767/offensive-creation.html. No2027/public/component promotion.

**Current SQ ablation completed, October19:** [mid/three corrections with fixed current priors](VPM_CURRENT_SQ_ABLATION.md), same research slot. GameRMSE no SQ13.9567/mid13.9221/three13.9253/both13.8953; both wins early four-way selection and later paired overall-game interval is positive. Season/low-continuity results remain unresolved. Owner retains D3 residual48000; larger-penalty exploratory fits were already completed and set aside. Independent raw-prefix/board/seven-test audits pass; four-way predictive reader8767/sq-ablation.html. No public/component release or2027.

**Role/unsplit follow-ups completed, October19:** [dated role-aware prior](VPM_ROLE_AWARE_PRIOR.md) and [all current features in both mappers](VPM_UNSPLIT_FEATURE_PRIOR.md), same research slot. Both win their fixed early-development comparisons but do not replace current: later gameRMSE13.8976/13.9140 versus current13.8953; all-input loses all seven game/season comparisons. Native all-input D-only crossed gain is a diagnostic, not a separately trained replacement. Independent source/profile/seven-test audits pass; combined8767/prior-comparisons.html exposes582/1556 ratings and O/D priors. No2027/component promotion/public deployment.

**Expanded forward-prior follow-up completed, October19:** [previous repaired with more features and covariance-aware future calibration](VPM_REPAIRED_FORWARD_EXPANDED.md), same VPM research slot. More features plus dated MPG in the forward mapper yields gameRMSE13.8953 vs previous13.9659, all seven historical paired gains positive; EPM superiority remains unresolved. Prior spread increases without an output multiplier. Overall early-development winner is still the feature-only old-label extension; the forward+MPG model wins within its declared family. Audited585-player prior/final-rating reader8767; no automatic public/component release or2027 outcomes.

**Corrected ratings follow-up completed, October19:** [fixed shooting fouls and contested-DREB replacement](VPM_FIXED_FOULS_CONTESTED_DREB.md). Actual seven-field Parent fits now replace totalDREB entirely. ContestedTD275 gameRMSE13.9600 vs total13.9700/previous13.9659; game gains remain unresolved, while the tiny native actual-point rebound increment excludes zero. Same research slot, no public/component launch choice. Audited585-player current comparison is at local8766; old8765 remains flagged pre-repair.

**Scope correction, October19:** the completed v51 work below concerns Parent VPM, not the ten-component product. The parent core+FT launch recommendation is withdrawn as a component launch choice. Owner needs one clear architecture map and component ratings before release; see [VPM model map](VPM_MODEL_MAP.md). Component launch selection remains unresolved.

**Owner launch follow-up completed, October19:** [defensive skill bank and ratings review](VPM_LAUNCH_DEFENSE.md), sibling immutable v51. Separate rim/disruption/foul/rebound and component maps;48 prior variants/date. Development chooses seven-field rim+foul defense4000/8000/TD275. Stress game13.9705/core13.9604 and teamnet3.2125/core3.2352 are unresolved; low-continuity regresses. Core+datedFT13.9564 has seven favorable unresolved point estimates and is the retrospective launch-review recommendation. Local six-model reader/CSV has585 recent players atSeptember1,2026; opening-night identities/roster remain a separate release task.25 tests and independent source/numerical/seven-test/actual-point audits pass. Owner requested ratings first; no public deployment or2027 outcomes. This resumes/completes the same VPM research slot and supplies the existing SHIP release train.

**Owner follow-up completed, October19:** [extended defense, dated expected FT and context](VPM_EXPECTED_FT_CONTEXT.md), immutable v50. All seven defense fields and dated FT give stress game RMSE13.9764 vs core13.9604; FT alone improves by0.0047 but game gain is unresolved. FT improves neutral future actual-point MSE by23.8e-6[1.8,45.3]e-6 on the common-native diagnostic. Home/rubberband do not add; tuned full package4000/8000/TD275 has gameRMSE14.0340 and loses to core with a paired interval excluding zero. All source/numerical/seven-test audits pass; preserve v44/v45, no promotion or2027 outcomes.

**Owner instruction completed, October18:** [per-game matchup feature research](VPM_MATCHUP_DEFENSE_FEATURES.md). All12,214 requests resolved;11,521 contain2,169,558 pairs. V47 assignment-only improves spent2023–26 game RMSE13.9604→13.9510, but paired gain is unresolved and no-warm-up development selects null. V48 archived2025 source fails; v49 reconstructs fresh BAM and selects exact null. Forced BAM improves seven headline point estimates on spent stress but all paired error gains versus baseline remain unresolved.135 source/feature/auditor tests and full independent seven-test/QR/actual-point audits pass. No promotion or2027 outcomes. Keep this NBA VPM research slot waiting for fresh confirmation; do not start a new research project automatically.

**Owner-requested blend follow-up completed:** immutable v46 [previous/new VPM blends](VPM_LONG_SPM_BLENDS.md).50/50 stress gameRMSE13.9474 versus previous13.9659/new13.9604/EPM13.9415. Development2020–22 chooses pure previous VPM. Preserve all five weights as research; no promotion, stress retuning or2027 outcomes.

**Current owner instruction completed:** [long-sample SPM calibration and forward evidence tuning](VPM_LONG_SPM_TRAINING.md), immutable sibling v44. Six-year pooled calibration, core defense, 4000/8000 and TD550; development2020–22 and report-only stress2023–26. Seven stress point estimates improve over current, but six paired differences remain unresolved; EPM retains better game/team-net errors. Keep as research, preserve current controls and untouched2027. No further stress tuning or promotion. Source, numerical and independent validation are complete; v43 remains rejected. This is the same NBA VPM research slot.

**Earlier completed instruction, October 15:** repair the identified upstream SQ lookahead,
prove prefix invariance, rebuild dated boards and rerun the existing predictive VPM on the
LOKI-style game-margin, season win-percentage, net-rating, low-continuity,
and veterans-only tests against available dated EPM, DARKO and LEBRON.
Use the completed monthly own-SQ history; freeze evaluation definitions and
save paired comparisons, coverage, source hashes and explicit vintage caveats.
No model retuning or 2027 outcomes. WNBA v6 is waiting for its existing gate.

**Completed receipt:** [Predictive VPM field benchmark](VPM_FIELD_BENCHMARK.md):
7,230 games / 180 team-seasons, all seven tests, EPM and lagged
DARKO/LEBRON/MAMBA, paired intervals, coverage and independent artifact audit.
VPM ties EPM on game prediction; season-level point estimates favor EPM.
**Completed repair:** [v39 SQ / v40 paired rerun](VPM_SQ_TEMPORAL_REPAIR.md).
22 dated SQ models, unchanged actual targets and priors, zero-change temporal
prefix audits, 800-file lineage validation and 72 tests. Game RMSE changes
13.9896→13.9928; no statistically resolved before/after or EPM game difference.
**Architecture follow-up:** owner requested a concrete design after the
zero-prior comparison. [Forward-trained Parent proposal](VPM_FORWARD_TRAINED_ARCHITECTURE.md)
learns prior means through the lineup solve against future actual games, then
precision and temporal residuals. The [v43 first-stage implementation](VPM_FORWARD_PRIOR_TRAINING.md)
is complete/audited but loses to the existing prior VPM and is rejected as a
replacement. Heterogeneous precision and explicit temporal residuals remain
unimplemented; diagnose/recover dated training support before advancing.
**Next trigger:** certify/freeze prospective 2026–27 boards and competitor
vintages under a separate protocol. Old v34/v37 remain flagged and preserved;
historical selection and oracle-minute limitations remain.
**Done boundary:** repair and seven-test paired receipt complete; no model
retuning, leaderboard publication or promotion.

### 3. MAINTENANCE — workspace rescue and git backup

**Outcome:** turn the current worktree into recoverable, project-sized units.

At this review the RAPM repo has 23 tracked changes and 113 untracked paths.
The `nbarapm-next`, legacy nbarapm, VPM Lab, and From Impact to Winning repos
also contain active changes. This is now a delivery risk, not cosmetic mess.

**Next physical actions:**

1. Use `audit_repo_inventory.py` in dry-run mode before reclassifying anything.
2. Group changes by project; do not make one giant “everything” commit.
3. Commit code, tests, docs, and curated small references; keep generated
   parquets, CSVs, caches, logs, and models ignored.
4. Record each clean stopping point in the relevant scratchpad/handoff.
5. Do not delete or move ambiguous untracked work until its owning project is
   identified.

**Done means:** each live project has a clean or deliberately documented dirty
state, recent work is backed up, and generated artifacts are not mixed with
source changes.

Canonical docs: [Git backup strategy](./GIT_BACKUP_STRATEGY.md),
[Artifact manifest](./ARTIFACT_MANIFEST.md), and
[Repo cleanup plan](./REPO_CLEANUP_PLAN.md).

## Portfolio register

| Project | State | What already exists | Next action / stop rule |
| --- | --- | --- | --- |
| WNBA predictive parent | WAITING | Completed v6 actual-FT parent, historical own priors, evaluation and reader | Refresh certified sources; pregame rotation test; frozen 2027 confirmation |
| Workspace rescue | MAINTENANCE | The source is present, but several repos have large uncommitted worktrees | Inventory and checkpoint by project; do not delete ambiguous work |
| Daily NBA RAPM | RUNNING | Automated 1Y-6Y, TD700, standalone metrics, downstream sync; smooth clock-aware rubberband model validated and promoted | Preserve the alpha-100 architecture, re-estimate its coefficients per rolling window, and do not treat a long pooled surface as era-specific |
| Net-continuation DECOMP | SHIP | Selected forward product; exact accounting identities and complete 26-window five-year archive | Build current 1Y-5Y release, mapper, consumer cutover, and rollback bundle |
| Gross eight-component DECOMP | RUNNING / COMPATIBILITY | Exact row algebra, rolling builder, 97-column schema | Maintain only for active consumers, matched comparison, and rollback during net migration |
| nbarapm-next player experience | SHIP | Large Next.js product; player-v3 and net-DECOMP work in progress | Choose finish-or-shelve boundary, then build/typecheck |
| VPM / Impact Monster reader | DONE | Live frozen EPM12 x2 board, comparison tabs, model card | Refresh only after a deliberate harness selection |
| VPM priors and game-prediction proof | WAITING | Seven-test 2021–2026 own-SQ field receipt complete; VPM/EPM game tie, season gaps, paired/coverage audits | Certify/freeze prospective boards and competitor vintages; preserve 2026–27 |
| Completed-game player attribution | PARKED | Box/event audit and research reader | Resume after predictive VPM integration reaches its defined receipt |
| rZTOV public stat | DONE | Full stat, explorer, public `/skills/rztov` route | Maintain; 2026-27 predictive lane remains separate |
| Predictive rZTOV prior | WAITING | Frozen annual and in-season challengers | Score 2026-27 once; no publish step before the gate |
| NBA playoff-droppers model | PARKED | Walk-forward history, entering factor/role archetypes, phase-change, and common-player rotation baseline; every generic archetype update selected the null | Next candidate is exposure-conditioned series/matchup translation with zero-minute roster selection; freeze before 2027 |
| zTS / scoring value | DONE | Production zTS/SV chain and public skill routes; completed annual stability audit finds smaller changes but mixed correlation (see ZTS_STABILITY.md) | Refresh snapshots; do not retune on spent seasons; split-game repeatability requires independent game-level playtype inputs |
| Title lift / Championship Help | DONE | 2002-2026 EPM title model plus a labeled 1998-2026 lagged-DARKO research sensitivity; reader leads with salary-share Title Lift over Contract and keeps teammate accounting separate | Maintain timing labels and publish derived metrics only, never raw EPM, DARKO, or salary rows |
| From Impact to Winning | SHIP | Live RATE→GAME→SEASON→CONTRACT→CONTEXT site; its declared prototype market assumptions now also power Contract EPM v1 in Title Lift | Replace the prototype hurdle only after a no-leakage UFA market fit; verify and checkpoint current cap-value work |
| Five-Man Fit Lab | DONE | Recovered 459-lineup product plus new 2014-2026 refit | Maintain as lineup fit, never relabel as player value |
| Teammate Help / Help Index | DONE | 1997-2026 same-season retrospective teammate context and title-lift reader | Maintain; keep context separate from player credit and never call it an as-of forecast |
| STOP RAPM | PARKED | Valid 2024-26 zero-prior STOP board; personal prior lost forward test | Publish only if it fills a clear product question; no prior promotion |
| Parent ORAPM/DRAPM priors | WAITING | Strong development/stress result | Add target-covariance uncertainty and summed VPM gate; 2026 matchup blocks 2027 |
| BAM matchup prior | PARKED | Stable annual association but neutral incremental VPM result | Keep diagnostic; do not distribute to components |
| Estimated play-type scoring RAPM | DONE / DIAGNOSTIC | Owner-requested 2022–2026 direct FGA decomposition; exact additive closure and split-game receipt | See PLAYTYPE_SCORING_RAPM.md; actor attribution and full-possession extension remain unbuilt, no VPM admission |
| PBP playtype difficulty | WAITING | Own-shot proxy beats box; usage-centered overall-mix parent is incremental-null | Freeze own-shot method for untouched 2026-27; keep overall parent diagnostic-only |
| Shot-clock / grenade burden | WAITING | Useful diagnostic; little incremental gain beyond zTS | Build multi-season panel only; no current VPM admission |
| Defender and rim-block SQ | PARKED | Geometry and block-risk experiments; promotion gates failed or remain incomplete | Wait for untouched data and continuation accounting |
| Three-point difficulty | PARKED | qSQ and rich-geometry descriptive readers | Keep separate from accepted VPM target; no prior admission |
| DRL / Shapley | PARKED | State model, event credit, synergy, additive outputs | Next build is the decomposition-quality harness, not a model rewrite |
| Historical 1997-2013 RAPM | MAINTENANCE | Raw/processed reconstruction and validation framework | Rerun remaining weighted-factor solves; retain coverage caveats |
| Historical transition archive | DONE | 1997-2026 scored archive and audits | Regenerate only when the frozen classifier contract changes |
| Modern lineup repair | MAINTENANCE | Validation-first repair tooling and promotion guards | Use for audited repairs; never promote while validation issues remain |
| WNBA live metric | RUNNING | Daily box-prior board, publishing, forecast logger, and owner-directed LUMA historical bridge | Operate the frozen live build; keep aggregate historical metrics explicitly separate from event DECOMP |
| WNBA RTSS / betting research | WAITING | Preregs, minutes engine, logger, RTSS candidates | Resolve AWS access; run the frozen season-end confirmation |

## Product map

Use one public hierarchy rather than making every project compete for the home
page:

```text
nbarapm.com
├── Impact
│   ├── RAPM leaderboard
│   ├── Net DECOMP
│   ├── Gross DECOMP comparison / compatibility
│   └── VPM / Impact Monster
├── Skills
│   ├── rZTOV
│   ├── zTS / scoring value
│   └── role and shot-difficulty explainers
├── Winning
│   ├── EPM game and season value
│   ├── title lift / Championship Help
│   └── contract and playoff context
├── Lineups
│   ├── Five-Man Fit
│   └── teammate help
└── Research notebook
    ├── STOP and net-continuation diagnostics
    ├── ShotQuality challengers
    ├── DRL / Shapley
    └── historical and WNBA work
```

The `From Impact to Winning` site already supplies much of the Skills and
Winning organization. Prefer linking or integrating those existing surfaces
over rebuilding them again inside another app.

## ADHD operating rules

### Starting a work session

1. Open this file.
2. Pick exactly one of the three active projects.
3. Write one next physical action that fits inside 60-90 minutes.
4. Define the visible receipt: test output, committed doc, built artifact, or
   deployed page.
5. Stop when that receipt exists; record the next action before switching.

### Weekly review

- Move finished work to DONE even if more ideas exist.
- Move anything dependent on future data to WAITING.
- Park failed challengers with their result instead of “improving” them.
- Promote at most one item into each active slot.
- Treat a new public metric as a release with a definition, audience, URL,
  validation receipt, and maintenance owner.

### Definition of finished

A project is finished for now when it has:

- one canonical definition;
- one reproducible build path;
- one validation receipt;
- one public or internal destination;
- an explicit status label;
- a written next trigger, if it is waiting.

“There are more ideas” does not make a project unfinished.

## Do-not-start list

Until one active slot is freed, do not start:

- another public leaderboard;
- another VPM prior family selected on 2025-26;
- another ShotQuality model without a named untouched gate;
- another standalone explainer site for a metric that already has a route;
- another broad cleanup before the existing work is checkpointed.
