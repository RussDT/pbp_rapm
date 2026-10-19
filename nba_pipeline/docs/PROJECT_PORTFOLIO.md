# NBA Analytics Project Portfolio

Last reviewed: 2026-10-18 (owner requested per-game matchup defensive VPM features)

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

### 2. RESEARCH — NBA predictive VPM defensive matchup features

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
