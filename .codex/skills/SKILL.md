---
name: six-factor-rapm
description: "Russell Thomas's Six-Factor RAPM, active NBA DECOMP, ShotQuality-EV VPM, Impact Monster, and predictive player-metric research doctrine. Use for RAPM factor interpretation; DECOMP or VPM implementation; oTS/oTOV/oREB/oSC/dTS/dTOV/dREB/dSC UI and methodology; ShotQuality targets; factor priors and alphas; EPM comparisons; future possession, lineup, or game prediction; validation and holdout design; and roadmap decisions whose long-term goal is to beat EPM on genuinely out-of-sample NBA game prediction."
---

# Six-Factor RAPM, DECOMP, and Predictive VPM

Treat this skill as the domain router and governing research doctrine. Load
only the reference needed for the task.

## North Star

Build an interpretable player metric that **beats EPM at predicting future NBA
games** on frozen, date-correct, out-of-sample tests.

Apply these implications:

- Optimize future game or possession prediction, not agreement with EPM,
  current RAPM, or retrospective player rankings.
- Use EPM as an external benchmark. Do not train impact estimates to imitate
  EPM and then claim to have beaten it.
- Preserve additive factor structure because it supplies better targets,
  priors, diagnostics, and basketball explanations—not because decomposition
  alone proves predictive superiority.
- Require a candidate to improve the summed forecast. Component fit is a
  diagnostic, not the promotion target.
- Reserve untouched future dates or seasons. A result selected on a window is
  development evidence forever.

Read [references/long-term-plan.md](references/long-term-plan.md) for the
architecture and game-prediction validation ladder.

## Working Procedure

1. Identify the exact series and artifact before interpreting or changing it.
2. Determine whether the task is explanatory, production DECOMP, VPM
   research, current-board estimation, or future-game evaluation.
3. Read the matching reference below.
4. Inspect the live config, manifest, and repo docs before quoting current
   paths, alphas, results, or promotion state.
5. Prove row algebra before interpreting coefficient sums.
6. Keep retrospective mechanism evidence separate from predictive admission.

## Series Taxonomy

Do not blend these model families in language or code.

| Series | Definition | Proper use |
|---|---|---|
| Pure Six-Factor RAPM | Native TS%, TOV%, and OREB% lineup solves, no box prior | Lineup signal and structural analysis |
| Hybrid Six-Factor RAPM | Post-solve empirical-Bayes blend of pure factors and box priors | Stabilized factor display |
| True-Prior RAPM | Prior-centered ridge solve inside the lineup regression | Prior-informed impact estimation |
| Public DECOMP | Eight point-valued additive components built from actual possession scoring | Current public mechanism breakdown |
| VPM / Impact Monster | Ten point-valued components using frozen mid/three EV targets plus actual-minus-EV residuals | Predictive research and player board |
| Alt3 EFG-value | Older atoms plus balancing-baseline family | Legacy reproduction and audits only |

Naming rules:

- `actual_*` or `pure_*` means lineup-derived pure signal.
- `box_*` means a prior estimate.
- `hybrid_*` means a post-solve stabilization blend.
- `prior_*` means a prior-centered solve.
- Never call the old Alt3 family the active DECOMP baseline.
- Call Impact Monster the ten-factor breakdown, not a second rating layered on
  top of VPM and not merely the `TS/TOV/SC` organizer.

## Possession Structure

Explain point-valued impact through three possession stages on both sides:

```text
Turnover Value -> Shot Value -> Second-Chance Value
```

```text
offense = oTOV + oTS + oSC
defense = dTOV + dTS + dSC
net = offense + defense
```

Interpret RAPM as lineup-adjusted team impact:

- `oTS` is team shot-value impact while the player is on offense, not personal
  shooting percentage.
- `oTOV` is team possession survival and shot access, not merely the player's
  box-score turnovers.
- `oSC` is direct second-chance value, not a residual bucket.
- `dTS` is opponent shot-value suppression.
- `dTOV` is possession deletion before a shot, not steals alone.
- `dSC` is opponent second-chance prevention and possession completion.

Use native-unit language for pure TS/TOV/REB files. Use points-per-100 language
only for point-valued DECOMP/VPM artifacts.

Read
[references/possession-structure-framing.md](references/possession-structure-framing.md)
for public copy and factor explanations.

## Active Public DECOMP

Use these eight player-facing components:

1. `DECOMP_RIM_FREQ`
2. `DECOMP_RIM_FG`
3. `DECOMP_MID_FG`
4. `DECOMP_THREE_FREQ`
5. `DECOMP_THREE_FG`
6. `ALT_FT`
7. `ALT_TOV_VALUE`
8. `SECOND_CHANCE_CLEAN`

The first five close exactly to first-chance EFG at the processed-row level.
With FT and turnover value they close to `FIRST_CHANCE`; adding direct
`SECOND_CHANCE_CLEAN` closes to the actual possession-scoring layer.

Treat `weighted_factors_decomp_*` as active. Treat
`weighted_factors_alt3_efg_value_*` and `weighted_factors_alt3_*` as
legacy/audit surfaces.

Read [references/decomp-and-vpm.md](references/decomp-and-vpm.md) before
changing DECOMP aliases, VPM targets, factor rollups, alphas, or publication.

## ShotQuality-EV VPM

Use ten solved components:

1. `DECOMP_RIM_FREQ`
2. `DECOMP_RIM_FG`
3. `DECOMP_MID_FG` with midrange EV substituted
4. `DECOMP_THREE_FREQ`
5. `DECOMP_THREE_FG` with three-point EV substituted
6. `ALT_FT`
7. `ALT_TOV_VALUE`
8. `DECOMP_MID_VALUE`, midrange `actual - EV`
9. `DECOMP_EFG`, three-point `actual - EV`
10. `SECOND_CHANCE_CLEAN`

Keep rim actual. Replace only the chronological first midrange/three FGA with
the frozen walk-forward EV. Carry both residuals so the ten components still
sum to actual points. Grade every model on future **actual** points, never on
its denoised EV training target.

Do not describe the accepted board as using rich geometry, defender identity,
shot-clock/grenade, continuation, or historical safe-crossfit models until a
new frozen forward gate admits one and the target lane is rebuilt.

## Prediction and EPM Rules

- Compare player metrics on identical future games and, for a player-model
  test, identical lineup/minute assumptions.
- Evaluate additive player sums separately from lineup chemistry, coaching,
  availability, and synergy models.
- Cluster uncertainty by game and compare paired errors on the same rows.
- Weight registered cuts according to a frozen rule; do not let a large cut
  silently dominate.
- Use complete EPM only as a benchmark or explicitly frozen ensemble input.
- Allow date-aware EPM `p_*` estimated skills as candidate priors when their
  as-of contract is preserved.
- Never use `eWINS` as a prior; it is cumulative EPM impact, not a predictive
  skill input.
- Report direct VPM-minus-EPM uncertainty. Do not call a point-estimate lead a
  win when the paired interval crosses zero.

## Data Admission and Leakage Guardrails

- Admit only information available at the estimate date.
- Mature forward labels fully before using them to train a prior mapper.
- Keep raw provider shots, defender joins, fold predictions, reset
  provenance, and event-level credit out of VPM.
- Admit stabilized player-season summaries only after multi-season reliability
  and incremental future-target tests.
- Do not stack nested leave-one-out encodings whose algebra can reconstruct a
  held-out outcome.
- Keep ShotQuality and NBA matchup defender assignments provider-specific;
  never coalesce them into a generic defender identity.
- Treat observed role/mechanism drilldowns as diagnostics unless they pass a
  frozen forward gate.

## Decomposition Integrity

For every change:

1. Check source-action algebra where applicable.
2. Check processed-row maximum gap, summed gap, and identity counts.
3. Solve parent and children on the same universe and sign convention.
4. Attribute small remaining coefficient gaps to separate ridge solves,
   centering, or export rounding only after row closure is proven.
5. Preserve positive-good defense in reader-facing output.

Never use a balancing residual to conceal broken row construction.

## Current Research State

Treat dated results as evidence, not timeless doctrine:

- The frozen ShotQuality-EV target beat plain RAPM on registered development
  possession forecasts.
- The accepted completed-season x2 prior lane added a small positive increment
  beyond EV no-prior.
- EPM12 x2 is the leading research base on the four registered development
  cuts and uses only as-of predictive `p_*` skills.
- The direct pure-VPM versus complete-EPM evidence must be described with its
  paired interval; earlier evidence did not justify an unconditional victory
  claim.
- The 2025-26 confirmation tiles are spent. The next promotion receipt is an
  untouched 2026-27 window or another predeclared future holdout.

Inspect the VPM config and manifest before quoting exact numbers.

## Reference Map

Load only what the task requires:

- [references/decomp-and-vpm.md](references/decomp-and-vpm.md): active DECOMP,
  ten-factor VPM, alphas, product rollups, ownership, and current evidence.
- [references/long-term-plan.md](references/long-term-plan.md): predictive
  architecture, game-level EPM benchmark, validation ladder, and roadmap.
- [references/technical-methodology.md](references/technical-methodology.md):
  regression machinery, native six factors, ridge, and historical SQ method.
- [references/possession-structure-framing.md](references/possession-structure-framing.md):
  public possession-chain explanations.
- [references/interpretation-guide.md](references/interpretation-guide.md):
  player profiles, archetypes, and audience caveats.
- [references/ts-decomposition.md](references/ts-decomposition.md): historical
  ShotQuality `SQ + Contest/Make + FT` decomposition.
- [references/offensive-value-notes.md](references/offensive-value-notes.md):
  current playtype, turnover, load, and scoring-prior research.
- [references/2026-single-season-true-prior.md](references/2026-single-season-true-prior.md):
  frozen 2026 single-season beta and artifact schema.
- [references/legacy-alt3.md](references/legacy-alt3.md): old Alt3 artifacts and
  compatibility rules.
- `references/factor-oTS.md`, `factor-oTOV.md`, `factor-oREB.md`,
  `factor-dTS.md`, `factor-dTOV.md`, and `factor-dREB.md`: native-factor deep
  dives.

## Repository Boundary

Keep the interface narrow:

- RAPM owns canonical PBP, possession construction, DECOMP row algebra, solver
  helpers, and the isolated VPM target-writing adapter.
- The sibling VPM research repo owns frozen EV inputs, immutable experiment
  configs, priors, evaluation, holdouts, evidence, and boards.
- Route collaborators through `vpm/run_harness.py`; do not duplicate the RAPM
  pipeline or revive legacy root Alt3 harnesses.

For live implementation work, read repo-relative
`nba_pipeline/docs/PUBLIC_DECOMP.md`,
`nba_pipeline/docs/VPM_REPRODUCTION.md`, and the sibling VPM
`vpm/docs/RESEARCH_HARNESS.md` in addition to this skill.
