# Preregistration: One-Shot 2026 RTSS Component-Prior Confirmation

Frozen August 10, 2026, before any 2026 RTSS evaluation. This document is the
binding contract for Phase 2 of `WNBA_BETTING_PROGRAM.md`. Amendments after
this date must be listed in the Amendments section with a date and a reason,
and no amendment may be informed by 2026 outcomes.

## What is being confirmed

Whether the frozen RTSS tracking corrections to the factor-specific WNBA box
priors improve future-possession prediction on the untouched 2026 season,
enough to promote them into the 2027 live prior.

## Fire conditions (all required)

1. The 2026 WNBA regular season is complete.
2. The RTSS mirror contains every ended 2026 regular-season game with 2D
   markings (follow-up `aws s3 sync` after reauth; see
   `WNBA_RTSS_FEATURE_AUDIT.md`). The 2026 mirror is consumed only by the
   secondary descriptive nowcast diagnostic (step 4); the primary forward
   endpoint (step 3) never reads it. The condition remains mandatory so the
   secondary diagnostic is always reported alongside the primary verdict.
3. The harness `wnba_test/run_wnba_rtss_confirmation.py` passes its
   own as-of audit (no 2026 rows in any fit input). The audit hook fails
   closed: any fit-input frame it cannot positively classify by WNBA game ids
   or a recognized season column raises instead of passing silently.
4. The sentinel `wnba_test/Results/rtss_confirmation/FIRED` does not exist.
   The harness writes it on completion. It fires once.

## Frozen candidate bundle

From the symmetric leave-one-season-out CV recorded in
`WNBA_RTSS_PRIOR_RESEARCH.md` (all of 2023-2025 is development data):

Tier A (3/3 folds, pooled RMSE and correlation both improved):

| Component | Side | Dev RMSE change | Lane |
| --- | --- | ---: | --- |
| THREE frequency | defense | -5.81% | compact tracking |
| Second chance | offense | -5.09% | rebound opportunity / crash |
| FT frequency | defense | -2.54% | compact tracking |
| MID frequency | defense | -1.31% | compact tracking + q context |
| MID finishing | offense | -0.79% | creation/touch structure |

Tier B (2/3 folds):

| Component | Side | Dev RMSE change | Lane |
| --- | --- | ---: | --- |
| FT total | defense | -1.99% | compact tracking |
| RIM frequency | offense | -0.42% | drives + creation roles |
| MID frequency | offense | -0.52% | creation roles only |
| THREE frequency | offense | -0.59% | creation + screens + q |

### Exact recorded CV lane per component (binding)

The lane keys below are the recorded `selected_prior_lane` values in
`wnba_test/Results/rtss_prior_cross_validation/best_by_component.csv`. The
harness asserts at runtime that its hard-coded lane spec equals this recorded
file and fails loudly on any drift.

| Component prefix | Side | Tier | Recorded CV lane |
| --- | --- | --- | --- |
| `ALT_EFG_THREE_FREQ` | def | A | `tracking_core` |
| `SECOND_CHANCE_CLEAN` | off | A | `tracking_core` |
| `ALT_FT_FREQ` | def | A | `tracking_core` |
| `ALT_EFG_MID_FREQ` | def | A | `tracking_core_plus_q` |
| `ALT_EFG_MID_FG` | off | A | `tracking_core` |
| `ALT_FT` | def | B | `tracking_core` |
| `ALT_EFG_RIM_FREQ` | off | B | `tracking_core` |
| `ALT_EFG_MID_FREQ` | off | B | `creation_roles_only` |
| `ALT_EFG_THREE_FREQ` | off | B | `tracking_core_plus_q` |

Note on the q quarantine: the quarantine below applies to raw `qSP-qSQ` used
as a shooter adjustment. It does not apply to q-level context features inside
a recorded CV-selected lane — the defensive MID frequency lane is
`tracking_core_plus_q` and the offensive THREE frequency lane is
`tracking_core_plus_q` per the recorded CV selection, and both remain in the
frozen bundle as recorded.

`ALT_FT_FREQ` is an exact additive child of `ALT_FT`, so its correction is
applied to the `ALT_FT` prior; every other candidate corrects its own additive
solve component.

Permanently excluded from this confirmation:

- offensive RIM finishing and THREE finishing (box-only, per development
  decision);
- raw `qSP-qSQ` shooter adjustment (quarantined until the provider timing
  audit and a frozen season-relative transform exist);
- any monolithic/direct-total RTSS prior (rejected lane).

## Frozen procedure

(As amended August 10, 2026 — see Amendments. The primary endpoint is the
strictly forward fold 2025 -> 2026; the in-sample 2026 nowcast is a secondary
descriptive diagnostic.)

1. Box backbone: factor-specific box priors trained strictly on earlier
   seasons, exactly as the live `5000/40000/1.5` build constructs them. The
   primary endpoint uses the rating-season backbone `box_priors_2025.csv`
   (trained on 2022-2024); the secondary nowcast diagnostic uses
   `box_priors_2026.csv` (trained on 2022-2025).
2. RTSS corrections: fit on cross-fitted box residuals with the per-candidate
   lanes above, exposure-weighted, standardized within season, multi-year
   stabilized. No 2026 rows in any fit.
   - Primary: residual fits use 2023-2024 only (strictly before rating season
     2025); the fitted corrections are applied to 2025 RTSS features. The
     2026 RTSS mirror is not consumed by the primary endpoint at all.
   - Secondary nowcast: residual fits use 2023-2025; corrections are applied
     to 2026 RTSS features (prior application only, mirroring the live
     nowcast contract).
3. PRIMARY endpoint (the promotion decision): the strictly forward fold
   2025 -> 2026, the same construction as `tune_wnba_component_alphas.py`.
   Solve the nine additive components on rating season 2025 (season type ALL)
   around (a) box-only priors and (b) RTSS-augmented priors, using the frozen
   live alphas `5000/40000` and prior scale `1.5`, with the rating-season
   mean possession intercept. If Phase 2's development-fold defense-alpha
   extension (below) selects a new defense alpha before firing, that alpha is
   used for BOTH arms symmetrically. Score both arms' summed player
   coefficient maps against actual 2026 possession points: joint
   component-sum future-possession MSE, paired and game-clustered
   (bootstrap over 2026 games).
4. SECONDARY descriptive diagnostic (never a promotion input): the in-sample
   2026 nowcast comparison — solve the nine components on 2026 around each
   arm (built per step 2's secondary lane) and score on the same 2026
   possessions. This mirrors the live nowcast prior-application contract and
   is reported for context only. It cannot rescue a failed arm or demote a
   passing one.
5. Prespecified arms, evaluated in this order on the primary endpoint:
   - Arm 1: Tier A only (five component-sides corrected).
   - Arm 2: Tier A + Tier B (nine component-sides corrected).
6. Promotion rule: an arm passes if it beats box-only joint MSE on the
   PRIMARY forward endpoint. If both pass, promote the arm with the lower
   MSE. If only one passes, promote it. If neither passes, promote nothing
   and record the null. No post-hoc per-component cherry-picking; the arms
   above are the only units of promotion.
7. Per-component RMSE/correlation on 2026 is reported as secondary,
   descriptive evidence only. It cannot rescue a failed arm or demote a
   passing one.
8. Output: `wnba_test/Results/rtss_confirmation/` with a JSON verdict,
   the paired game-clustered bootstrap CI, and the sentinel file. Update
   `WNBA_RTSS_PRIOR_RESEARCH.md` and `WNBA_BETTING_PROGRAM.md` with the
   verdict in the same pass.

Dev smoke: `--dev-smoke` runs the identical pipeline shape one year earlier —
primary forward fold 2024 -> 2025 (corrections fit on 2023 residuals only,
applied to 2024 features; backbone `box_priors_2024.csv`) plus the 2025
nowcast diagnostic — writing to `Results/rtss_confirmation_dev_smoke/`. It is
development evidence only and never touches 2026 data or the sentinel.

## Defense-alpha grid extension (same cycle, separate decision)

The live defense alpha `40000` sits on the audited grid edge. Extend the grid
(40000, 60000, 80000, 120000, 160000) scored on the development forward folds
`2023->2024` and `2024->2025` only, with the joint component-sum test. 2026 is
excluded from this selection. If a higher alpha wins development, it becomes
the live defense alpha for 2027 and (per step 3) is used symmetrically in the
confirmation solve. This selection must be run and frozen before the
confirmation fires.

## What promotion means

Passing candidates enter the 2027 live prior as cross-fitted residual
corrections on the box backbone — the box prior remains the backbone, RTSS
remains a compact correction, and the 2027 season becomes the next untouched
holdout. Nothing changes in the live 2026 daily update.

## Amendments

### August 10, 2026 — design-review amendment (primary endpoint made strictly forward)

A design review found the original draft internally inconsistent: step 3 said
"solve the nine additive components for 2026 around (a) box-only priors and
(b) RTSS-augmented priors" while step 4 pinned the primary endpoint to "joint
component-sum future-possession MSE ... the same construction as
`tune_wnba_component_alphas.py`". Those two statements conflict: the tuner's
construction is strictly forward (solve components on rating year Y with
priors trained strictly before Y, then score the summed coefficient maps on
year Y+1 actual possessions), whereas solving on 2026 and scoring the same
2026 possessions is in-sample. The inconsistency is resolved in favor of the
strictly forward construction:

- PRIMARY endpoint is now the forward fold 2025 -> 2026 (steps 1-3 above):
  components solved on rating season 2025 around each prior arm, corrections
  fit on 2023-2024 residuals only and applied to 2025 RTSS features, scored
  on 2026 possessions with the paired game-clustered bootstrap. The primary
  endpoint does not consume the 2026 RTSS mirror at all.
- The original in-sample 2026 comparison is demoted to a SECONDARY
  DESCRIPTIVE diagnostic (step 4). It mirrors the live nowcast
  prior-application contract but is not the promotion decision. Because it
  still requires the complete 2026 mirror, fire condition 2 stays mandatory,
  scoped to this secondary arm.
- The dev smoke is reshaped to the same forward shape one year earlier
  (2024 -> 2025 primary fold, corrections fit on 2023 only; see "Dev smoke"
  above). The previous dev-smoke result (tier_a -14.97e-6,
  tier_a_plus_b +31.14e-6, in-sample-2025 shape) is obsolete and superseded.
- This amendment also records the exact lane-per-component table (including
  that defensive MID frequency is `tracking_core_plus_q`) and the runtime
  assertion that the harness lane spec equals the recorded
  `best_by_component.csv`, and clarifies that the q quarantine covers raw
  `qSP-qSQ` as a shooter adjustment, not q-levels inside a recorded
  CV-selected lane.

This amendment was made before any 2026 evaluation ran: the harness `--check`
is BLOCKED (2026 season incomplete, mirror incomplete) and real mode has never
fired (no FIRED sentinel exists). It is informed only by design review of the
draft's internal inconsistency, not by any 2026 outcome.
