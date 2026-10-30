# Model Assurance Ledger

October15 repair: [SQ temporal reconstruction](VPM_SQ_TEMPORAL_REPAIR.md)
(v39/v40) now passes computational prefix/lineage checks and repeats the seven
tests. Scores barely change; game performance versus EPM is unresolved.
Retrospective selection, provider-vintage and oracle-minute limits remain;
this is research evidence, not promotion.

October 15 correction: the [predictive VPM field benchmark](VPM_FIELD_BENCHMARK.md)
has an upstream SQ full-panel-prior lookahead finding. Its dates/algebra audit
is not leakage certification. Preserve the outputs as retrospective diagnostics;
clean temporal SQ reconstruction and a new-version rerun are required.

Last reviewed: 2026-10-15 (VPM SQ repair; other model rows retain their dated evidence)

This is the portfolio-level trust register for NBA model outputs. It answers a
different question from the methodology documents: **what can we responsibly
claim about an output today, and what evidence is still missing?**

It is deliberately short. Detailed methods, commands, artifacts, and numeric
receipts stay in each model's canonical document. Update this ledger whenever
a model is promoted, rejected, materially rebuilt, or given a new public use.

## Read the statuses correctly

| Status | Meaning | May be public? | May drive a production decision? |
|---|---|---|---|
| **CURRENT** | The artifact has a documented contract, reproducible build path, and the validations required for its stated use. | Yes, with its stated scope. | Yes, only within that scope. |
| **RESEARCH** | Useful and inspectable evidence, but an open confirmation, comparison, or integration gate remains. | Yes, only if labelled research. | No silent promotion. |
| **DIAGNOSTIC** | Explains a mechanism or evaluates a hypothesis; it is not an estimate to rank players or change the main model. | Usually as an explicitly limited study. | No. |
| **PARKED / REJECTED** | The candidate failed its relevant gate, is blocked by missing data, or has already answered its question. | Only as historical context. | No. |
| **RECOVERY** | A preserved output or rebuilt product whose original training proof is incomplete. | Yes, with recovery provenance. | Not as evidence of an unobserved training claim. |

These are not quality grades. A **CURRENT** accounting model can still lack a
future-prediction claim; a **RESEARCH** model can have strong development
evidence but still need untouched confirmation.

## What every important output must prove

Use these five receipts rather than relying on a single “validated” label.

| Receipt | Question | Minimum evidence |
|---|---|---|
| **Data** | Are the source rows complete, correctly joined, and date-safe? | Coverage/ambiguity report; source and as-of dates where relevant. |
| **Identity** | Does the arithmetic mean what the output says it means? | Row and player-table closure; sign/unit checks. |
| **Rebuild** | Can the same artifact be regenerated? | Versioned command/config, input manifest, and output hash or deterministic audit. |
| **Forward test** | Does it improve the predeclared real target on unseen time? | Same-sample baseline, held-out score, and paired uncertainty. |
| **Use boundary** | Is the public/product claim no broader than the evidence? | Plain-language label, known failure modes, and promotion gate. |

An empty receipt is a known gap, not permission to infer that the test passed.

## Current model map

| Surface | Status | What is established | What is not established / next gate | Canonical receipt |
|---|---|---|---|---|
| **NBA RAPM parent** | CURRENT | Possession processing, lineup construction, solver conventions, and reproducible model definitions are documented. | It is an adjusted impact estimate, not a direct future-game forecast or causal player value. | [RAPM Methodology](./RAPM_METHODOLOGY.md) |
| **21st Century RAPM age-27 career board** | RESEARCH | Fixed season environments, frozen possession-derived age curve, negative-LAKER fringe center, and one-season young-player precision are reproducible. The adaptive `4x/3x` O/D penalties win on 1998-2021 first-to-second-season development transitions and improve again in 2022-2025 stress. | The adaptive gate is an annual attenuation proxy, not an exact future-possession comparison of the full career solve. Keep fixed-alpha and zero-prior controls and label the adaptive board research. | [21st Century RAPM](./THIRTY_YEAR_AGE_ADJUSTED_RAPM.md) |
| **Clock-aware RAPM rubberband adjustment** | CURRENT | The smooth alpha-100 architecture beat no adjustment and legacy buckets in every 2022-2026 forward stress season. After historical score repair, a frozen 1997-2001 curve still helped every era, but rolling five-season re-estimation added value. | It is a nuisance game-state adjustment, not causal effort or a standalone forecast. One surface is fit per solve; a 1997-2026 run is pooled, not era-specific. | [Rubberband Adjustment](./RUBBERBAND_ADJUSTMENT.md), [Era Stability](./RUBBERBAND_ERA_STABILITY.md) |
| **Global RAPM home-court adjustment** | RESEARCH / PROMOTION CANDIDATE | Repaired venue labels resolve 99.3%-99.8% of 1997-2026 possessions. Smooth+home beat smooth across every broad era and removed most home/away residual bias with negligible player-board movement. | The increment is small and the coefficient declines materially over time; use time-local windows and keep rest/travel/arena effects out until separately tested. | [Home-Court Adjustment](./HOMECOURT_ADJUSTMENT.md) |
| **Net-continuation DECOMP** | CURRENT for explanatory use | Exact child, kappa, second-chance payback, and grand identities are enforced; all 26 historical five-year windows are reproducible. Owner selected it as the forward DECOMP on 2026-09-03 because it separates miss/rebound opportunity from conversion and prices turnover loss consistently. | Build the current 1Y-5Y release family and migrate every consumer reversibly. This status does not promote net VPM or claim an untouched predictive win. | [Net-Continuation DECOMP](./NET_CONTINUATION_DECOMP.md) |
| **Gross eight-component DECOMP** | CURRENT compatibility surface | The five first-chance children close exactly to first-chance EFG; the full eight-component artifact preserves its parent accounting contract. | Retain for rollback and matched comparison while Net DECOMP migration completes; do not start new products on it. | [Gross DECOMP](./PUBLIC_DECOMP.md) |
| **VPM EV target layer** | CURRENT | The EV-adjusted target has identity-gated coverage and machine-checked row/tree closure; the target builder is reproducible. | This proves target construction, not that VPM beats EPM at the game level. | [VPM Reproduction](./VPM_REPRODUCTION.md#current-validation) |
| **1997-2015 Russell-v2 SQ bridge** | RESEARCH | A chronological native-v2 overlap gate selects the v2-distilled midrange stack and explicit portable-v3 three fallback. All 2,583,850 emitted targets are unique, complete, and close exactly. | It is a retrospective bridge, not untouched confirmation or a completed historical VPM. Keep native v2 from 2016 onward and use season effects across the nonzero 2015/2016 seam. | [Russell v2 Historical Bridge](./RUSSELL_V2_HISTORICAL_BRIDGE.md) |
| **VPM accepted v2 x2 board** | RESEARCH | Frozen prior-centred player-board recipe; forward actual-possession evidence beats no-prior on its selected development cuts. | The direct VPM-versus-EPM interval crosses zero; no untouched same-game VPM-versus-EPM result yet exists. | [VPM Reproduction](./VPM_REPRODUCTION.md#hardened-research-harness) |
| **VPM EPM12 x2 board / public VPM Lab** | RESEARCH | The strongest current research base: +153.02e-6 beyond EV no-prior across four development cuts, 4/4 positive, with paired game-clustered interval [+73.38, +233.87]. Ten-factor identities close. | The cuts are spent development evidence. Freeze and score the predeclared 2026-27 same-game VPM-versus-EPM test. Keep the jumbo-defense warning visible. | [VPM Reproduction](./VPM_REPRODUCTION.md#epm12-predictive-rate-development-base) |
| **zTS / SV per 100 reader surface** | CURRENT | The production formula and intended interpretation are fixed: role-adjusted shooting efficiency and an SV/100 translation. | It is not ShotQuality, playmaking, total offense, or RAPM; new role difficulty candidates need fresh-season confirmation. | [RAPM Methodology](./RAPM_METHODOLOGY.md) |
| **EPM game and season value translation** | CURRENT | Exact PBP exposure and date-stamped EPM inputs; 2014-25 season calibration and 2022-25 held-out game-link scoring. | It is a standardized EPM translation, not an availability, matchup, or player-win forecast; replacement level remains a sensitivity. | [EPM Game Value](./EPM_GAME_VALUE.md) |
| **Five-Man Fit Lab, recovered table** | RECOVERY | The deployed table and its algebraic display contract were recovered; the output is inspectable. | Original training pipeline and coefficients were not recovered. | [Five-Man Fit Lab](./FIVE_MAN_FIT_LAB.md) |
| **Five-Man Fit Lab, historical refit** | RESEARCH | Forward-only development selected a usage construction model; later 2024-26 readout is positive. Defensive-construction candidate failed and is excluded. | The later readout is no longer pristine confirmation. Keep it a lineup tool, not a player metric or causal fit claim. | [Five-Man Fit Lab](./FIVE_MAN_FIT_LAB.md#2014-2026-historical-epm-refit) |
| **STOP RAPM and EPM-skill prior** | RESEARCH | Possession union/algebra and prior development comparisons are documented. | Component weights were selected on 2024-26 folds; 2027 is required before promotion. | [RAPM Methodology](./RAPM_METHODOLOGY.md#4d-stop-guaranteed-zero-possession-rate) |
| **Shot-clock, playtype, defender, rich-geometry, and other context challengers** | DIAGNOSTIC / PARKED | Several have useful mechanism or development receipts; failed or inconclusive increments are retained rather than hidden. | None may enter production zTS, Net DECOMP, or VPM until its stated fresh forward and integration gates pass. | [Docs index](./README.md) |
| **WNBA clean DECOMP and box-prior board** | CURRENT within stated league scope | The live board uses frozen season-forward box-prior rules and preserves additive identities. | RTSS and component-alpha challengers remain research; 2026 confirmation rules are separately fenced. | [WNBA Metric Build Status](./WNBA_METRIC_BUILD_STATUS.md) |

## The one north-star gap

The principal missing receipt is not another component ablation. It is a fair,
frozen **same-game VPM versus EPM** test:

```text
as-of player estimates + shared availability/minutes
                ↓
identical future NBA games
                ↓
predeclared game-level score + paired game-clustered uncertainty
                ↓
VPM − EPM result, with no post-holdout model changes
```

Until that report exists, VPM can be described as a promising, additive
research player-quality estimate with real forward possession evidence—not as
a demonstrated replacement for EPM or a proven game-prediction winner.

## Operating rhythm

At the start of any substantial model task, add a one-line entry to the
project handoff or pull request answering:

```text
Surface | status now | receipt being improved | holdout used? | decision this can change
```

At completion, update the relevant row above only if the status or a stated
gate changed. Link the detailed report rather than copying numbers here.

Before publishing or refreshing a reader surface, use this release check:

1. Does the output have an input/as-of-date manifest and a reproducible command?
2. Do row-level and displayed additive identities close?
3. Is its status label shown wherever a reader could mistake research for a
   settled rating?
4. Is the precise public claim no broader than the best completed forward test?
5. Did a new result spend a holdout? If yes, record it and name the next
   untouched gate before further tuning.

## What this ledger deliberately does not do

- It does not rank ideas by novelty or decide every research priority; use the
  [Project Portfolio](./PROJECT_PORTFOLIO.md) for work-in-progress limits.
- It does not replace technical reports, test suites, manifests, or source
  inspection.
- It does not convert descriptive, retrospective, or provider-derived outputs
  into predictive evidence.

The goal is a calmer operating picture: one place to see what is real, what is
promising, what is blocked, and exactly what would change that answer.
