# Pregame VPM implementation status

Owner authorized implementation October21,2026 after the full predictive-stat library completed. This is the existing VPM RESEARCH project. The [training plan](VPM_PREGAME_PRIOR_PLAN.md) still governs the comparisons. No improved VPM or selected-model replacement has been established.

## Implemented

- [Neutral query adapter](../scripts/build_pregame_neutral_skills.py) and [compiled kernel](../scripts/pregame_neutral_kernel.cpp): query every player/game using legally dated annual OU recipes, remove contextual/early-season prediction offsets and restore the dated league center. Zero-opportunity and missing-measurement queries do not commit an observation, time or season crossing. Positive observations reproduce the reference filter's posterior states/variances.
- [Validated source bridge](../scripts/build_validated_pregame_source.py): pinned2014–2026 portion of the new core source, exact identities, unsafe possession denominators missing, and the existing personal-foul counter only on matching identities. Do not silently replace the PF counter with a sum of foul types.
- [Complete-adapter perturbation audit](../scripts/audit_pregame_neutral_prefix.py): change target-date and later FT makes, three-point makes and assists, rerun correction/context construction and filtering, compare neutral pregame states. Counterfactual exports stay diagnostic and never replace originals.
- Sibling VPM `pregame_dynamic.py`: dynamic possession feature design with joint O/D player residuals and free nuisance terms profiled out. Full covariance, positive-good defense signs, /100 feature units and global M0 weight normalization are preserved. Direct joint/profile parity and the static-design limit pass; it does not regress onto precomputed RAPM labels.
- Sibling immutable v66 / `pregame-vpm --stage prepare|pilot-audit|expand-core`: source locks, neutral-core assembly, exact ten-player joins and whole-game exclusions for identity/scoring disagreement. It does not fit an impact mapper or admit new features.

## Source issue exposed by the pilot

The legacy monthly source contains49 positive-exposure player/game rows across19 games absent from the pinned validated core census. One example assigns Mahinmi, Isaiah Thomas, Beal, Gary PaytonII and Thomas Bryant99 possessions each in `0021900656` onJanuary22,2020, as well as their actual game `0021900658`. The newer core contains those five players only in their actual game. These are false observations if consumed as zero-stat playing exposures.

Initial legacy-source replay v2 therefore remains unadmitted. Its assist-state differences arise from same-date batching across those legacy rows; do not waive all posterior-parity checks merely because any same-date pair exists. Source v1 and neutral replay v4 now use validated identities/counts; earlier v1 had a receipt-serialization error and v3 had a nullable-home dtype error. Preserve earlier diagnostic outputs and use only the named current package.

The new source changes observation counts/exposures relative to the selected library. **Annual parameters are transported from the old legal vintages, not refitted on these corrected counts.** This is a declared source/estimator change. A/B timing controls must use the same transported source/estimator; comparison with selected v61 additionally includes source/estimator changes. A source-clean upstream refit comparison remains necessary before promotion. This work does not silently repair or rewrite selected ratings/parameters.

## Verified pilot

All17 neutral core families at annual vintage2019 emit24,087 player/game queries fromOctober22,2019 throughOctober11,2020. Their reconstructed observation states agree with the reference replay on the new source to1.12e-16 or better. Missing/zero-opportunity rows are queries, not observations; FT has10,353 such rows and threes5,106.

End-to-end perturbation checks leave14,129 earlier-or-same-date predictions unchanged for each of FT%,3P% and assists (maximum difference0), while later states do change. This exercises the complete context/correction/filter adapter rather than only the kernel.

The pilot sees229,352 native actual-point rows across1,142 games. Exactly1,128 games /226,492 rows survive the common-panel gate:3 games have missing player/game states and11 more disagree between the native target sum and validated player-point ledger. The latter are disagreements pending independent scoreboard adjudication, not proof that either ledger is universally correct. Exclude entire games from both arms while unresolved; never score a selectively surviving part or invent zero states. Exact game lists and hashes are in the generated v66 audit.

Tests:26 focused RAPM tests (the previous17 plus9 neutral/source semantics tests) and7 new sibling dynamic/admission tests pass:33 in total. Real neutral/source/native audits are additional checks.

## Expanded core and independent scoring audit

All187 family/vintage exports completed and hashes were verified:17 families across annual vintages2015–2025,301,706 distinct player/game queries in the assembled11 panels. The core-only native target join covers2,781,191 rows across14,128 games before common-panel exclusions. These counts do not make the missing tracking/entrant/static-cut adapters ready.

The [independent scoreboard audit](../scripts/audit_pregame_target_scoreboards.py) reads terminal score strings in clock order from the pinned local2016–2026 RS/PS raw files, separately from native actual-point sums and provider player-point sums. It requires a terminal end-of-regulation/overtime snapshot; an earlier partial score is not filled forward to invent a terminal result.

Of14,128 native games,13,954 match that terminal scoreboard;148 disagree and26 have no usable terminal snapshot.85 games match the native target to the terminal scoreboard despite player-point totals disagreeing. Thus the original player-point ledger was too strict for those85: unattributed scoring can make a player sum incomplete without making the native game total wrong. Conversely, agreement between native and player sums alone cannot admit a game whose terminal scoreboard disagrees.

The v66 per-vintage audits remain immutable conservative first-pass receipts. The independent audit is `results/estimated_skills/pregame_scoreboard_audit_v1/` with game lists, exact gaps,22 raw and22 target file hashes and a receipt. The sibling `pregame-native-admission` stage now applies this gate to **all** games, preserving the12 missing-player-state exclusions:13,942 games admitted and186 excluded (148 native/scoreboard disagreement,26 unresolved scoreboards,12 missing states). Its `native_admission/games.csv` and receipt preserve exact reasons and source hashes.85 games are correctly recovered despite incomplete player-point sums.

Source completeness remains a separate gate:211 games have incomplete/unresolved player-point attribution against the terminal scoreboard. Before calibration, investigate or mask affected raw stat measurements rather than treating partial counts as fully measured zeros; retain their pregame queries. Extend that source check over the2014–2015 upstream warm-up period as well. Independently recount unresolved/mismatched targets or retain identical whole-game exclusions in every arm. No new VPM has yet been fitted.

## Artifacts and current run

- RAPM `results/estimated_skills/pregame_source_validated_v1/`:356,856 core queries2014–2026, source/identity audit,94 missing PF counters and145 rows with unsafe rate denominators.
- RAPM `results/estimated_skills/pregame_neutral_v4/`: completed187 neutral core family/vintage exports/receipts and `run_status.json`; all output hashes verified.
- RAPM `results/estimated_skills/pregame_neutral_prefix_v1/receipt.json`: passed complete-adapter perturbation audit. Reference/counterfactual files are diagnostic only.
- Sibling `vpm/results/harness/vpm_pregame_neutral_core_v1/`: immutable config/source locks, assembled core panels and per-vintage ten-state/score-ledger audit. Log `vpm/results/pregame_core_v66.log`.
- Its `native_admission/`:13,942-game common admission list,186 exclusions, source-completeness flags and hashed receipt. It explicitly remains unready for a complete selected-recipe fit until observation/adapter gates pass.

These generated artifacts remain ignored. The separate1996–2026 full-history library is retained. Its context-free production forecasts are not silently substituted for neutral inputs, and its fixed-budget decay fits are not called optimal.

## Required continuation

1. Finish source/scoring admission using the independent terminal-scoreboard audit. Core export/hashes are complete. Independently adjudicate excluded target/scoreboard games and partial observations without changing archives.
2. Finish game-date rim/DFGA, corrected foul/contested-rebound tracking, retained365-day MPG and entrant adapters. Apply model-aware aggregate fallbacks before training-only preprocessing; an observed zero is valid. The17-core panel is not the complete selected VPM recipe.
3. Implement neutral annual-cut query snapshots and deduplicated eligible cohort assignments for the matched static control. Preserve source/estimator changes as separate controls.
4. Register the dynamic calibration experiment separately after the evidence gate passes; add season/home and calibration-decay comparisons in the plan's order. The dynamic solver module is ready for integration, but no real impact mapper has been fitted in v66.
5. Replay rolling games and the existing seven-test benchmark, retain exact Net lanes, and report paired/season uncertainty. Historical2020–2026 is spent;2027 remains protected. An improvement is an empirical outcome, not an assumption.

Selected clean v61, previous comparisons, quarantined v64 and separate v65 remain unchanged. No public deployment or live-refresh readiness is claimed.
