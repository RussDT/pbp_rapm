# Codex Scratchpad

Short-term session memory. Durable rules belong in AGENTS.md / CLAUDE.md.

## 2026-10-19 — Fixed-fouls prior inspection page added

- Owner requested the fixed-fouls priors. New local8766/priors.html defaults to contestedDREB skill-only SPM, with totalDREB/previous selectors and exact blended-prior toggle. Shows prior O/D/total beside final O/D/VPM; all585 recent/1556 universe and sorted/filtered CSV available. Jokic prior7.0565/2.5779/9.6344; Wemby1.8264/6.0041/7.8305. Beringer own1.8724 versus combined−2.1735 at.635 background weight illustrates first2000-possession blend.
- Renderer checks all three models' exact IDs, finite inputs, sums and blended-prior identities. Browser verifies defaults/search/model/basis/universes and screenshot. Final-rating page links to priors. Original reader HTML/source archived in ignored reader_before_priors; curated summary records UI-only hash change and new priors_view_manifest. No model/rating/forecast changes, refits, new results, public deployment or2027 outcomes. Receipt VPM_FIXED_FOULS_CONTESTED_DREB.md.

## 2026-10-19 — Fixed-foul and contested-DREB Parent rerun completed

- Owner requested corrected shooting fouls, contested DREB replacing total DREB, previous repaired comparison and ratings. Separate immutable sibling v52 has seven active numeric defense fields and exact removal of totalDREB+flag. Fixed4000/8000 TD275 (TD550 check), actualFT, own scale1, no tuning. Stress game13.960006 contested /13.969975 total /13.965886 previous; all seven point estimates favor contested versus previous but all seven paired MSE intervals cross zero. Core low-continuity team-win and EPM team-net remain better.
- Native423644 future possessions: contested-total MSE gain57.8e-6[25.0,88.8]e-6 atTD275;50.4e-6[22.9,78.6]e-6 atTD550. Small historical possession gain, not established game superiority.28 focused tests plus independent185-source/24-mapper/154-score/105-pair audits pass. Core/benchmark forecasts preserved; no2027, public deployment or component promotion.
- Current Sep1 cut585 recent/1556 universe, still high bigs (Jokic10.765,Wemby9.517,RWIII5.599,Clingan5.421,Reed5.090). New local8766 three-choice reader/CSV checked via browser; old8765 remains flagged pre-repair. Servers intentionally stay available. Receipt `VPM_FIXED_FOULS_CONTESTED_DREB.md`; curated sibling `fixed_fouls_contested_dreb_summary.json`. Keep Parent/component scope clear and actualFT explicit.

## 2026-10-19 — Defensive overlap and shooting-foul extraction corrected

- Owner suspected collinearity and asked about contested DREB logs.638-player exposure-weighted calibration correlations: blocks/contests.735, blocks/rim-saved.682, total/contestedDREB.920. Contested DREB exists in dated DREB_CONTEST and was tested as an addition to totalDREB (devRMSE13.7788→13.7443), not replacement; the original rim/other-foul recipe scored13.7298. Correlation is not proof of double-counting; big-prior jumps already occur in the core without rim fields.
- Found real source bug: archive foul action-subtype is zero, so subtype2 extraction made every shootingPF count/coefficient zero. Original seven-field claim was wrong: six numeric fields were active. New separate `defense_bank_v51_shooting_pf_repaired` parses S.FOUL descriptions; independent26-file recount verifies327958 events/355432 player-games and244909 packet cells. Other source lanes identical;26 tests pass. Preserve old source/code snapshot and source_field_issue.json. Current reader/manifest disclose defect; original ratings/forecasts are NOT refitted. Do not treat old zero-field source audit as validated shooting-foul measurement. Receipt `VPM_DEFENSIVE_FEATURE_OVERLAP.md`.

## 2026-10-19 — Big-player prior shift diagnosed

- Owner asked why bigs jump versus previous repaired VPM. Exact saved-rating bridge atSeptember1 holds lineup data/4000-8000/TD550/SQ/entrants fixed: changed long-SPM prior mapping plus core-feature replacement already raises D Wemby3.807→6.997, Reed1.741→4.025, RWIII1.917→4.176, Clingan1.229→2.803. Expected-FT increments are only+.052,+.009,−.054,+.168. Wemby D prior1.742→6.327; O prior changes also raise bigs (ClinganO.619→2.210; OREB mapper slope.168→.540). This identifies the prior pull, not proven positional overrating or proof old ratings are correct. Component architecture is not involved. Expanded rim/foul/decay changes are smaller in these examples; all-player path closes exactly.
- Added explanation to `VPM_MODEL_MAP.md`; private sibling `previous_to_parent_prior_changes.csv` and `prior_shift_audit.json` preserve all1556 exact-cut comparisons/source hashes. No fit, tuning, new score, rating/publication change or2027 outcomes. Positional/lineup forward calibration remains untested for this specific prior shift.

## 2026-10-19 — Component versus Parent VPM scope corrected

- Owner was confused by the experiment labels and clarified the component-product scope. The recent v44–v51 ratings are Parent VPM (one O/D parent plus two SQ residual corrections), not the ten-component Impact Monster. Three broad component-prior calibration targets in v51 do not make its deployed board the component model. Withdraw the parent core+FT recommendation as a component launch decision; valid parent scores stay unchanged.
- Canonical simple reference `VPM_MODEL_MAP.md`: two architectures; expected FT, priors, decay and blends are variants. Local product bundle `vpm-explorer/public/data/vpm-manifest.json` identifies gross ten-component epm12_x2, July24,2026, TD550; this is a checked-out bundle, not a live-deployment verification. Component launch selection remains unresolved. Reader labels/header and review manifest now say Parent VPM research only. Do not equate newest experiment number with best or launch-approved.
