# Codex Scratchpad

Short-term session memory. Durable rules belong in AGENTS.md / CLAUDE.md.

## 2026-10-20 — NBA skill live-refresh readiness audited

- Owner asked whether skills are organized to update as season games arrive. Research library yes; operational automatic refresh no. Skills reader fixed Sep1,2026, creation adapter historical cuts/season<=2026, tracking current_2026 exports. Existing daily RAPM shell hardcodes ending2026/old path and calls no estimated-skills refresh. No local crontab or dedicated NBA skills job found among inspected launch agents/Codex automation files. No script executed, new games fetched or schedules modified.
- Need one season-aware ingest→validated observations→frozen-parameter state refresh→dated checkpoint/export/reader runner, all families/rookie identities/missingness/source delays, pregame archives and monitored schedule. Do not confuse nightly filtering with refitting/tuning on2027. Documented readiness gap in VPM_OFFENSIVE_CREATION_TEST.md; no live-ready claim or VPM change.

- Follow-up owner emphasized organization of all skills for future daily2027 flow. Built estimated_skills_operations.json (59 current/28 standalone/1 retained output,16 groups,20 research lanes), inventory reader and manage_estimated_skills.py inventory/freeze/check. Carry-forward bundle copies current2025-vintage recipes, not a new preseason fit.16 meaningful checks pass; actual check blocks16 missing live-adapter receipts. Browser inventory88 rows verified. Source adapters, state replay and schedule remain unwired. No automatic updating/new data or2027 scoring claimed. Canonical ESTIMATED_SKILLS_DAILY.md.

## 2026-10-20 — Creation stats retained independently of VPM admission

- Owner explicitly retains predictive rim/potential assists, assisted/unassisted zone makes and playtype difficulty/points as predictive box-score/advanced stats. Extended existing skills.html/CSV/field dictionary with15 auxiliary outputs/supports, Creation & shot difficulty group, Box score assists and Advanced TSA/difficulty. Player details show Retained/not in VPM with dated provenance; original24 mapper inputs and all priors/ratings preserved (reconstruction gap0).
- Raw missingness preserved; splits close exactly, rim assists match bounded share×AST, approximate TSA=FGA+.44FTA and point conversion labeled.582-row browser CSV matches saved source values; count/method/provenance views verified. Added curated predictive_creation_library_summary. No new fitting, skill-parameter changes, VPM admission or2027 outcomes. Rebuild existing source export plus skills renderer; canonical VPM_OFFENSIVE_CREATION_TEST.md owner-decision section.

## 2026-10-20 — Three offensive creation priors completed

- Immutable sibling v58 separately adds dated rim/potential assists, coherent assisted/unassisted zone makes and playtype points to current forward Parent. Current/assist/split/role stress game13.895289/13.894258/13.907115/13.897795; role wins early development, no later increment. Assist prior-only native gain positive but final unresolved and low-continuity game worse; split game worse, role native/prior worse. Retain current, no2027/promotion.
- 387962 player-games/2913812 shots,55 annual finite-budget OU fits,11 mix fits; source counts/state/prefix/price audits pass.629-source/24-mapper/24-board/84-score/63-pair independent model audit and6 tests pass; eight controls reproduce. Supplemental helper pins match pre-existing frozen sources.582/1556 ratings, priors and inputs at8767/offensive-creation.html; search/sort/download verified. Owner clarified role should be points: already2*predictedTSA*difficulty/100. TSA uses conventional predictedFGA+.44*predictedFTA, not exact FT-trip counts; explicitly disclosed without post-result refit. Canonical VPM_OFFENSIVE_CREATION_TEST.md; curated sibling evidence.

## 2026-10-20 — Offensive creation feature gaps inventoried

- Owner asked predictive rim assists and playtype/assistedness inputs. Exact current13-field O mapper has none; v4 skill registry and monthly player-game panel lack rim-assist/assistedness filters/counts. Observed season rim-assists exist; lineup RIM_ASSIST is not passer-credited own skill. Separate potential-assist next_ou forecast exists (dev2020–22/stress2023–25/2026 export), unused by current VPM. Need proper historical parameter vintages before admission.
- Current SQ already has description/role cues but no player-level difficulty prior. Prior PBP role study improves creation-controlled future-oTS1.4940→1.4744; not a current-VPM gate. Candidate blocks: predictive rim/potential assists, coherent assisted/unassisted zone makes replacing redundant totals, dated prior-priced mix difficulty. Made-only assist tags cannot classify missed-shot creation. Added source/interpretation note to VPM_REPAIRED_FORWARD_EXPANDED.md; no new model fit, source change, reader change or2027 access.

## 2026-10-19 — Displayed VPM mean checked without changing ratings

- Owner clarified average displayed ratings, not expected SQ gain. Current forward585-event reader simple mean−1.1323 (582 positive-exposure−1.1399);2025–26 RS-minute weighted mean−.0754, RS+PS+.00457. Full-universe O/D own TD550-exposure means0. Only178/585 positive;77 players above2000 RS minutes average+1.233. Explain exposure reference and selected high-minute players, not a broad positive offset or forced future zero. Added read-only note to VPM_REPAIRED_FORWARD_EXPANDED.md; no model/reader/forecast/source/2027 changes.
