# Monthly predictive VPM history

October15 follow-up: the [new SQ temporal repair and paired rerun](VPM_SQ_TEMPORAL_REPAIR.md)
is complete in sibling v39/v40. The original artifacts described here remain
preserved; their identified leakage warning is not retroactively erased.

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


Completed September 29, 2026 for the owner's monthly history request: **93 first-of-month estimates, January 2019 through September 2026**, including offseason and pandemic months. This is the own-skill O/D parent plus midrange/three-point SQ residual research series with dated entrant priors. Original v18 remains frozen.

## Deliverables

All outputs are in sibling `vpm/vpm/results/harness/vpm_monthly_2019_v1/` (relative to the common research directory):

- `monthly-vpm.html`: standalone date selector, searchable/sortable board, player history, entrant-prior details and CSV export.
- `monthly_ratings.parquet` / `monthly_ratings.csv`: **110,458 player-month rows, 1,556 distinct players**; O/D/net, raw ratings, prior components, exact prior NBA exposure, source labels and coverage.
- `monthly_validation.csv`, `monthly_manifest.json`: all-cut validation, source/config/code hashes and artifact hashes.
- `monthly/YYYY-MM-DD/`: per-cut ratings, fitted own-prior maps and manifest containing reference offsets, raw/centered intercepts, continuation prices and equation audits.
- `entrant_models/`: eight chronological background vintages (2018–2025), identity rejection audit and hashed source manifest.
- `parent_forward_labels/`: ten 365-day label cuts, October 2015–2024 except December 2020.
- `reader_runtime_validation.json`: JavaScript behavior checks; this is not a visual browser check.

The default reader shows NBA players observed within the preceding 365 days: 610 at the first cut and 585 at September 2026. Its full fit universe grows from 794 to 1,556. This is not a transaction-aware roster reconstruction or a certified next-season rookie roster. The 585 recent players differ from the older 582-player board because coverage/date rules differ.

## Model contract and changes from v18

Immutable configuration: sibling `vpm/configs/research_harness_v34_monthly_own_sq.json`, experiment `vpm_monthly_2019_v1`. Use the `monthly-vpm` command; inherited component/evaluation sections are not a claim that this stage ran those experiments.

The parent retains O/D penalties 4000/8000, own-prior scale 1.5, mapper ridge 0.1 and 550-day half-life. Midrange and three residuals each use 12000/48000, with zero residual priors. The exact solve is actual parent plus strongly regularized residual minus the same residual at parent penalties. Equal-penalty algebra is checked at every cut. Decay is anchored to the last observed training game, as in v18.

This series changes more than rookie priors: upstream skill/correction parameters and prior maps are fitted chronologically, targets use actual FT outcomes, continuation pricing uses only earlier games, and displays use an explicit weighted reference. Therefore differences versus original v18 must not be attributed solely to entrants.

### Dated own skills and labels

The isolated 2014–2018 warm-up contributes 150,670 rows. Combined source `results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet` contains 387,962 unique player/game rows, October 29, 2013–June 13, 2026. Original modern sources are unchanged. Source reconciliation receipts retain imperfect team-score matches rather than claiming universal exactness.

`build_monthly_own_skills.py` fits 17 core skill models at each annual 2015–2025 vintage, before October 1 (December 1 in 2020), with past-only context/age/experience corrections. The accelerated OU kernel is algebraically checked against the original filter. `build_monthly_defensive_skills.py` adds 11 annual defensive model packages. DFGA uses measured matching opportunities. Early unprefixed 2014–2016 rim files are cumulative: recover only validated consecutive PBP appearances, exclude gaps/post-trade ambiguity, and never difference percentages. The 2017–2018 prefixed daily files and modern repaired-v3 counts retain their separate lineage.

`assemble_monthly_own_skills.py` produces `results/estimated_skills/vpm_monthly_neutral_v1/`: 198 model artifacts, 21 observed features, 127,351 player-cut rows and 2,627,535 provenance rows over 97 dates (93 requested months plus four early mapper-training dates). Makes/100 are exact attempt-rate times probability products. Latest neutral states are held; states older than 730 days become missing. Missing features use coherent mapper imputation. No next-game context is passed as player talent.

Own O/D maps use only fully matured 365-day actual-point parent labels with label penalties 2000/4000. January 2019 has three matured early feature/label pairs. Observations and fit dates are fenced at each estimate; architecture and optimizer/search settings remain retrospective choices.

### Entrants and established players

The background model combines audited draft metadata, dated college/G League counts and a profiled population fallback fitted against actual NBA possession outcomes. Missing draft pick remains unknown, never automatically undrafted. Identity joins retain the v33 strict exact-match/audited-crosswalk contract.

Background weight is `max(1 - prior_career_NBA_possessions / 2000, 0)`. Prior NBA exposure includes exact partial-season game counts. At or above 2,000 possessions, combined prior equals the existing own prior exactly. This differs deliberately from the broad v33 blend that retained background weight for veterans. The fixed 2,000-possession transition is a research design choice, not newly optimized predictive evidence.

All 93 cuts have **zero change to established own-prior means**. Fitted veteran ratings can still move because shared lineups connect their estimates to entrants. September 2026 Alex Antetokounmpo is −3.976 displayed VPM, with 45 prior NBA possessions and 97.75% background weight from G League evidence; there is no player-specific rating override.

### Actual FT and pre-cut continuation targets

Canonical expected FT uses full-season shooter conversion, so blindly filtering its saved possession rows is not a date-safe monthly target. `build_monthly_vpm_targets.py` writes separate actual FG+FT atoms for all 22 RS/PS pairs from 2016–2026, with exact game/event/lineup alignment and actual point closure. The archived walk-forward SQ gross residual is attached with its source hashes; its upstream shooter context includes dated predictive skill inputs. This does not imply all SQ inputs are own-estimated skills.

`build_monthly_vpm_kappa.py` captures raw per-game first-chance misses and subsequent points before terminal-row/lineup filtering. That population matters: estimating continuation from retained terminal rows alone omits rare incomplete possessions and fails canonical reconstruction. The optional raw capture leaves default clean-chance behavior unchanged.

Each monthly cut recomputes season/phase continuation value using games strictly before the cut. Net residual equals gross residual times `1 - kappa / shot_value`. Raw-prefix rebuild audits for January 2019 and January 2026 match actual points exactly; maximum residual differences are below 4.26e-10. Future-appended observations cannot change earlier pricing.

### Reference means and predictions

A common decline changes the mean; it is not automatically evidence that every player's relative talent declined. On the older same-582 comparison, the saved-exposure-weighted mean changed from +0.631993 to +0.157697 (−0.474296), while the equal-player mean changed from −0.30345 to −1.361105. Jokić's raw −0.534879 change was −0.060583 after aligning those means. These are older-board diagnostics, not monthly-series validation or a rookie-only causal estimate.

Monthly offense and defense each subtract their own full-fit-universe TD550 possession-weighted mean. The matching intercept adjustment preserves lineup PPP. Raw O/D/net, offsets and intercepts are retained; never recenter priors themselves or use a filtered current-player subset as the reference. A player with zero exposure on one side receives zero reference weight on that side. A filtered or unweighted displayed subset need not average zero.

## Validation and limitations

Assembly verifies all source/config hashes, every requested month, unique finite player-month ratings, O+D closure, prior/target observation dates, matured labels, established-prior identity, weighted reference means and exact solve equations. Maximum all-cut errors:

| Check | Maximum |
|---|---:|
| Established prior change | 0 |
| Reference prediction change | 2.78e-16 |
| Possession target closure | 4.45e-16 |
| Equal-penalty coefficient closure | 4.02e-15 |
| Relative normal-equation error | 6.33e-15 |

All 24 focused tests pass. They cover original-filter agreement, chronological prefix invariance, rim recurrence, raw continuation populations, exact monthly exposure, veteran-prior identity and prediction-preserving centering. Reader runtime checks cover dates, search, all-player toggle, four sorts, history, absent-date behavior, SVG coordinates, CSV export and empty results. Chrome visual verification was unavailable because remote debugging was not enabled; no visual pass is claimed.

This completes the requested research history, not predictive promotion. Source publication/biographical vintages are not certified, model design was chosen retrospectively, and individual uncertainty is not calibrated. Draft source coverage ends in 2025; add audited new draft/roster identities before live admission. No 2027 outcomes were used; untouched confirmation remains required.

## Reproduction

Run from RAPM with its `.venv/bin/python`. Source preparation runners are `build_monthly_own_skills.py` and `build_monthly_defensive_skills.py` (each `--stage prepare`, `train`, `export`), then `assemble_monthly_own_skills.py`. Targets use `build_monthly_vpm_targets.py` and `build_monthly_vpm_kappa.py`. Model/source locks fail if their frozen inputs change; use a new version for changed methodology.

Run the following harness command sequentially with stages `labels`, `entrants`, `fit`, then `assemble` (the separate `audit` stage inventories source readiness):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python ../vpm/vpm/run_harness.py \
  --config ../vpm/vpm/configs/research_harness_v34_monthly_own_sq.json \
  monthly-vpm --stage assemble
```

The fit stage resumes saved cuts; assembly verifies their source hashes before emitting the combined reader. Generated data/model/HTML artifacts stay ignored under the backup policy; preserve scripts, configuration, tests and this receipt.
