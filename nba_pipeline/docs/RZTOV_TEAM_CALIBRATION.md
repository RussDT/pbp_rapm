# Team workload calibration check

> Source correction, 2026-09-05: the initial NBA results below mixed a partial-season reader TSA rate with complete-season PBP exposure. They are superseded by the same-sample RS rebuild documented in `RZTOV_SAME_SAMPLE_CORRECTION.md`. Do not quote the original NBA numerical receipts as current.

## Frozen check design — 2026-09-05

This is the held-out-workload follow-up to `RZTOV_TEAM_RECONCILIATION.md`.
Run `nba_pipeline/scripts/check_rztov_team_calibration.py`; results live in the
gitignored `nba_pipeline/results/rztov_team_calibration/` directory. The check
writes diagnostics, not revised player ratings or production team adjustments.

NBA primary: outer chronological seasons 2018-2026. For every outer test year,
all feature-family and alpha selection repeats the existing expanding-season
validation on earlier seasons only. Then refit those earlier seasons and predict
the outer test year's assisted share and potential assists. 2018 is the first
outer fold with enough earlier years for the inner chronological selection.

WNBA primary: 2025, trained on 2023-2024, with alpha chosen on 2023-to-2024.
Secondary: nested leave-one-season-out over 2023-2025; alpha selection sees only
the other two seasons. Secondary folds can use later years and are not forward
forecasts. Three observed-label years do not support a strong stability claim.

Both leagues hold canonical TSA and incumbent turnover prices fixed while
comparing predicted and observed role inputs. Price fitting itself is not held
out: turnover-unit errors are a diagnostic conversion, not an end-to-end rZTOV
validation. The existing model-feature TSA definitions stay frozen.

Allocate each season-level player quantity to teams by local game-resolved
OffPoss shares. Preserve an unknown-team bucket, check that team allocations sum
to the original player error, and retain traded-player splits. These are not
separately fitted team-stint role estimates. Output only covers players with
observed labels, usable features, and a ledger row; fold receipts list scored
and retained counts.

Export both raw errors and common-sample season-centered errors. Removing the
test season's mean is strictly a diagnostic of between-team drift, not a
correction learned for deployment. Report errors as counts and percentages of
the same common-sample incumbent allowance. Do not force any team's allowance
to equal its actual turnovers.

All historical seasons here are already spent development evidence. Nested
fitting excludes outer labels from fitting and selection within this run; it
does not undo earlier methodological selection. NBA 2027 remains untouched;
WNBA needs fresh observed role labels. There is no invented pass threshold or
automatic promotion from this diagnostic.

## Completed result

All 13 outer folds completed with 4,359 player-fold rows. Every player's
workload-error count survives the team/unknown-team allocation exactly within
floating-point tolerance; no scored ledger player lacks an allocation. Three
focused tests pass, covering chronological exclusion, LOSO exclusion, and the
fixed-TSA allowance-error identity.

The team differences below are after removing the common-sample season mean:

| Design | Team-seasons | Median absolute shift | 90th percentile absolute shift | Signed range |
| --- | ---: | ---: | ---: | ---: |
| NBA chronological 2018-2026 | 270 | 1.38% | 3.52% | -7.01% to +5.97% |
| WNBA chronological 2025 | 13 | 1.28% | 3.09% | -3.88% to +3.17% |
| WNBA nested LOSO 2023-2025 | 37 | 1.32% | 2.98% | -3.88% to +4.66% |

NBA 2026 alone ranges from -42.65 to +43.46 allowance counts, or -4.31% to
+4.79%. WNBA chronological 2025 ranges from -20.97 to +18.17 counts. Positive
means predicted workloads grant more allowance than observed workloads under
the same incumbent prices.

The uncentered NBA common-sample workload overallowance is positive in every
outer season, ranging from 0.59% to 2.23% of the incumbent allowance. WNBA's
chronological 2025 overallowance is 1.00%; LOSO ranges from -2.37% to +1.00%.
These exclude the small unknown-team bucket from the displayed ratios; the
underlying allocation checks include it. Keep these raw differences visible:
season centering removes a league shift but cannot establish calibration.

NBA consecutive-season team error correlation is 0.516, descriptively. This
suggests errors need not cancel randomly within squads, but is not a validated
team correction: repeated players, roster continuity, common model structure,
and shared data can contribute to persistence.

## Decision

The check confirms a modest typical team effect with materially larger tails.
Do not mark the estimated workloads as team-calibrated, and do not force their
allowances to equal actual team turnovers. A correction remains unselected.
The next candidate needs team-stint workload inputs or a defensible aggregate
workload constraint, training-only calibration, and fresh-label/future-target
confirmation. Current historical diagnostics cannot admit a production change.

Artifacts:

- `team_folds.csv`: raw and centered team error counts, workload quantities,
  comparison allowances, known/unknown team attribution, and source counts.
- `player_folds.csv`: held-out predictions, observed role labels, common TSA,
  incumbent prices/allowances, and exact player error accounting.
- `fold_receipts.json`: training years, selected families/alphas, and coverage.
- `summary.csv` and `manifest.json`: result and explicit validation limitations.
