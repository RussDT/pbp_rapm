# DRL/Shapley Autoresearch Program

This folder is a resumable research harness for the DRL/Shapley value model.

## Scope

You are trying to improve the value model only.

What you can edit:
- `train.py`

What you cannot edit during the research loop:
- `prepare.py`
- `results.tsv`
- cached prepared data

## Setup

1. Run `python prepare.py` once if the cache is missing or stale.
2. Read the latest baselines from `cache/prepared_context.json`.
3. Read `latest_summary.json` if it exists.
4. Use `results.tsv` as the experiment ledger.

## Objective

Primary metric:
- `research_score` from `train.py`
- lower is better

The score is fixed in `prepare.py`:
- 55% normalized margin RMSE vs tree baseline
- 25% normalized Brier vs logistic baseline
- 20% normalized log-loss vs logistic baseline
- small penalties for broken monotonicity / uncertainty behavior

## Ambitious Target

The target is not just "slightly better."

Promote changes only if they push toward:
- `research_score < 0.95`
- `audit_score < 0.98`
- `research_rmse / tree_rmse <= 0.85`
- `research_brier / logistic_brier <= 1.05`
- `research_logloss / logistic_logloss <= 1.10`

Stretch target:
- beat the tree baseline on RMSE by a wide margin
- nearly match the logistic baseline on calibration

## Current Frontier

Start from the actual ledger, not memory.

As of March 9, 2026:
- Best research-score run: `0.930009` research, `1.043975` audit
- Best audit-score kept run: `0.930830` research, `1.034000` audit

What already worked:
- `dropout = 0.0`
- longer training budget with the hardened evaluator
- modestly higher regularization (`weight_decay = 5e-4`)
- slightly stronger regularization (`weight_decay = 6e-4`) to improve audit calibration
- incremental regularization increase (`weight_decay = 7e-4`) to further improve audit calibration

What failed:
- post-training temperature scaling on the research split

Active interpretation:
- the research target is met
- the audit target is not
- the next useful work should be calibration/generalization oriented, not just more capacity

Future expansion:
- after the value harness, build a second harness around the additive decomposition contract in [ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)
- the fixed evaluator for that harness should read `reconciliation_report.json` and the canonical additive tables from the production run

If you beat only the research score by a hair but lose a meaningful amount on audit, do not assume that is a better frontier.

## Loop

1. Edit `train.py`.
2. Run `python train.py > run.log 2>&1`.
3. Read the summary from `latest_summary.json` or `grep '^research_score:\\|^audit_score:\\|^research_rmse:\\|^research_brier:\\|^research_logloss:' run.log`.
4. Append the result to `results.tsv` with `python record_result.py --status keep|discard|crash --description "..."`.
5. Keep only changes that genuinely improve the score without obvious overfitting or unnecessary complexity.

Also check:
- `latest_summary.json`
- whether `evaluation_error` is empty

If a run reports good final metrics but a non-empty `evaluation_error`, log that caveat in the description or discard it if the instability looks material.

## Ground Rules

- Prefer simpler changes over fragile hacks.
- Do not optimize only one metric if it breaks the others.
- Treat `audit_score` as a guardrail, not the main search target.
- Once research is comfortably below target, prioritize moves that narrow the audit gap without giving back much research score.
- If a run crashes, log it and move on unless the fix is trivial.
- If you are stuck, change one major thing at a time: optimizer, normalization, architecture, dropout, target sync, depth/width, residual structure.
- Do not re-run known losers unless you have a concrete new reason:
  - non-zero dropout
  - research-split temperature scaling
