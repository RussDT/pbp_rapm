# DRL/Shapley Autoresearch Harness

This directory packages the DRL/Shapley value model into an `autoresearch`-style loop.

## Files

- `prepare.py`: fixed data prep, splits, baselines, score function
- `train.py`: the single editable research surface
- `record_result.py`: append the latest run summary to `results.tsv`
- `program.md`: instructions for autonomous iteration
- `results.tsv`: experiment ledger

## Goal

Improve the value model for the public impact metric stack while keeping the evaluation fixed.

The current ambitious target is:
- `research_score < 0.95`
- `audit_score < 0.98`

## Current Frontier

The search has now cleared the research target but not the audit target.

Current active candidate in [train.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/train.py):
- `learning_rate = 7e-4`
- `weight_decay = 7e-4`
- `dropout = 0.0`
- `max_epochs = 24`
- `patience = 3`

Most recent kept run on a 60 second budget:
- `research_score = 0.930830`
- `audit_score = 1.034000`
- `research_rmse = 8.672895`
- `research_brier = 0.180703`
- `research_logloss = 0.540165`
- `audit_brier = 0.113605`
- `audit_logloss = 0.378338`

Best raw research run so far:
- `research_score = 0.930009`
- `audit_score = 1.043975`
- description: `60s higher weight decay 5e-4; improved research and audit with stable evaluation`

Established findings:
- Removing dropout from the value model was a real win.
- Allowing the hardened trainer to run longer than the original 12 epoch frontier was a real win.
- Slightly stronger regularization (`weight_decay = 6e-4`) further improved audit calibration.
- Increasing regularization to `weight_decay = 7e-4` produced another small audit improvement with stable evaluation.
- Post-training temperature scaling on the research split was a loss and should not be retried casually.

Current bottleneck:
- Research is below target.
- Audit calibration is still the limiting gap.

## Next Frontier

Once the value model is strong enough, the next harness should optimize the additive decomposition layer, not just the value head.

Reference files:
- [ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)
- [reconciliation_report.json](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/drl_shapley/26/reconciliation_report.json)

The eventual decomposition harness should score:
- state attribution efficiency
- event conservation
- player bucket reconciliation
- held-out stability of player bucket outputs

## Quick Start

```bash
cd /Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley
python prepare.py
python train.py
python record_result.py --status keep --description "baseline"
```

For a short smoke run:

```bash
DRL_AR_MAX_SECONDS=20 python train.py
```

## Notes

- The harness optimizes the value model only, not the full attributor/allocator/synergy stack.
- The prepared dataset is fixed and cached so future sessions can resume from the same research state.
- `latest_summary.json` stores the most recent run summary for quick restart.
- `results.tsv` is the source of truth for what actually improved.
- If `latest_summary.json` contains a non-empty `evaluation_error`, treat the run as partially unstable even if final metrics were produced.
