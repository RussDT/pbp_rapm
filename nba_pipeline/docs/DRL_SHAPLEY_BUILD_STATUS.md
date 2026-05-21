# DRL/Shapley Build Status

This document is the durable source of truth for the DRL/Shapley model we are building in this repo.

It covers:
- what the Sloan paper is trying to do
- what we have implemented locally
- what the additive decomposition means
- what the autoresearch harness is optimizing
- what still differs from the paper

## Why We Are Building This

The target is not "replace RAPM with a neural net."

The target is a DRL/Shapley layer that can do the parts additive RAPM misses:
- context-sensitive play value
- off-ball / distributed defensive credit
- interaction-sensitive decomposition
- pair synergy as a separate diagnostic

The intended long-run metric stack for this repo is still:
- six-factor RAPM as the stable mechanical core
- DRL/Shapley as the context / off-ball / interaction layer

See [BEST_PUBLIC_IMPACT_METRIC_2026.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/BEST_PUBLIC_IMPACT_METRIC_2026.md) for the full public-metric direction.

## What The Paper Attempts

At a high level, the Sloan DRL/Shapley paper attempts to do six things:

1. Learn a value function over game states from play-by-play.
2. Define event value as:
   `delta_V = reward + gamma * V(next) - V(current)`
3. Attribute state value to players with Shapley-style methods.
4. Attribute event value to players with a hybrid event + Shapley decomposition.
5. Aggregate those credits into player metrics and action values.
6. Measure pair synergy separately from the core additive player number.

Important structural point:

The paper naturally creates multiple related outputs, but they are not all the same total.
State-value attribution and event-credit attribution are related layers, not a single scalar identity.

## Current Local Implementation

Main production trainer:
- [train_drl_shapley.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/train_drl_shapley.py)

Current implementation status:
- event-transition dataset from raw PBP: implemented
- numeric state encoder + lineup encoder: implemented
- distributional value model over final margin: implemented
- Monte Carlo Shapley target generation: implemented
- neural attributor with exact efficiency correction: implemented
- hybrid offensive / defensive event credit allocation: implemented
- pair synergy output: implemented
- additive decomposition contract and reconciliation outputs: implemented
- dedicated decomposition autoresearch harness: not yet built

Current autoresearch harness:
- [README.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/README.md)
- [program.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/program.md)
- [train.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/train.py)
- [results.tsv](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/results.tsv)

## Additive Decomposition Contract

The current production output is organized into two additive layers.

### Layer 1: State attribution

For each state `s`:

`sum_i phi_i(s) = V(s)`

This is the state-value attribution layer.

Output files:
- `state_values.parquet`
- `state_player_phi.parquet`
- `player_ratings.csv`

This layer is about context-state value, not event-credit totals.

### Layer 2: Event attribution

For each event `e`:

`delta_V(e) = reward(e) + gamma(e) * V(next) - V(current)`

and:

`sum_i credit_i(e) = delta_V(e)`

This is the event-credit layer.

Output files:
- `event_player_credit.parquet`
- `player_bucket_totals.csv`
- `player_value_decomposition.csv`
- `player_totals.csv`

For each player-season:

`event_total_i = scoring_i + playmaking_i + offensive_rebounding_i + defensive_actions_i + turnovers_i + defensive_presence_i`

`defensive_presence` is the residual bucket that closes the event-credit identity exactly.

Full schema:
- [ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)

## Current Production Outputs

The production trainer now writes these important DRL/Shapley outputs:
- `value_model.pt`
- `attributor.pt`
- `player_ratings.csv`
- `player_ratings_min_possessions.csv`
- `action_values_by_class.csv`
- `action_values_context_grid.csv`
- `player_value_decomposition.csv`
- `pair_synergy.csv`
- `team_synergy.csv`
- `state_values.parquet`
- `state_player_phi.parquet`
- `event_player_credit.parquet`
- `player_bucket_totals.csv`
- `player_totals.csv`
- `reconciliation_report.json`

Current example output directory:
- [results/drl_shapley/26](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/drl_shapley/26)

## Reconciliation Status

The additive contract is not just documented; it is now checked in outputs.

Current smoke-run reconciliation:
- [reconciliation_report.json](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/drl_shapley/26/reconciliation_report.json)
- `all_contracts_pass = true`
- state efficiency max abs gap: `4.47e-07`
- raw event conservation max abs gap: `4.56e-07`
- stabilized event conservation max abs gap: `1.18e-07`
- player bucket reconciliation max abs gap: `0.0`

## Autoresearch Status

The current autoresearch harness only optimizes the value-model layer.

It does not yet optimize:
- attributor quality
- event-credit quality
- bucket reconciliation quality
- synergy quality

Current best known harness frontier from [results.tsv](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/results.tsv):
- best research-score run: `research_score 0.930009`, `audit_score 1.043975`
- best audit-kept run: `research_score 0.930830`, `audit_score 1.034000`

Current active research direction:
- `dropout = 0.0`
- `learning_rate = 7e-4`
- `weight_decay = 7e-4`
- `max_epochs = 24`
- `patience = 3`

The autoresearch automation is now configured to run in the real workspace, not a worktree:
- [automation.toml](/Users/russellthomas/.codex/automations/drl-iterate/automation.toml)

Important limitation:
- the automation is only improving `nba_pipeline/autoresearch_drl_shapley`
- it is not automatically modifying the broader classical RAPM pipeline

## What Still Differs From The Paper

We are closer now, but this is not yet a literal paper replica.

Remaining gaps include:
- the current value harness is still the main optimization loop; we do not yet have an autoresearch harness for decomposition quality
- the production output exists, but we have not yet run and frozen a fully trusted multi-year production frontier with the new additive outputs
- synergy exists as a diagnostic, but it is not yet part of any stabilized public metric design
- the paper-style decomposition is implemented structurally, but the research loop is not yet directly scoring the additive event decomposition

## What We Have Learned So Far

Autoresearch findings that should not be rediscovered:
- removing dropout was a real win
- longer training than the original short baseline was a real win
- mild additional regularization improved audit calibration
- post-training temperature scaling on the research split was a loss
- probability stability needs to be handled at the shared value path, not only in a harness-specific fallback

## What To Do Next

The next important step is not another large freeform modeling rewrite.

The next step is to build a second autoresearch harness for the additive decomposition layer.

That harness should score:
- state attribution efficiency
- event conservation
- player bucket reconciliation
- held-out stability of player bucket outputs

Inputs for that future harness:
- [ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)
- [reconciliation_report.json](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/results/drl_shapley/26/reconciliation_report.json)
- `state_values.parquet`
- `state_player_phi.parquet`
- `event_player_credit.parquet`
- `player_bucket_totals.csv`
- `player_totals.csv`

## Read This First

If you are a future agent touching this system, read these in order:

1. [DRL_SHAPLEY_BUILD_STATUS.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/DRL_SHAPLEY_BUILD_STATUS.md)
2. [ADDITIVE_OUTPUT_SCHEMA.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/docs/ADDITIVE_OUTPUT_SCHEMA.md)
3. [README.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/README.md)
4. [program.md](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/program.md)
5. [results.tsv](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/autoresearch_drl_shapley/results.tsv)
