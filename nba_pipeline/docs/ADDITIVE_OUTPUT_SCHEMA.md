# Additive Output Schema

This document defines the additive contracts for the DRL/Shapley production outputs in
[train_drl_shapley.py](/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/scripts/train_drl_shapley.py).

## Why This Exists

The value model, attributor, and event allocator produce multiple useful views of player impact.
Those views are only trustworthy if each layer has a clear reconciliation identity.

This schema separates the model into two additive layers:

1. State-value layer
2. Event-value layer

These layers are related, but they are not the same total and should not be forced into one identity.

## Contract 1: State Value Attribution

For each output state `s`:

`sum_i phi_i(s) = V(s)`

Where:
- `V(s)` is the model's expected remaining home margin from the current state
- `phi_i(s)` is player `i`'s Shapley-style attribution for that state

Saved outputs:
- `state_values.parquet`
- `state_player_phi.parquet`

Key columns:
- `state_values.parquet`
  - `value_current`
  - `value_next`
  - `phi_sum_current`
  - `phi_sum_next`
  - `state_efficiency_gap`
  - `next_state_efficiency_gap`
- `state_player_phi.parquet`
  - `phi_current`
  - `phi_next`
  - `phi_delta`

Interpretation:
- `player_ratings.csv` is derived from this layer
- these are context-state ratings, not event-credit totals

## Contract 2: Event Credit Allocation

For each event transition `e`:

`delta_V(e) = reward(e) + gamma(e) * V(next) - V(current)`

and:

`sum_i credit_i(e) = delta_V(e)`

Where:
- `credit_i(e)` is the player-level event allocation
- both raw and stabilized versions are saved

Saved outputs:
- `event_player_credit.parquet`
- `state_values.parquet`

Key columns:
- `event_player_credit.parquet`
  - `credit_raw`
  - `credit_stabilized`
  - `role`
  - `phase_side`
- `state_values.parquet`
  - `delta_v_raw`
  - `delta_v_stabilized`
  - `event_credit_raw_sum`
  - `event_credit_stabilized_sum`
  - `event_conservation_gap_raw`
  - `event_conservation_gap_stabilized`

## Contract 3: Player Event Decomposition

For each player-season:

`event_total_i = scoring_i + playmaking_i + offensive_rebounding_i + defensive_actions_i + turnovers_i + defensive_presence_i`

This is the core additive player decomposition.

Saved outputs:
- `player_bucket_totals.csv`
- `player_value_decomposition.csv`
- `player_totals.csv`

Key points:
- `player_value_decomposition.csv` is the canonical wide table
- `defensive_presence` is the residual bucket that closes the identity exactly
- `bucket_gap_stabilized` should be approximately zero

## Non-Additive Diagnostic Outputs

These are useful, but not part of the core additive contract:
- `pair_synergy.csv`
- `team_synergy.csv`

Synergy is a separate together-vs-apart diagnostic and should not be mixed into the core player total in v1.

## Reconciliation Report

The run writes `reconciliation_report.json` with:
- state attribution efficiency
- next-state attribution efficiency
- event conservation
- player bucket reconciliation

This file is the first thing any future autoresearch harness should check.

## Practical Reading Guide

Use these files in this order:

1. `reconciliation_report.json`
2. `player_totals.csv`
3. `player_value_decomposition.csv`
4. `player_ratings.csv`
5. `pair_synergy.csv`

This avoids mixing the event-credit decomposition with the state-rating leaderboard.
