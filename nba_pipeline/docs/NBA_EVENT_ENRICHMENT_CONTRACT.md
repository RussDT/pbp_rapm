# NBA Event-Enrichment Contract

The canonical cross-repository design is
[NBA_EVENT_ENRICHMENT_PLAN.md](/Users/russ/Documents/Codex/basketball/shared/NBA_EVENT_ENRICHMENT_PLAN.md).
`nba_pipeline` owns the eventual versioned event layer; RAPM and ShotQuality
research consume provider-specific views rather than maintaining private,
lossy joins.

## Defender providers are separate

NBA matchup assignments and ShotQuality closest-defender data measure related
but different provider concepts:

- `nba_matchup` is the matchup-derived defender source. Audited
  `matchup_fga` shot-assignment coverage is season-end 2017-2025, meaning
  2016-17 through 2024-25. Its supporting game/player index reaches earlier,
  but those rows are not equivalent to shot-level defender assignments.
- `shotquality` is the licensed closest-defender/geometry source. The recovered
  rich store is 2024-25 only. It has a true primary-defender ID on about 76% of
  FGA.

Never collapse these into one generic `defender_id`. The canonical long form is
one row per `event_key`, provider, and rank, retaining assignment method,
confidence, provider play ID, and source version. A future consensus field may
be derived, but provider disagreement must remain observable.

NBA matchup `DEF_ID` can contain multiple pipe-separated assigned defenders on
one shot. Normalize those to separate provider link rows with an assignment
count/single-assignment flag; do not force the first ID into a generic closest
defender field.

ShotQuality has no native second-defender ID. Its second-defender geometry is
useful, but the repo's biometric identity reconstruction is inferred and can
equal the primary defender. Keep the geometry, label any reconstructed identity
as inferred, and do not present it as a provider-supplied secondary assignment.

## Matching contract

`(game_id, action_number)` is not inherently a unique event key. NBA raw files
can contain auxiliary rows sharing an action number, and the existing SQ/NBA
research merge produced 186 excess assignments on already-used keys.

Therefore:

- exact duplicate provider rows may collapse only when every payload field
  agrees;
- distinct provider candidates remain explicitly ambiguous;
- duplicate raw event keys remain explicitly ambiguous;
- ambiguous keys do not receive defender identity or other SQ payload;
- provider joins must never use `drop_duplicates(..., keep="first")`;
- promoted enrichment requires a stable `event_key`, one-to-one crosswalk,
  match method, confidence, and coverage/disagreement reports.

The transitional `scripts/01b_enrich_pbp_shotquality.py` now follows this rule.
It writes `sq_primary_defender_id`, `sq_provider`, `sq_match_status`,
`sq_match_candidate_count`, and `sq_source_row_count`. It removes the legacy
`sq_defender_id` field on rerun and leaves ambiguous payloads null.

## Defensive decomposition

Closest-defender identity is not causal proof. The research target is a
centered on-ball/off-ball hierarchy on one common shot sample:

```text
role = is_primary - player_season_primary_share
on  = parent + (1 - primary_share) * contrast
off = parent - primary_share * contrast
```

The published additive contributions are:

```text
on_ball_contribution  = primary_share * on
off_ball_contribution = (1 - primary_share) * off
parent = on_ball_contribution + off_ball_contribution
```

The conditional `on` and `off` rates are diagnostics and do not themselves sum
to the parent. Fit the interaction with grouped-by-game validation, strong role
shrinkage, and provider-sensitivity checks.

Provider comparison should retain separate NBA-only, SQ-only,
agreement/high-confidence, and SQ geometry variants. Switching, help, scheme,
and scramble defense remain confounded.

The first strict NBA-only implementation is
`scripts/run_primary_defender_net_decomp.py`. On 753,830 2021-2025
regular-season first-chance FGA rows, leave-one-season-out validation selects
role alpha 3,000 and improves MSE in all five folds by 0.0162% on average versus
role-off. The aggregate role block generalizes, but early/late player contrast
repeatability is modest (`r=0.306`, rank `0.280`, 59.2% sign agreement). Keep
the player surface diagnostic-only pending a hierarchical role parent and
separate provider sensitivity. See
[Primary-Defender Net-DECOMP Split](./PRIMARY_DEFENDER_NET_DECOMP.md).

## Research ownership

- RAPM owns defender-role models, additive attribution, state/event value, and
  validation.
- ShotQuality research owns the provider challenger and feature ablations.
- VPM may consume only stabilized player-season outputs after multi-season
  reliability and incremental-target testing. It must not ingest raw provider
  joins, defender geometry, crosswalk diagnostics, or fold predictions.

The repo-owned five-defender lineup residual in
`build_pbp_shot_quality.py` is not a provider assignment. Its internal
`defender_id` labels refer to ordinary on-court lineup members and must not be
joined or described as NBA/SQ primary defenders.
