# ShotQuality closest-defender features as a dTS prior

Status: **failed its promotion gate.** The one spec that beat the incumbent on the
forward fold did not replicate on split-half selection. Nothing here is promotable.
Retained as a reproducible negative result plus a data-quality finding.

## What the export actually contains

Source of record is now `nba/research/shotquality/store/shots_{season}.parquet`,
built by `scripts/sync_shotquality.py`. The old surface — 2,740 ragged per-game
CSVs with 28 distinct column counts — is superseded; see "Why the store replaced
the CSVs" below.

Scope correction against the original premise of "24-26, three years":

| SQ season | repo file | FGA extracted | defender | release geometry | closing velocity |
|---|---|---|---|---|---|
| 2025 (2024-25) | `NBA25` | 246,977 | 86.2% | 86.5% | **61.0%** |
| 2026 (2025-26) | `NBA26` | 241,165 | 46.1% | 51.6% | **0.0%** |
| 2024 (2023-24) | `NBA24` | — | — | — | not on this key |

So it is **two seasons, not three**, and only one of them is well covered.
2025-26 coverage degrades through the season (0.76 of shots in Oct 2025 down to
0.19 in May 2026), and its motion fields are absent entirely. That is tolerable
for a prior, which is built from the earlier season by construction.

**This is a provider gap, confirmed by re-pull, not a stale-fetch artifact.** The
old `sync_sq_nba_games.py` skipped any game already on disk, which made a
backfill-lag explanation plausible, and the provider's `pbp_updated_at` had moved
on 2,738 of 2,742 games. But re-pulling those games returns byte-identical
coverage (0.0% stays 0.0%, 78.4% stays 78.4%, same row counts). Games without
tracking carry only shooter priors and shot distance; the geometry does not exist
upstream and will not arrive. `pbp_updated_at` therefore is **not** a usable
staleness signal for geometry content.

**Two seasons is the hard ceiling on this API key.** Season ids 3225-3230
(2018-19 through 2023-24) all return HTTP 403; only 4857 (2024-25) and 149560
(2025-26) return 200. A third season requires a key with wider entitlement.

### Why the store replaced the CSVs

**The old store was silently dropping features.** Re-pulling with
`feature_store=true` returns a uniform 117-column response. The ragged CSVs
carried the motion columns in only 951 of 2,740 files, so a scan across the union
measured closing velocity at ~2% populated and it was dropped without ever being
tested. In the parquet store it is 61.0% of 2024-25 FGA. Unioning a ragged schema
on every read is both slow and quietly lossy: a column absent from a file becomes
NaN, which is indistinguishable from a genuinely missing value and corrupts any
coverage measurement taken from it. The re-sync also recovered 35,848 additional
2024-25 FGA (211,129 → 246,977).

`defender_id` and `feature_store.Closest Defender ID` are identical wherever both
appear (4,526/4,526 in sample); both are ShotQuality-internal player ids.

### Provider data quality: corrupt motion values

About **0.9% of motion values are physically impossible** — `d1_speed` peaks at
53,629 ft/s, `d1_v_shooter` reaches −16,742,002 ft/s, and season-level
`rim_closing_v` bottomed at −2,268 ft/s. An NBA player tops out near 25 ft/s.
Percentiles are sane through p99 (25–34 ft/s) and explode only past p99.9, so
this is a tracking glitch on a small tail, not a real distribution.

`build_sq_defender_features.py` sets out-of-envelope values to missing (±30 ft/s
for velocity, ±40 ft/s² for acceleration) rather than clipping, which would pile
garbage onto the boundary and still drag the mean. 13,368 values dropped across
both seasons. Without this guard the effort arm's RMSE is **2.4342** instead of
1.2331 — a two-fold error driven entirely by ~1% of rows. Any future consumer of
these fields needs the same guard.

## Identity bridging

`nba_id_to_player_id_mapping.csv` maps 563 SQ player ids to NBA ids directly;
a name fallback through `players.csv` adds 3,341 more. Defender mapping given a
non-null defender is 99.6% (2025) and 89.4% (2026). Unmapped players are G-League
fringe who never reach the shots-defended minimum.

**Defender-season features key on `(nba_id, season)` and therefore do not need the
SQ->NBA game id bridge**, which is missing for all 1,355 season-2026 games. Only
re-enriching raw PBP needs that bridge; that is a separate track.

## Build

```bash
python nba/research/shotquality/scripts/sync_shotquality.py --refresh stale
python nba_pipeline/scripts/build_sq_defender_features.py all --min-shots 100
python nba_pipeline/scripts/test_sq_defender_dts_prior.py          # confirm fold

# split-half gate
python nba_pipeline/scripts/solve_ts_split_half.py 25
python nba_pipeline/scripts/build_sq_defender_features.py features \
    --min-shots 60 --half h1 --cut-date 2025-01-25 \
    --out sq_defender_h1_features.csv
python nba_pipeline/scripts/test_sq_defender_split_half.py         # select fold
```

Produces `results/sq_defender_shots_{25,26}.parquet` (488,142 FGA) and
`results/sq_defender_season_features.csv` (775 defender-seasons at >=100 shots).
All are generated artifacts and stay out of git.

Features split into three arms that are always reported separately:

- **geometry** — pre-release position only: contest distance, cosine angle,
  behind/inside/blocking flags, bounding-box height difference (contest
  verticality), defenders closer to the basket, paint/restricted-area occupancy,
  shot diet shares.
- **effort** — pre-release motion: closing velocity and acceleration toward the
  shooter, velocity toward the basket, defender and shooter speed. Recovered by
  the store rebuild; 2024-25 only.
- **outcome** — lagged realized value: points allowed, provider EV allowed, and
  `suppression` = EV allowed minus points allowed, split rim/mid/three.
  Legitimate for a prior (it is prior-season), but it is an outcome, not geometry.

## Forward test

One honest fold exists: 2024-25 features -> 2025-26 single-season dTS, n=296
players with >=1,000 defensive possessions in both seasons and >=150 shots
defended. Generalization estimated by 5-fold CV across players inside that fixed
forward fold; possession-weighted. Incumbent prior is the player's own prior-season
dTS.

| spec | RMSE | R² | vs incumbent (player-clustered bootstrap) |
|---|---|---|---|
| league mean | 1.2713 | −0.006 | |
| **prior dTS (incumbent)** | 1.2071 | +0.093 | |
| prior dTS + geometry (20 feat) | 1.2438 | +0.037 | +0.0370 [−0.0042, +0.0774] neutral |
| prior dTS + compact geometry (5) | 1.2126 | +0.085 | +0.0055 [−0.0145, +0.0256] neutral |
| prior dTS + **effort** (9 feat) | 1.2331 | +0.053 | +0.0260 [+0.0006, +0.0513] **worse** |
| prior dTS + geometry + effort | 1.2819 | −0.023 | +0.0750 [+0.0154, +0.1357] **worse** |
| prior dTS + outcome (8 feat) | 1.1749 | +0.141 | −0.0320 [−0.0717, +0.0094] neutral |
| prior dTS + `rim_suppression` | 1.1708 | +0.147 | −0.0359 [−0.0743, +0.0011] neutral |
| **prior dTS + compact outcome (3 feat)** | **1.1593** | **+0.163** | **−0.0474 [−0.0929, −0.0009] improves** |

Strongest univariate correlates of next-season dTS: `pts_allowed` −0.356,
`rim_suppression` +0.330, `dts_prev` +0.329, `rim_ev_allowed` −0.294,
`ev_suppression` +0.280.

## Findings

1. **Tracking geometry does not predict future dTS.** The marquee contest features
   are flat univariately (`mean_bbox_diff` −0.038, `pct_blocking` −0.003,
   `mean_contest` −0.040), and the full geometry arm never beats the incumbent.
   This replicates the repo's existing negative results for defender identity on
   three-point SQ and for opposing-lineup D-EPM.
2. **Defensive effort does not predict it either — this is now a real test, not a
   missing-data artifact.** Closing velocity was the single most promising
   recovered feature family and it is inert: `mean_closing_v` +0.032,
   `rim_closing_v` −0.028, `mean_d1_speed` −0.010, `mean_closing_a` +0.014, all
   within noise of zero. The arm is significantly *worse* than the incumbent, and
   adding it to geometry compounds the damage. Whatever separates good rim
   protectors, it is not how fast they close.
3. **Value allowed relative to the provider's own pre-shot expectation does
   predict it**, concentrated at the rim. `rim_suppression` alone correlates with
   next-season dTS about as strongly as prior-season dTS does.
4. **The incumbent is weak.** Prior dTS vs league mean is itself neutral
   (−0.0628 [−0.1294, +0.0040]). Single-season dTS is noisy enough that both sides
   of this comparison are underpowered; part of why the compact outcome spec wins
   is likely that it is a better-shrunk estimate of the same construct measured on
   hundreds of shots rather than a noisy possession-level ridge.

The shape of the result is consistent across all three arms: **what a defender
*does* before the shot carries no forward signal in this data; what he *allows*
relative to expectation does.** Position and motion are both pre-release
measurements and both fail; the only thing that survives is a lagged outcome.

## Why this is not yet promotable

- The compact specs (`GEOM_C`, `OUT_C`) were chosen **after** reading the
  univariate correlation table on this same fold. That is selection and confirmation
  on one sample. The 3-feature win is the best case among specs tried, not a
  pre-registered result — and it is the *only* spec whose interval excludes zero,
  by a margin (upper bound −0.0009) that would not survive a multiplicity
  correction over the eleven specs tested.
- n=296, one fold, one season pair. No untouched confirmation season exists.
- `suppression` embeds the provider's EV model, so it inherits that model's
  biases and cannot be interpreted as causal defensive credit.
- Applying this forward means building the prior from 2025-26, which has only
  ~46% defender coverage that is correlated with calendar time, and no motion
  fields at all. A spec fit on well-covered 2024-25 will not transfer cleanly.

## Step 1 of the gate: split-half replication — **FAILED**

2024-25 was split on its median possession date (2025-01-25) into two halves of
131,987 and 126,839 possessions. `solve_ts_split_half.py` solves dTS separately on
each half by reusing rapm.py's own load / transform / design-matrix / solver
functions, so the half-season targets carry identical conventions to the
full-season incumbent. Verified: run on the full season it reproduces
`ts_25_all_results.csv` exactly, max|diff| = 0.0000 across all 569 players.

    SELECT   H1 2024-25 features -> H2 2024-25 dTS   n=304  (min 500 poss / 75 shots)
    CONFIRM     2024-25 features ->    2025-26 dTS   n=296  (min 1000 poss / 150 shots)

| spec | select ΔRMSE | confirm ΔRMSE | replicates? |
|---|---|---|---|
| geometry (20) | +0.0211 neutral | +0.0370 neutral | no signal either way |
| **effort (9)** | **+0.0347 worse** | **+0.0260 worse** | **yes — consistently harmful** |
| outcome (8) | −0.0354 neutral | −0.0320 neutral | directional, never significant |
| compact geometry (5) | −0.0060 neutral | +0.0055 neutral | no |
| `rim_suppression` (1) | −0.0114 neutral | −0.0359 neutral | no |
| **compact outcome (3)** | **−0.0097 neutral** | **−0.0474 improves** | **no** |

**The compact outcome spec does not replicate.** Its confirm-fold win shrinks to a
fifth of its size on the selection fold and its interval covers zero comfortably
([−0.0339, +0.0153]). No spec improves significantly on both folds.

The leading outcome feature also changes between folds: `ev_allowed` (−0.351) and
`shot_taking_allowed` (−0.333) lead on the selection fold, while `pts_allowed`
(−0.356) and `rim_suppression` (+0.330) lead on the confirm fold. Which member of
the outcome family looks best is unstable, which is what picking one on a single
fold would be expected to produce if the family carries a weak common signal and
the ranking within it is noise.

What survives replication is the **negative** result: the effort arm is
significantly worse than the incumbent on both folds, and geometry helps on
neither. The outcome arm is directionally helpful on both folds and significant on
neither — a real but weak effect this sample cannot resolve, not a promotable one.

## Remaining gate before any promotion

1. ~~Split-half replication inside 2024-25.~~ **Run; failed.** The compact spec was
   a selection artifact.
2. If pursued further, test the **outcome arm as a family** rather than a
   hand-picked subset — it is the only arm with consistent directional support —
   and pre-register the spec before touching a new season.
3. Confirm on untouched 2026-27 when it exists, or on a third SQ season if a key
   with wider entitlement becomes available.
4. Only then: test as a component prior inside the VPM harness against the
   no-prior and box-prior baselines on the identical possession universe, per the
   cross-lane bootstrap rules.

**Do not feed `rim_suppression`, or any spec in this document, into VPM or the
public reader.** Step 1 was the promotion gate and it did not clear.
