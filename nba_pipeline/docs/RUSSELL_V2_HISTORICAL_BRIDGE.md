# Russell ShotQuality v2 Historical Bridge

## Decision

The research bridge supplies reasonable midrange and three-point expected
make values for 1997-2015 while leaving the frozen native
`RUSSELL_SHOTQUALITY v2` files authoritative for 2016-2026.

The selected historical series is intentionally bucket-specific:

- **midrange:** the calibrated v2-distilled stack;
- **three-point:** the portable historical v3 fallback.

The three-point distillation improved resemblance to native v2 but slightly
worsened held-out make prediction, so it failed the predeclared no-regression
gate. The fallback is explicit in every output row and in the manifest.

This is a retrospective historical bridge, not untouched forward evidence and
not a silent replacement for the accepted 2016-2026 VPM targets.

## Build

```bash
PYTHONPATH=. .venv/bin/python -u \
  nba_pipeline/scripts/build_russell_v2_historical_bridge.py
```

The builder:

1. validates and deduplicates the two frozen v2 parquets;
2. joins their 2016-2026 midrange/three labels to the outcome-safe portable
   historical feature contract;
3. fits a v2 student on 2016-2020, calibrates the v2/portable stack on
   2021-2022, and evaluates it without refitting on 2023-2026;
4. requires both no Brier regression versus portable v3 and improved RMSE to
   native v2 for each shot bucket;
5. refits the selected emission recipe using only 2016-2026 information and
   scores 1997-2015.

No pre-2016 outcome is used to fit the emitted bridge. Current shot outcomes
are excluded from its inputs. Historical safe-crossfit probabilities are
joined only for retrospective audit and are never emitted as VPM targets.

## Frozen held-out gate

The held-out sample is 2023-2026 native v2 overlap.

| Bucket | Shots | Portable Brier | Stack Brier | Portable RMSE to v2 | Stack RMSE to v2 | Selection |
|---|---:|---:|---:|---:|---:|---|
| Midrange | 229,243 | 0.244067 | 0.243943 | 0.072245 | 0.058783 | v2-distilled stack |
| Three | 376,931 | 0.228528 | 0.228738 | 0.075502 | 0.068022 | portable v3 fallback |

For midrange, the game-clustered Brier gain over portable v3 is
`+0.000124`, 95% interval `[+0.000015, +0.000226]`. For threes, the stack
gain is `-0.000211`, interval `[-0.000265, -0.000153]`, so the actual-outcome
gate rejects it despite better v2 fidelity.

## Historical output and audit

The emitted parquet contains 2,583,850 unique midrange/three shots across
23,851 games:

- 1,778,214 midrange rows from the v2-distilled stack;
- 805,636 three-point rows from portable v3;
- zero duplicate shot keys;
- zero missing probabilities;
- exact `ev_pts = p_make * shot_value` closure.

Retrospective outcome checks are supportive but are not model-selection
evidence. The selected midrange bridge improves Brier versus portable v3 in
all three historical eras:

| Era | Bridge mid | Portable mid | Bridge three |
|---|---:|---:|---:|
| 1997-2003 | 0.234881 | 0.237997 | 0.224278 |
| 2004-2013 | 0.234403 | 0.238058 | 0.225929 |
| 2014-2015 | 0.233153 | 0.237443 | 0.226122 |

The three column equals portable v3 by design.

## 2015/2016 seam

The handoff to native v2 is observable and must remain explicit.

| Bucket | 2015 mean EV probability | 2016 native v2 | Probability change | Actual-rate change |
|---|---:|---:|---:|---:|
| Midrange | 0.393957 | 0.418814 | +0.024857 | +0.005920 |
| Three | 0.361786 | 0.354426 | -0.007360 | +0.004126 |

Therefore a pooled 1997-2026 VPM must include fixed season effects, or an
equivalent explicitly audited within-season centering treatment. Do not treat
the raw EV levels as a seamless era-independent scale, and do not post-hoc
calibrate the historical values against the same historical outcomes used to
judge them.

## Artifacts

Generated research artifacts live under:

```text
nba_pipeline/results/historical_shot_quality/russell_v2_historical_bridge/
```

The principal files are:

- `russell_v2_historical_bridge_ev_1997_2015.parquet`;
- `heldout_overlap_metrics.csv`;
- `bridge_selection.json`;
- `historical_outcome_audit.csv`;
- `season_seam_audit.csv`;
- `manifest.json`.

The parquet exposes `zone`, `p_make`, and `ev_pts` for the existing VPM target
swap adapter. Building a full historical no-prior VPM remains a separate
operation: zero prior means must still use the registered ridge penalties,
the same component identities, and season controls.
