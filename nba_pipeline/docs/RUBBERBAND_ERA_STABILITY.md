# Rubberband Era Stability

## Answer

The rubberband mechanism persists back to 1997, but its exact curve is not
constant across eras.

`rapm.py --rubberband` estimates one common margin-by-clock surface over all
seasons loaded into a single solve. Therefore a single 1997-2026 run currently
imposes one pooled curve across that entire span. Separate rolling runs estimate
separate surfaces automatically.

That pooled-curve assumption is useful as an average nuisance correction, but
it is not the best time-local specification. Use rolling five- or six-season
windows when the output is intended to represent a particular season or era.

## Historical score-state repair

Older processed files store cumulative score only on scoreboard-update rows and
use `0-0` placeholders elsewhere. The old transform interpreted those
placeholders as real score resets. That made pre-2018 margin and venue labels
invalid even though offense/defense player assignment remained correct.

The transform now carries the last observed score pair forward within each game,
seeds home/away player identity only from unambiguous score changes, and
propagates that identity through lineup continuity. Across audited 1997-2026
seasons it resolves venue orientation on approximately `99.3%-99.8%` of
possessions, with home-offense shares near `49.7%-49.9%`. Unresolved rows receive
a neutral score margin for context modeling.

## Forward test

The 1997-2001 smooth alpha-100 surface was frozen. For every test season from
2002 through 2026, player coefficients were refit on the preceding five seasons
under three otherwise identical specifications:

1. No rubberband adjustment.
2. The fixed 1997-2001 surface.
3. A jointly re-estimated trailing-five-season surface.

All three models predicted the same next-season possessions. Comparisons are
paired and clustered by game.

| Test era | Rolling vs none | Fixed 1997-2001 vs none | Rolling vs fixed |
| --- | ---: | ---: | ---: |
| 2002-2009 | +0.001305 | +0.001150 | +0.000155 |
| 2010-2017 | +0.001586 | +0.001296 | +0.000291 |
| 2018-2026 | +0.002158 | +0.001187 | +0.000971 |
| 2002-2026 | +0.001713 | +0.001210 | +0.000503 |

Every interval clears zero. For the full 2002-2026 sample, rolling versus fixed
is `+0.000503 [0.000474, 0.000530]` possession MSE. The gain from localization
is largest in the modern era.

The 2021-2025 surface used to predict 2026 remains structurally similar to the
1997-2001 surface (`r=0.942` on a dense state grid), but differs by `1.71` points
per 100 RMS. This is gradual drift, not a completely different phenomenon.

## Operating rule

- Freeze the smooth basis and `alpha_rb=100`; do not freeze one coefficient
  vector for all history.
- Rolling and daily outputs should re-estimate the rubberband surface inside
  their own training window.
- A long 1997-2026 lifetime solve may retain one pooled surface as a nuisance
  average, but must not call it era-specific. Use rolling historical outputs for
  season-local interpretation.
- Do not add hard era buckets without a separate forward gate. A temporally
  smoothed long-span surface is a future candidate, not current production.

The reproducible harness is
`nba_pipeline/scripts/research_rubberband_era_stability.py`. Generated fold,
surface, bootstrap, and report artifacts live under
`nba_pipeline/results/rubberband_era_stability/` and remain research artifacts.

