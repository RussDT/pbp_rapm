# EPM Championship Help

scripts/build_championship_help.py owns the current championship surface in
The Help Index. It is a retrospective playoff-context allocation, not a
pregame forecast or causal ring-credit model.

## Data contract

- Coverage begins with the 2001-02 season because the EPM endpoint first
  returns data in November 2001.
- scripts/fetch_monthly_epm_title.py fetches one private, game-agnostic
  Dunks & Threes EPM snapshot per calendar month. Raw provider rows and
  manifests stay under gitignored nba_pipeline/cache/epm_title/.
- Historical trade-stint duplicates collapse only when player identity,
  provider dates, O-EPM, D-EPM, and total EPM are identical. Conflicting EPM
  rows fail loudly. The general E-Skills cache remains strict and unchanged.
- For each season, freeze the latest monthly snapshot requested on or before
  the first postseason game. This prevents later playoff performance from
  updating the rating being allocated.
- Weight that frozen EPM by the actual playoff lineups. Normalize within team,
  so advancing four rounds creates more observed rows but not more modeled
  team strength.

## Probability model

Only playoff teams enter the conditional-softmax choice set. Non-playoff
teams have zero title odds, so probability over all NBA teams still sums to
one each season. Exactly one playoff team is labeled champion.

The model grid compares total EPM with separate offensive/defensive EPM and
selects regularization by the one-standard-error rule on leave-one-season-out
champion log loss. Every player who appeared for the same playoff team gets
the same actual team title probability.

## Neutral playoff rotation

Neutral is not a five-player 0-EPM fiction. For each season it is the average
EPM strength of that season's actual playoff rotations. This common baseline
captures the fact that postseason rotations shorten and the typical playoff
player is better than the all-NBA minute pool.

For a focal player:

- **hard-carry odds** replace the neutral contribution for the focal player's
  actual playoff workload with the player's real contribution;
- **actual-context title lift** replaces the focal player in the actual team
  with a 0-EPM player at the same workload;
- **normalized self** is the player's additive continuous-Shapley allocation
  along the common neutral-rotation-to-actual-roster path;
- **shared-floor teammate lift** uses actual teammate overlap in playoff
  possessions;
- **off-floor lift** is the rest of the team's actual playoff minutes while
  the focal player sat.

The exact identity is:

    neutral playoff-rotation odds
    + normalized self lift
    + shared-floor teammate lift
    + off-floor lift
    = actual team title odds

Individual teammate allocations use the same continuous response multiplier
and sum exactly to shared-floor plus off-floor teammate lift.

## Current validation

The 2002-2026 build contains 25 seasons, 400 playoff team-seasons, 4,894
player-seasons, and 58,485 focal/teammate season rows. The selected model is
overall EPM with C=0.2; leave-one-season-out champion log loss is 1.711152.

Validation maxima:

- annual probability-mass gap: 4.44e-16;
- actual-state reconciliation: 3.33e-16;
- teammate allocation gap: 2.22e-16;
- neutral/self/on-floor/off-floor gap: 1.11e-16;
- team/roster EPM gap: 3.55e-15.

## Rebuild

    set -a
    source .env
    set +a
    PYTHONPATH=. .venv/bin/python \
      nba_pipeline/scripts/fetch_monthly_epm_title.py \
      --start 2001-11-01 --end 2026-07-01

    PYTHONPATH=. .venv/bin/python \
      nba_pipeline/scripts/build_championship_help.py \
      --start-year 2002 --end-year 2026

Only derived CSV/JSON title estimates are published to the app. Never publish
the monthly provider snapshots.
