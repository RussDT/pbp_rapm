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

For the actual-team additive accounting path:

- **actual-field hard-carry odds** replace the neutral contribution for the focal player's
  actual playoff workload with the player's real contribution. The focal
  player's actual team is removed from the softmax field and replaced by this
  counterfactual target team. This is retained as an internal contextual
  diagnostic, not the comparable hard-carry leaderboard, because removing the
  actual team gives stars on stronger teams an easier opponent field;
- **actual-context title lift** replaces the focal player in the actual team
  with a 0-EPM player at the same workload;
- **player share of team lift** is the player's additive continuous-Shapley
  allocation along the common neutral-rotation-to-actual-roster path. The
  stored field remains `normalized_self_title_probability_added` for schema
  compatibility; it is accounting, not the primary player-value metric;
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

## Comparable hard-carry environment

The reader-facing hard-carry lift removes both actual-team help and
player-specific opponent-field help. Every player-season is placed into one
fixed neutral 16-team environment:

- neutral title odds are exactly `1/16 = 6.25%`;
- the opponent field is 15 equal-strength neutral teams;
- the target receives only the focal player's normalized title-score increment
  from the selected EPM title regression.

If `s_i` is that player increment, comparable hard-carry odds are
`logistic(s_i - log(15))` and hard-carry lift subtracts `1/16`. The same
environment is used in every season, so a player is not rewarded because his
actual powerhouse team disappears from the opponent set. A second stored
diagnostic uses one common field within each season—15 times the mean softmax
mass of all 16 playoff teams—but the era-neutral version is the public ranking
measure.

Neither neutral-field hard-carry measure is an additive bucket in the actual
team equation. The neutral/self/teammate closure above remains the source of
truth for actual team odds.

## Title Lift over Contract

The primary player-value measure is now **Title Lift over Contract**. It holds
the neutral 16-team field and the player's actual playoff workload fixed, then
changes only the player's impact rate:

    p_actual   = neutral-field title probability(actual EPM, same workload)
    p_contract = neutral-field title probability(Contract EPM, same workload)

    Title Lift over Contract = p_actual - p_contract

Both probabilities use the one full-sample overall-EPM title-score coefficient
and 15 equal neutral opponents. A 0-EPM player has `1/16` title odds before
workload scaling. This removes the player's actual teammates, actual team, and
season opponent field from the contract comparison. Meeting the Contract EPM
hurdle produces exactly zero; negative values remain negative.

The cost input is annual salary divided by the official NBA salary cap. It is
called **salary share**, never cap hit. The private crosswalk currently supplies
exact player-team-season salary rows through 2024-25. Missing team matches and
2025-26 stay missing; they are never zero-filled or guessed from another team's
row. The current player-season coverage is 4,378 of 4,894 rows (89.46%). Career
totals sum only covered seasons and publish the covered-season count beside the
total.

Contract EPM v1 reuses the declared prototype market assumptions from the
From Impact to Winning reader, expressed as fixed cap shares for cross-era use:

- replacement EPM: `-2.0` (assumption);
- replacement salary share: `$2.1M / $154.647M = 1.358%`;
- market price: `$8.5M / $154.647M = 5.497 cap points per WAR`;
- normalized workload: `60 possessions/game * 82 games = 4,920 possessions`;
- wins per point: `0.0297496`, so +1 EPM at normalized workload is `1.4637 WAR`.

Thus:

    Contract WAR = (salary share - 1.358%) / 5.497%
    Contract EPM = -2.0 + Contract WAR / 1.4637

A 35% salary share implies about `+2.18 Contract EPM`. These are transparent
prototype parameters, not a fitted unrestricted-free-agent price curve. The
production data/licensing and no-leakage market-fit requirements are documented
in [Title Lift over Contract: data and method](./TITLE_LIFT_OVER_CONTRACT_DATA.md).

## Reader contract

The public player page leads with Title Lift over Contract and shows actual EPM,
Contract EPM, and salary share for a covered season. The exact team equation is
retained underneath as neutral rotation + player share of team lift + teammate
help = actual team odds. This makes its accounting role explicit.

The season-spike panel defaults to Title Lift over Contract and toggles among
over-contract lift, actual-context lift, hard-carry lift, and teammate help.
The league comparison is one sortable table with over-contract title lift,
context title lift, hard-carry title lift, teammate title lift, and team title
odds visible together. Compact two-line headers keep each counterfactual name
above the quantity. The older shared-floor DARKO explorer and methodology stay
collapsed so they do not compete with the title-lift answer.

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
the monthly provider snapshots or the raw private salary history. Pass a
different private salary bridge with `--salary-path`; it must use ending-year
season, NBA player ID, exact team abbreviation, annual salary, and official cap.
