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

## Title Lift over Contract

The primary player-value measure is **Title Lift over Contract**, implemented as
a realized salary-peer roster swap rather than a hand-set Contract EPM hurdle.
For every focal player-season with salary coverage:

1. take every complete historical playoff bracket outside the focal season;
2. on each of its 16 teams, identify the playoff rotation player with the
   closest salary-cap share, breaking ties toward the larger playoff workload;
3. remove that peer's actual EPM contribution and insert the focal player at
   the peer's actual playoff workload;
4. propagate the original and swapped teams through that season's actual
   four-round bracket; and
5. average `swapped title odds - original roster title odds` across placements.

There are 351 or 352 placements for a covered row because the focal season is
excluded and a small number of historical teams lack any salary-covered playoff
rotation player. This design keeps the purchased roster slot, workload, actual
teammates, opponent distribution, and bracket together. A star does not receive
an easier field merely because his real powerhouse team was removed.

The public **raw title lift** uses the identical salary-matched placements and
workloads but compares the focal player's EPM with a 0-EPM insertion. Therefore
the gap between raw lift and Title Lift over Contract is the title value already
supplied by the removed salary peer. Actual-context title lift remains a separate
question on the focal player's real team.

Bracket probabilities are evaluated exactly, not estimated with Monte Carlo
noise. A fitted playoff game logit prices team EPM difference plus home court;
each matchup is converted to an exact first-to-four, best-of-seven probability
under the `2-2-1-1-1` schedule, averaging which team owns home-court advantage.
Those series probabilities are then propagated through the observed 8-4-2-1
bracket tree. The one-standard-error leave-one-season-out game-log-loss rule
selects `C=0.05`, home advantage `0.4783` log odds, and `0.1342` log odds per
team-EPM point on 2,085 playoff games. Mean season log loss is `0.6177`.

The cost input is annual salary divided by the official NBA salary cap. It is
called **salary share**, never cap hit. The private crosswalk currently supplies
exact player-team-season salary rows through 2024-25. Missing team matches and
2025-26 stay missing; they are never zero-filled or guessed from another team's
row. The current player-season coverage is 4,378 of 4,894 rows (89.46%). Career
totals sum only covered seasons and publish the covered-season count beside the
total.

The earlier Contract EPM v1 curve is retained only as a prototype sensitivity.
It reuses declared market assumptions from the From Impact to Winning reader:

- replacement EPM: `-2.0` (assumption);
- replacement salary share: `$2.1M / $154.647M = 1.358%`;
- market price: `$8.5M / $154.647M = 5.497 cap points per WAR`;
- normalized workload: `60 possessions/game * 82 games = 4,920 possessions`;
- wins per point: `0.0297496`, so +1 EPM at normalized workload is `1.4637 WAR`.

Thus:

    Contract WAR = (salary share - 1.358%) / 5.497%
    Contract EPM = -2.0 + Contract WAR / 1.4637

A 35% salary share implies about `+2.18 Contract EPM`. This does not define the
headline measure and is not shown as the player's salary-peer hurdle. It remains
a transparent sensitivity pending a signing-date market model. Data/licensing
and no-leakage requirements are documented in
[Title Lift over Contract: data and method](./TITLE_LIFT_OVER_CONTRACT_DATA.md).

## Reader contract

The public player page leads with Title Lift over Contract and shows actual EPM,
average realized salary-peer EPM, salary share, and bracket-placement count for
a covered season. The exact team equation remains underneath as neutral rotation
+ player share of team lift + teammate help = actual team odds. It is team
accounting and is not the salary-peer counterfactual.

The season-spike panel defaults to salary-peer title lift and toggles among that,
actual-context lift, raw title lift, and teammate help. The league comparison is
one sortable table with salary-peer title lift, context title lift, raw title
lift, teammate title lift, and team title odds visible together. Compact
two-line headers keep each counterfactual name above the quantity.

## Current validation

The 2002-2026 build contains 25 seasons, 400 playoff team-seasons, 4,894
player-seasons, and 58,485 focal/teammate season rows. The selected model is
overall EPM with C=0.2; leave-one-season-out champion log loss is 1.711152.
The bracket layer reconstructs exactly 375 series—15 in every season—with an
8-4-2-1 round pattern. LeBron James in 2008-09 has 351 salary-peer placements,
average peer EPM `+2.55`, swapped title odds `21.09%`, original peer-roster odds
`6.27%`, and Title Lift over Contract `+14.82` percentage points. The median
placement's absolute salary-share gap across covered player-seasons is about
`0.60` cap points; the median 90th-percentile gap is `1.84` cap points.

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
