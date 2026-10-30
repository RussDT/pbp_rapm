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
For every focal player-season with salary coverage and a frozen pre-playoff EPM
rating—whether or not the player's own team reached the playoffs:

1. take every complete historical league season outside the focal season;
2. on each of its 16 teams, identify the playoff rotation player with the
   closest salary-cap share, breaking ties toward the larger playoff workload;
3. insert the focal player at the peer's actual regular-season workload, then
   simulate qualification and seed from the full historical schedule and all
   teams in the destination conference;
4. remove the peer's playoff EPM contribution, insert the focal player at the
   peer's actual playoff workload, and propagate each simulated seed through a
   standard four-round bracket with higher-seed home court; and
5. average `swapped title odds - original roster title odds` across placements.

There are 368 placements for every covered row: 16 teams in each of the other
23 postseasons. This design keeps the purchased roster slot, both realized
workloads, historical league environment, opponent distribution, and bracket
together. A star now receives credit for improving qualification, seed, and
home court instead of inheriting the removed peer's realized playoff seed.

Actual-team odds, context lift, teammate help, and the additive team equation
remain playoff-only because they require the player's real postseason rotation.
Salary-peer and raw lift do not require that context: they use the destination
peer's workload. The public export marks `is_playoff_participant` and leaves
playoff-only fields null for nonplayoff seasons rather than turning missing
context into zero.

The public **raw title lift** uses the identical salary-matched placements and
workloads but compares the focal player's EPM with a 0-EPM insertion. Therefore
the gap between raw lift and Title Lift over Contract is the title value already
supplied by the removed salary peer. Actual-context title lift remains a separate
question on the focal player's real team.

### Era-local, budget-neutral challenger

The build now retains **Title Lift over Contract v2** beside the existing
full-history result. It uses only realized playoff teams from the four seasons
before and four seasons after the focal year, excluding the focal year. A
middle-season row therefore has at most `8 * 16 = 128` placements; edge years
use the complete neighboring seasons that are available and report the smaller
placement count.

The closest-salary peer rule is unchanged, but a non-exact salary match is no
longer treated as budget neutral. Let `gap_pp = 100 * (focal salary share -
peer salary share)`. The v2 team-rating changes are:

```text
regular-season change = (focal EPM - peer EPM) * peer RS workload
                        - 0.067 * gap_pp

playoff change = (focal EPM - peer EPM) * peer playoff workload
                 - 0.078 * gap_pp
```

The prices are robust within-season slopes estimated on actual playoff rotation
players earning 8%-20% of the cap. They are in team-rating points per one
percentage point of cap, not EPM-rate points. A more expensive focal player is
debited and a cheaper player receives the symmetric credit. The subtraction is
made before the nonlinear qualification, seed, series, and bracket calculation.
The public reader shows v2 and full-history v1 simultaneously in the player
header and league table. A shareable selector controls which model drives the
season spike and detailed actual-versus-baseline explanation; neither result is
discarded or hidden.

### Entering-playoffs EPM-prior challenger

The reader also exposes a research **Playoff prior** model. It starts from each
player's frozen pre-postseason O-EPM and D-EPM, then adds a persistent playoff
translation learned only from that player's earlier postseasons. The O/D
translation is a zero-centered ridge solve with `alpha=24000`, using the
previous eight postseasons; a player without earlier playoff possessions gets
exactly zero adjustment. This is an entering-playoffs prior, not a fit using the
postseason being valued.

Regular-season qualification and seeding remain unchanged on pre-playoff EPM.
Inside the playoff bracket, adjusted impact is used consistently for the focal
player, the salary peer, every destination teammate, and every opponent field.
The model therefore asks whether persistent playoff translation changes value
over the contract slot; it does not give the focal star a one-sided postseason
bonus while leaving peers and opponents at regular-season ratings.

On the 2007-2025 held-out playoff gate, the adjusted ratings improve possession
MSE by `31.53e-6` versus pre-playoff EPM across 1,586 games and 294,451
possessions. The game-clustered 95% interval is `[-84.90e-6, +23.97e-6]`, so it
crosses zero. The point estimate is encouraging but not a validated promotion.
The public reader therefore labels it a research challenger and retains v2 as
the default. LeBron's career total moves from `1.057` under v2 to `1.206` with
the playoff prior, rather than receiving an assumed fixed postseason boost.

#### Dynamic prior-informed residual prototype

`build_dynamic_playoff_epm_ratings` tests the single-season prior-informed RAPM
analogy directly. It fits one positive-good net playoff residual after each
completed postseason, centers the next postseason's residual ridge on that
posterior, and mean-reverts the carried state annually. The entering rating is
always emitted before the current postseason update, so the valued postseason
cannot leak into its own prior. The net update is split evenly across the O-EPM
and D-EPM communication fields; only their exact sum is interpreted.

An exploratory grid used 2007-2015 as development and did not inspect player
rankings for selection. The best point row was `alpha=4000` with annual
persistence `0.25`, improving possession MSE by `12.56e-6`; the exact zero
adjustment remained within one standard error and is therefore the formal
one-standard-error selection. With the point-best row frozen, the 2016-2025
report window improved MSE by `40.16e-6` across 834 games and 159,340
possessions, with game-clustered 95% interval `[-89.29e-6, +11.23e-6]`, and
improved 7 of 10 individual postseasons.

The existing eight-year model improved the same report window by `64.10e-6`
and 5 of 10 postseasons. Dynamic minus existing MSE was `+23.94e-6`
`[-46.23e-6, +95.88e-6]`; over the full 2007-2025 walk-forward period it was
`+4.03e-6 [-43.05e-6, +51.96e-6]`. The models are empirically tied. The
dynamic version is retained as a reproducible research prototype, not added to
the public reader and not promoted into official Title Lift over Contract.

Regular-season qualification and seed use 2,048 reproducible common-random
draws from each schedule-implied Poisson-binomial win distribution, using its
exact mean and variance. A fitted regular-season game logit uses 30,064 games,
selects `C=0.05`, home advantage `0.4067` log odds, and `0.1433` log odds per
team-EPM point. The simulated focal team is ranked against every team in its
conference; missing the top eight produces zero title probability.

Conditional on a seed, bracket probabilities are evaluated exactly rather than
sampled. A fitted playoff game logit prices team EPM difference plus home court;
each matchup is converted to an exact first-to-four, best-of-seven probability
under the `2-2-1-1-1` schedule. The higher seed owns home court; tied seeds in
the Finals are averaged. Those series probabilities are propagated through the
standard 1-8/4-5 and 2-7/3-6 bracket. The one-standard-error leave-one-season-out game-log-loss rule
selects `C=0.05`, home advantage `0.4777` log odds, and `0.1320` log odds per
team-EPM point on 2,085 playoff games. Mean season log loss is `0.6165`.

The seed simulation is intentionally tractable rather than a 30-team roster
replay: it approximates schedule win totals from their mean and variance, ranks
without historical division/play-in tiebreak rules, and retains the seven
realized non-focal playoff teams while moving the destination team among seeds.
The destination can miss the top eight and receive zero title odds, but a new
nonplayoff opponent roster is not inserted into the field.

The cost input is annual salary divided by the official NBA salary cap. It is
called **salary share**, never cap hit. The private crosswalk currently supplies
salary rows through 2024-25. Team-stint salary amounts are summed to one annual
player-season salary share; the largest salary stint supplies only the display
team when no playoff context exists. Missing salaries and 2025-26 stay missing;
they are never zero-filled. The current focal universe is 11,613 player-seasons:
5,110 playoff-context rows plus 6,503 nonplayoff rows. Salary-peer lift is covered
for 11,241 rows (96.8%). Career totals sum every covered season and separately
publish the number with actual playoff context.

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

The research-only [scalar Title Lift over Contract challenger](./CONTRACT_TITLE_CURVE_CHALLENGER.md)
tests whether the realized placement result can be compressed to expected
contract EPM plus one title curve. The literal EPM-only version is inadequate;
adding the expected workload attached to the salary slot raises held-out-season
correlation with the full model from `0.902` to `0.987`. This is a
compressibility diagnostic trained against the placement output, not
independent validation and not a reader-facing replacement.

## Reader contract

The public player page leads with Title Lift over Contract and shows actual EPM,
average realized salary-peer EPM, salary share, playoff-only lift, seeding plus
qualification lift, expected seed, playoff probability, and placement count for
a covered season. Full-season lift closes exactly as `playoff-only lift + seeding
and qualification lift`. The exact team equation remains underneath as neutral rotation
+ player share of team lift + teammate help = actual team odds. It is team
accounting and is not the salary-peer counterfactual.

The season-spike panel defaults to salary-peer title lift and toggles among that,
actual-context lift, raw title lift, and teammate help. The league comparison is
one sortable table with salary-peer title lift, context title lift, raw title
lift, teammate title lift, and team title odds visible together. Compact
two-line headers keep each counterfactual name above the quantity.

The player page also ranks non-overlapping five-year Title Lift over Contract
peaks for the selected contract model. Only complete runs of five consecutive
seasons with finite model values qualify. The reader selects the highest summed
five-season lift first, then greedily selects the highest remaining window that
shares no season with an already selected window, continuing while another
complete window exists. Values are expected titles added over contract across
the five seasons; they are not percentage points or a new simulation.

## Current validation

The 2002-2026 build contains 25 seasons, 400 playoff team-seasons, 5,110
playoff player-seasons, 11,613 salary-peer focal rows, and 61,227 focal/teammate
season rows. The selected team-accounting model is overall EPM with C=0.2;
leave-one-season-out champion log loss is 1.726082. The 2020 bubble cutoff is
explicitly labeled season 2020; this repairs its former zero EPM coverage to
100%.
The bracket layer reconstructs exactly 375 series—15 in every season—with an
8-4-2-1 round pattern. LeBron James in 2008-09 has 368 salary-peer placements,
average peer EPM `+2.49`, full-season Title Lift over Contract `+15.08` percentage
points, split into `+13.63` points of fixed-bracket playoff lift and `+1.45`
points from qualification/seeding. Across careers, LeBron rises from `1.28` to
`1.42` titles over salary peers and Stephen Curry from `1.11` to `1.23`; their
seeding/qualification components are `0.14` and `0.12` titles, respectively. The median
placement's absolute salary-share gap across covered player-seasons is about
`0.57` cap points; the median 90th-percentile gap is `1.64` cap points.

The v2 rebuild uses 64-128 era-local placements per covered row (median 128).
LeBron's career result is `1.057` titles and Curry's is `0.672`, widening their
gap from `0.182` under the full-history metric to `0.385`. LeBron's 2008-09
season is `+17.53` percentage points, split into `+15.59` fixed-bracket points
and `+1.93` qualification/seeding points. Its average signed salary gap is only
`+0.62` cap points, producing average regular-season and playoff rating debits
of `0.042` and `0.049`. The old and v2 career totals correlate `0.960` across
2,275 common careers. These are sensitivity results, not a validation against
realized championships.

Validation maxima:

- annual probability-mass gap: 4.44e-16;
- actual-state reconciliation: 2.22e-16;
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
