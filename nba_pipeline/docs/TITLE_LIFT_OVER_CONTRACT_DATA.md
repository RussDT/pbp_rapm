# Title Lift over Contract: data and method

## Decision

There is no complete, official, publicly licensed NBA player **cap-hit** history for 2001-02 through 2025-26. The public metric should therefore have two possible releases:

1. **Preferred production:** license a cap-accounting vendor and call the input `cap hit`.
2. **Reproducible research:** use annual player salary divided by the official NBA salary cap and call it **salary share**, never cap hit.

The first research build should use the second option. It answers, “How much title lift did the player produce above the performance normally purchased by this share of the cap?” It does not attempt to grade every exception, incentive, dead-money charge, or general-manager decision.

## Season convention

The Title Lift app uses the season's **ending year**. Thus `season = 2002` means 2001-02 and `season = 2026` means 2025-26.

BALLDONTLIE (BDL) uses the **starting year**, so:

```text
title_season_end = bdl_season_start + 1
```

Every stored row should carry both `season_start` and `season_end`; all reader joins should use `season_end`.

## Player-cost sources

### Current Help Index research extract

The deployed Help Index does not call a live salary API. Its current bundled
contract input is Gabriel's `player_sheets` regular-season extract,
`data/player_sheets/modern_selected_deduped.csv`, pinned to repository commit
`cae13ad` and downloaded 2026-07-09. The reader uses `SALARY_PCT_CAP` when
present and otherwise computes `salary / SALARY_CAP`; postseason records inherit
the regular-season contract columns on `PLAYER_ID + year`. The pinned file has
native 2026 coverage (537 of 582 player rows with salary share) and a
2025-26 cap denominator of `$154,647,000`.

This is still annual salary share, not audited CBA cap hit. The reader credits
Steph Noh's NBA Salary Model as the public model behind the contract-value
fields. Keep the pinned source/date in release metadata and never silently
replace missing salary with zero.

### Modern primary source: BALLDONTLIE contracts

The official [BDL API documentation](https://docs.balldontlie.io/) exposes annual player-contract rows with `player_id`, `season`, `team_id`, `cap_hit`, `total_cash`, and `base_salary`, plus contract-level type, status, guarantees, free-agency, and signing fields. These endpoints require the GOAT tier.

Retrieve every player in the Title Lift universe, rather than treating the team endpoint as a complete payroll ledger:

```text
GET https://api.balldontlie.io/v1/contracts/players
    ?player_id={bdl_player_id}
    &seasons[]={season_start}
    &per_page=100
    &cursor={next_cursor}

GET https://api.balldontlie.io/v1/contracts/players/aggregate
    ?player_id={bdl_player_id}
```

The local Databallr integration already maintains `balldontlie_player_mapping` (`bdl_player_id` to `nba_id`) and pagination logic in `backend/app/services/balldontlie.py`. Use the current official `/v1/` routes above; the local wrapper's older `/nba/v1/` routes are compatibility code, not the contract to build against.

A 30-team audit of the Databallr BDL middleware found no rows for 2009 or 2010 start years, 404 rows across only 28 teams for 2011, and generally about 450 rows thereafter. Many team-season responses contain exactly 15 players. Therefore BDL is a useful modern player-level source and cap-hit sensitivity, but its team endpoint is not a complete historical cap ledger and does not close 2002-2011.

BDL's current [terms](https://www.balldontlie.io/terms.html) allow lawful copying, caching, publishing, and derivative databases, while prohibiting credential sharing, limit circumvention, competing services, and resale of the raw unmodified data. They also warn that data may be stale, duplicated, incomplete, or wrong. Store retrieval date and a response hash, never publish the API key, and publish derived rows rather than a raw provider dump.

### Historical bridge: frozen CC0 salary snapshot

For 2002-2011, the most reproducible public bridge found is the Kaggle dataset [NBA Players Salaries Details 1985-18](https://www.kaggle.com/datasets/iampunitkmryh/nba-players-details-198518), designated CC0 by its uploader. It contains `salaries_1985to2018.csv` with Basketball-Reference slugs, nominal salary, start/end season, and team. The 2022-07-26 snapshot contains 459, 459, 458, 478, 494, 511, 486, 471, 472, and 467 rows for ending years 2002-2011, respectively.

Exact retrieval:

```text
kaggle datasets download \
  -d iampunitkmryh/nba-players-details-198518 \
  --unzip
```

Pin the Kaggle version/date and SHA-256 of `salaries_1985to2018.csv`. This is a **salary** source, not a CBA cap-hit source. The uploader's CC0 designation also does not prove that every upstream source granted database-republication rights, so retain the snapshot privately and publish only the derived salary share and metric unless legal review approves raw redistribution.

The repository already has a year-specific Basketball-Reference-slug to NBA-ID pattern in `nba_pipeline/scripts/build_bref_ft_pct_lookup.py`, backed by Gabriel `site_Data/index_master.csv`. Reuse that exact join, reject conflicting `season + bref_id` mappings, and maintain an explicit alias table for unresolved rows. Do not fuzzy-match silently.

### Audit sources, not canonical inputs

- ESPN salary pages use `https://www.espn.com/nba/salaries/_/year/{season_end}/page/{page}` and expose ESPN IDs, but early coverage is incomplete and the current [Disney/ESPN terms](https://disneytermsofuse.com/english/) expressly prohibit scripted extraction, web scraping, dataset construction, and commercial use without permission. Do not automate this source.
- Basketball-Reference's own [salary methodology](https://www.basketball-reference.com/about/salary.html) says that it assigns minimum salaries to missing rows through 2004-05 and may extrapolate other gaps; its [data-use policy](https://www.sports-reference.com/data_use.html) restricts bulk reuse. It is unsuitable for exact early cap accounting.
- Spotrac is the strongest readily visible cap-accounting product, including cap hit, cash, incentives, and dead money. However, it has no public NBA-ID API, its 2002 pages are visibly incomplete, and its [terms](https://www.spotrac.com/service/) prohibit systematic extraction and database republication. Use only under a negotiated license.
- HoopsHype provides historical salary, not cap hit. The open-source [`hoopR` documentation](https://sportsdataverse.r-universe.dev/hoopR/doc/manual.html) describes an ending-year season convention and HoopsHype IDs, but public deployment still requires source permission.
- The official [NBA transactions page](https://www.nba.com/players/transactions) identifies signing, re-signing, waiver, and trade events but normally omits dollars. NBA Stats `CommonAllPlayers` is useful as an NBA-ID/name/career-year crosswalk, not as a salary source; its schema is documented by the MIT-licensed [`nba_api` client](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/commonallplayers.md). The client license is not a license to republish NBA data.

## Official salary-cap denominators

Maintain a small checked-in denominator table. The primary sources are the [NBA Communications salary-cap archive](https://pr.nba.com/tag/salary-cap/) and its [second page](https://pr.nba.com/tag/salary-cap/page/2/). Special official releases cover [2011-12](https://pr.nba.com/nba-cba-ratify/) and [2020-21](https://www.nba.com/news/nba-and-nbpa-agree-on-2020-21-season-start-and-adjustments-to-cba); the official [2009-10 release](https://www.nba.com/lakers/news/0910_nba_salarycap.html) also states the 2008-09 cap. Archived NBA releases should be retained for the earlier seasons, with a second-source reconciliation.

| Title season end | NBA season | Cap ($M) |
|---:|:---:|---:|
| 2002 | 2001-02 | 42.500 |
| 2003 | 2002-03 | 40.271 |
| 2004 | 2003-04 | 43.840 |
| 2005 | 2004-05 | 43.870 |
| 2006 | 2005-06 | 49.500 |
| 2007 | 2006-07 | 53.135 |
| 2008 | 2007-08 | 55.630 |
| 2009 | 2008-09 | 58.680 |
| 2010 | 2009-10 | 57.700 |
| 2011 | 2010-11 | 58.044 |
| 2012 | 2011-12 | 58.044 |
| 2013 | 2012-13 | 58.044 |
| 2014 | 2013-14 | 58.679 |
| 2015 | 2014-15 | 63.065 |
| 2016 | 2015-16 | 70.000 |
| 2017 | 2016-17 | 94.143 |
| 2018 | 2017-18 | 99.093 |
| 2019 | 2018-19 | 101.869 |
| 2020 | 2019-20 | 109.140 |
| 2021 | 2020-21 | 109.140 |
| 2022 | 2021-22 | 112.414 |
| 2023 | 2022-23 | 123.655 |
| 2024 | 2023-24 | 136.021 |
| 2025 | 2024-25 | 140.588 |
| 2026 | 2025-26 | 154.647 |

The row-level research input is:

```text
salary_share_it = annual_base_salary_it / official_cap_t
```

Never replace missing salary with zero. Preserve `salary_source`, `source_player_id`, `nba_id`, `mapping_method`, `mapping_confidence`, `retrieved_at`, `source_url`, `salary`, `cap_hit` when separately available, and `official_cap`.

Do not silently reconstruct cap hit from salary. Incentives, minimum-salary reimbursement, dead money, waiver/stretch or amnesty treatment, trade bonuses, set-off, and other CBA accounting can make salary, cash, and cap hit differ.

## Realized salary-peer hurdle

The production headline does not ask a hand-set curve what EPM a salary should
buy. It uses the league's realized playoff rosters as the contract hurdle.

For any focal player-season with a frozen pre-playoff EPM rating and annual
salary coverage, match the focal salary share to the closest
salary-covered playoff rotation player on each historical playoff team outside
the focal season. Break exact distance ties toward the peer with more playoff
workload. Remove that peer's actual team-EPM contribution and add the focal
player's EPM at the peer's workload. The swap uses two realized roles: the
peer's regular-season workload changes team strength for qualification and
seeding, while the peer's playoff workload changes series strength. This is a
real roster slot in the destination team's historical league environment.

```text
peer_team_rating = original_team_rating
                   - peer_EPM  * peer_playoff_workload_share

swap_team_rating = original_team_rating
                   - peer_EPM  * peer_playoff_workload_share
                   + focal_EPM * peer_playoff_workload_share

regular_season_rating_shift = (focal_EPM - peer_EPM)
                              * peer_regular_season_workload_share

p_swap = sum_seed P(seed | regular_season_rating_shift)
                  * P(title | swap_team_rating, seed)

p_peer = sum_seed P(seed | 0)
                  * P(title | original_team_rating, seed)

placement_lift = p_swap - p_peer

Title Lift over Contract = average(placement_lift)
```

The same placements also produce **raw title lift** by replacing `peer_EPM`
with 0 in the baseline comparison. Thus raw lift and salary-peer lift differ
only in the value credited to the contract slot.

The focal player's own team does not need to make the playoffs. Nonplayoff
seasons use the destination salary peer's postseason workload, exactly like
playoff seasons. Only actual-team context, teammate help, and additive team
accounting require a focal postseason appearance and remain null otherwise.

### Era-local, budget-neutral v2

The full build also emits a research challenger that preserves the placement
simulation while changing two assumptions:

1. use destination playoff teams only from `focal season ± 4`, excluding the
   focal season; and
2. charge a signed opportunity cost when the closest salary peer is not an
   exact cap-share match.

For a destination peer:

```text
gap_pp = 100 * (focal_salary_share - peer_salary_share)

v2_RS_shift = (focal_EPM - peer_EPM) * peer_RS_workload
              - 0.067 * gap_pp

v2_playoff_rating = original_team_rating
                     - peer_EPM * peer_playoff_workload
                     + focal_EPM * peer_playoff_workload
                     - 0.078 * gap_pp
```

Positive gaps debit the focal player; negative gaps create the equal and
opposite credit. The `0.067` regular-season and `0.078` playoff prices are
robust within-season contribution slopes among playoff rotation players in the
8%-20% cap band. Using team-rating contribution slopes avoids double-counting
workload: the price already represents the expected team-rating value of one
additional cap point.

The export keeps `title_lift_over_contract` for the full-history production
comparison and adds `title_lift_over_contract_v2`. Supporting columns retain
the v2 placement count, actual and baseline title/qualification probabilities,
fixed-bracket and seeding components, signed mean cap gap, median/p90 absolute
gap, share of placements beyond five cap points, and mean RS/playoff debits.
The exact v2 identity is:

```text
title_lift_over_contract_v2
  = era_local_budget_fixed_bracket_title_lift
  + era_local_budget_seeding_title_lift
```

### Playoff-prior challenger

The build also exports an entering-playoffs EPM-prior sensitivity on the same
era-local, cap-gap-priced placements. For target postseason `y`, fit player O/D
translations from postseasons `[y-8, y-1]` only, centered on the frozen
pre-playoff EPM lineup expectation and ridge-shrunk to zero at `alpha=24000`.
Players with no earlier playoff possessions retain their exact pre-playoff EPM.

Regular-season qualification and seeding still use pre-playoff EPM. The
adjusted playoff rating is used for the focal player, removed salary peer,
destination roster, and opponent field before calculating series and bracket
odds. Supporting season columns are `playoff_adjusted_dpm`, the O/D/net
translation and prior-evidence counts, adjusted actual/baseline bracket odds,
and the fixed-bracket/seeding decomposition. Career columns sum those annual
probability differences. The exact identity is:

```text
playoff_adjusted_title_lift_over_contract_v2
  = playoff_adjusted_fixed_bracket_title_lift
  + playoff_adjusted_seeding_title_lift
```

The 2007-2025 forward playoff gate has 1,586 games and 294,451 possessions.
Adjusted minus pre-playoff-EPM MSE is `-31.53e-6`, where negative favors the
challenger, with game-clustered 95% interval `[-84.90e-6, +23.97e-6]`.
Because the interval crosses zero, this is a labeled research sensitivity, not
a replacement for era-local v2. It raises LeBron's career result from `1.057`
to `1.206` titles over contract; all peers, teammates, and opponents receive
the same date-safe adjustment rule.

### Lagged DARKO + playoff-prior challenger

The Help Index also exposes a separate, explicitly lagged DARKO sensitivity.
For target season `t`, it starts from the completed public DARKO row for season
`t-1`; same-season DARKO is never used. It then estimates zero-centered O/D
playoff residuals at `alpha=24000` using only the eight postseasons before `t`.
Players without prior playoff evidence retain their lagged DARKO exactly.

Every focal player, salary peer, destination teammate, and opponent is rebuilt
under the same timing rule. The model then recalibrates the regular-season and
playoff logistic game layers and runs the same era-local, cap-gap-priced
qualification, seeding, and four-series bracket simulation as EPM TLOC. Its
public coverage is 1998-2026: DARKO 1997 supplies the prior for 1998. The 2026
salary-share rows come from the pinned `player_sheets` extract above.

The build emits 11,256 player-seasons and 2,116 careers. Its 2006-2026 forward
playoff gate contains 1,760 games and 326,805 possessions. Adjusted minus
lagged-DARKO MSE is `-3.70e-6`, with a game-clustered 95% interval of
`[-62.58e-6, +53.80e-6]`. The interval crosses zero, so the playoff update is a
labeled research sensitivity—not evidence that it improves on lagged DARKO.

Build and publish with:

```bash
.venv/bin/python nba_pipeline/scripts/build_darko_playoff_title_lift.py
.venv/bin/python nba_pipeline/scripts/publish_darko_playoff_title_lift.py
```

Salary rows are annual cash salary, not cap hit. When a player has multiple
team-stint rows in one season, sum them before calculating salary share; use the
largest salary stint only as a display-team label for a nonplayoff row. This
prevents traded players from being compared on one partial salary segment.

The regular-season layer uses the actual historical schedule and full league
field. A leave-one-season-out-selected game logit turns each matchup's team-EPM
difference and home court into a win probability. For every destination team
and rating shift, 2,048 common normal draws use the matching Poisson-binomial
mean and variance to simulate win totals; the focal team is ranked against all
teams in its conference. The top eight qualify. The selected model uses 30,064
games, `C=0.05`, home advantage `0.4067` log odds, and `0.1433` log odds per
team-EPM point; mean season log loss is `0.5911`.

The playoff layer retains the realized historical 16-team field but re-slots
the destination team at each simulated seed, shifting the other seven teams in
record order. A playoff-game logistic model prices home court and team-EPM
difference, then an exact dynamic program converts those game odds into a
first-to-four series probability under a `2-2-1-1-1` schedule. The higher seed
owns home court. Exact series probabilities propagate through the standard
1-8/4-5 and 2-7/3-6 four-round bracket; only the regular-season seed layer is
simulated.

This is a schedule-based seed simulation, not a possession-by-possession replay
of all 30 teams. Win totals use a normal approximation to each schedule's
Poisson-binomial distribution, standard standings rank replaces historical
division and play-in tiebreak rules, and the seven non-focal playoff teams in
each conference remain the realized field. The focal team can miss the top
eight and receive zero title probability, but a simulated nonplayoff entrant's
roster is not substituted into the opponent field. These approximations keep
the salary-placement surface reproducible and computationally tractable and
must remain visible in reader methodology.

The current series model is selected by the one-standard-error rule on
leave-one-season-out game log loss. It uses 2,085 playoff games across 25
seasons and selects `C=0.05`, home advantage `0.4777` log odds, and `0.1320`
log odds per team-EPM point. The raw-game parser rejects synthetic trailing
`0-0` score reset rows before final-score recovery.

The delayed 2020 postseason is passed an explicit season label when selecting
its August 1 EPM snapshot. Inferring season from the August calendar date would
incorrectly label it 2021 and erase 2020 rating coverage.

This is a retrospective realized-contract comparison, not a signing-date
forecast of market value. It uses annual salary share, not cap hit; it does not
model availability expectations, contract guarantees, CBA exception value,
tax/apron constraints, or transaction feasibility. A max player's peer pool is
also thinner than a mid-market player's, so the reader and export retain median
and 90th-percentile salary-match gaps.

## Prototype Contract-EPM sensitivity

### What is being estimated

Salary purchases both impact rate and expected availability. Therefore the economically sound model prices **expected season value/WAR at signing**, then converts that value into an EPM communication hurdle at one declared normalized workload:

```text
contract_WAR = (salary_share - replacement_salary_share) / market_salary_share_per_WAR

contract_EPM = replacement_EPM + contract_WAR / K_norm
```

`K_norm` is WAR generated by +1 EPM at the fixed normalized workload used by the Title Lift model. This keeps “a supermax should perform like +3 EPM” as an estimated, auditable result rather than a hand-set debit.

This signing-date market curve remains the economically richer future research
target. It is not the current headline hurdle. The public row instead shows
actual EPM, salary share, average realized salary-peer EPM, placement count,
salary-match diagnostics, and Title Lift over Contract.

### Fit only a market-identifying sample

Fit the price curve on unrestricted free-agent contracts using information available **when the contract was signed**:

- response: first-year cap share or the present value of guaranteed cap shares;
- value: expected WAR from a frozen pre-signing EPM forecast, age curve, and availability forecast;
- unit: one contract, not every contract-year as an independent observation;
- estimator: monotone or otherwise economically constrained, with player/contract clustering and signing-cohort cross-validation.

Never train on subsequent realized EPM, same-season playoff performance, awards earned later, or the eventual Title Lift result. For retrospective history, build each signing forecast from only pre-signing seasons and cross-fit by signing cohort; an expanding-window fit is preferable. This avoids grading a contract with outcomes that were unknowable when it was priced.

### Constrained contract regimes

The CBA makes one universal cap-share curve misleading. The current official [CBA summary](https://cms.nba.com/wp-content/uploads/sites/4/2024/11/2024-25-CBA-101.pdf) describes the normal maximum salary bands of 25%, 30%, and 35% of the cap by service, subject to higher-max exceptions, while rookie-scale salaries are preset by the CBA.

| Regime | Use to fit market curve? | How to score |
|:---|:---:|:---|
| Unrestricted free agent | Yes | Primary market-pricing sample |
| Extension, RFA, sign-and-trade | No | Apply UFA opportunity-cost curve; badge `constrained` |
| Rookie scale, two-way, minimum | No | Apply the same curve only as `CBA-price surplus`, not negotiation skill |
| Max or supermax | No ordinary regression row | Treat salary as right-censored by the 25/30/35% ceiling; report a max-constrained sensitivity/interval |

At the top end, fit the non-max UFA curve with a censored likelihood or report a range from defensible extrapolations. Do not let capped salaries flatten the market curve and make every max player appear to have a low hurdle. A genuine max-induced surplus is still economically meaningful, but it is surplus created partly by the CBA.

## Title Lift over Contract calculation

The production calculation averages full-season placements:

```text
p_swap = title_probability(peer roster with focal EPM,
                           peer RS and playoff workloads,
                           simulated qualification and seed)
p_peer = title_probability(original peer roster,
                           simulated qualification and seed)

Title Lift over Contract = average(p_swap - p_peer)
```

The subtraction happens after both the nonlinear seed mixture and four-round
bracket calculation for each placement. Do not subtract EPM linearly after
title probability has been calculated, and do not replace the opponent
distribution with 15 equal teams.

For auditability, the export retains the former same-bracket result. The exact
identity is:

```text
full-season Title Lift over Contract
  = fixed-bracket playoff lift
  + seeding and qualification lift
```

The primary reader metric is **realized salary-peer title lift at the peer's
actual regular-season and playoff workloads**. A separately labeled signing-date expected-value
metric may later include availability and market pricing, but it must not be
mixed into this retrospective result.

The research-only [scalar challenger](./CONTRACT_TITLE_CURVE_CHALLENGER.md)
tests the simpler `EPM surplus -> title-lift curve` summary with
leave-one-season-out predictions. EPM surplus alone is materially incomplete;
expected salary-slot workload is required to avoid granting small contract
roles max-contract title leverage. The role-aware shortcut closely compresses
the current result but is trained against it and therefore does not validate
the underlying bracket model.

The era-local, budget-neutral full-placement challenger is a different test:
it keeps the nonlinear bracket integration rather than compressing it to one
curve. On the rebuilt career board, LeBron is `1.057` titles, Curry `0.672`,
and their gap is `0.385`; the corresponding full-history values are `1.417`,
`1.235`, and `0.182`. Career correlation is `0.960` across 2,275 common rows.

## Release gate

Before publishing 2002-2026 results:

1. Freeze the salary snapshot, denominator table, ID crosswalk, regime labels, and source hashes.
2. Report source coverage and unresolved-ID rates by season; no unknown salary becomes zero.
3. Reconcile the BDL overlap against the historical salary source and explain material salary-versus-cap-hit differences.
4. Reconstruct every regular-season schedule plus 15 playoff series and the
   8-4-2-1 bracket pattern in every season.
5. Validate both game models by leave-one-season-out log loss; test seed
   monotonicity, series symmetry, bracket closure, exact swap-minus-peer
   identity, and exact full-season = fixed-bracket + seeding closure.
6. Report peer-placement counts, raw mean gap, median gap, p90 gap, and the
   share outside 3- and 5-cap-point calipers; do not hide thin max-salary
   matches behind an average peer EPM.
7. Keep the signing-date market curve labeled prototype until its UFA,
   constrained, rookie/minimum, and max-censored regimes pass their own gate.
8. Publish derived salary share and metric rows only until raw-data publication rights are confirmed.

## Playoff injury extension (source audit, 2026-10-11)

The focal player's own injury availability is not included in salary-peer lift: the swap inherits the destination peer workload. Actual-team exposure accounting reflects absences but does not label their medical causes. The [injury extension receipt](./PLAYOFF_INJURY_TITLE_LIFT.md) audits six historical cases and specifies a separate game-varying healthy-versus-injured title-chance comparison. Keep current metric fields unchanged; no injury-loss probability has been computed or published. [Primary-source evidence](./PLAYOFF_INJURY_SOURCE_RESEARCH.md) supports the pilot, while complete historical source coverage remains unverified.

### Participation-scaled sensitivity (implemented)

The owner-selected reader extension multiplies each season’s original lift by actual playoff participation (games appeared / actual team playoff games), or a shrunk other-career-season estimate for a known nonplayoff team. Nested leave-one-season-out tuning selects a 32-game league prior; the target season is excluded from player history, league baseline and tuning. Career values sum adjusted seasons with coverage counts. The new fields coexist with every original title model and do not modify their accounting identities. Zero-appearance roster proxies are explicitly labeled, unresolved assignments stay missing, and the estimate may use future career seasons retrospectively. See the [implementation and coverage receipt](./PLAYOFF_INJURY_TITLE_LIFT.md#implemented-participation-sensitivity-2026-10-11). This is signed participation scaling, not injury-only title-chance loss or a contract-cost penalty.
