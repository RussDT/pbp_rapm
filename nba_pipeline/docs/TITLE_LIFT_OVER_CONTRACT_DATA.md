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

For a focal player-season, match the focal salary share to the closest
salary-covered playoff rotation player on each historical playoff team outside
the focal season. Break exact distance ties toward the peer with more playoff
workload. Remove that peer's actual team-EPM contribution and add the focal
player's EPM at the peer's workload. This is a real roster slot: the teammates,
opponents, workload, and four-round bracket all remain those of the destination
team and season.

```text
peer_team_rating = original_team_rating
                   - peer_EPM  * peer_playoff_workload_share

swap_team_rating = original_team_rating
                   - peer_EPM  * peer_playoff_workload_share
                   + focal_EPM * peer_playoff_workload_share

placement_lift = P(title | swap_team_rating, historical_bracket)
                 - P(title | original_team_rating, historical_bracket)

Title Lift over Contract = average(placement_lift)
```

The same placements also produce **raw title lift** by replacing `peer_EPM`
with 0 in the baseline comparison. Thus raw lift and salary-peer lift differ
only in the value credited to the contract slot.

The bracket engine uses the observed 16-team topology for each season. A
playoff-game logistic model prices home court and team-EPM difference, then an
exact dynamic program converts those game odds into a first-to-four series
probability under a `2-2-1-1-1` schedule. Either team is averaged as the home-
court owner because seeding is not used as a talent credit. Exact series
probabilities are propagated through all four rounds; Monte Carlo sampling is
unnecessary and would add noise to the same expectation.

The current series model is selected by the one-standard-error rule on
leave-one-season-out game log loss. It uses 2,085 playoff games across 25
seasons and selects `C=0.05`, home advantage `0.4783` log odds, and `0.1342`
log odds per team-EPM point. The raw-game parser rejects synthetic trailing
`0-0` score reset rows before final-score recovery.

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

The production calculation averages actual bracket placements:

```text
p_swap = title_probability(peer roster with focal EPM, peer workload, actual bracket)
p_peer = title_probability(original peer roster, actual bracket)

Title Lift over Contract = average(p_swap - p_peer)
```

The subtraction happens after the nonlinear four-round bracket calculation for
each placement. Do not subtract EPM linearly after title probability has been
calculated, and do not replace the opponent distribution with 15 equal teams.

The primary reader metric is **realized salary-peer title lift at the peer's
actual playoff workload**. A separately labeled signing-date expected-value
metric may later include availability and market pricing, but it must not be
mixed into this retrospective result.

## Release gate

Before publishing 2002-2026 results:

1. Freeze the salary snapshot, denominator table, ID crosswalk, regime labels, and source hashes.
2. Report source coverage and unresolved-ID rates by season; no unknown salary becomes zero.
3. Reconcile the BDL overlap against the historical salary source and explain material salary-versus-cap-hit differences.
4. Reconstruct 15 series and the 8-4-2-1 bracket pattern in every season.
5. Validate the game model by leave-one-season-out log loss and test series
   symmetry, monotonicity, bracket closure, and exact swap-minus-peer identity.
6. Report peer-placement counts and salary-match gaps; do not hide thin max-
   salary matches behind an average peer EPM.
7. Keep the signing-date market curve labeled prototype until its UFA,
   constrained, rookie/minimum, and max-censored regimes pass their own gate.
8. Publish derived salary share and metric rows only until raw-data publication rights are confirmed.
