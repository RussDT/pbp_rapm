# oTOV: Offensive Turnover Rate RAPM

## Definition

How much a player affects their team's turnover rate when they are on the court, isolated via ridge regression controlling for every teammate and opponent.

**Units**: TOV% (native), converted to points via reconstruction beta weight.

**Sign convention**: Positive = team turns the ball over LESS when this player plays. Fewer turnovers = more possessions finishing with a shot = more scoring opportunities. This is a possession-control factor (not an efficiency factor).

## What Shows Up Here

oTOV captures everything that affects whether your team successfully keeps the ball and completes possessions:

### Direct Ball Security
- **Ball security**: Not forcing passes. Reading the defense. Taking care of it.
- Ball handling quality under pressure
- Passing accuracy and decision-making
- Not forcing plays into traffic
- Protecting the ball in the post

### Decision-Making and Tempo
- Understanding defensive rotations and reading when passing windows close
- Not over-dribbling into help defense
- Making the simple pass that leads to the easy pass (hockey-assist quality)
- Tempo management -- a team that plays under control turns it over less
- Knowing when to reset vs when to attack

### Role-Adjustment by Nature
oTOV is implicitly role-adjusted. High-usage players handle the ball more, pass more, and have more chances to turn it over. Raw turnover counts can mislead -- a point guard with 3 turnovers a game might actually be elite relative to their ball handling volume. What matters here is the team's turnover rate when you play vs when you sit. The lineup data handles the role context implicitly.

### Creation Burden
- High-usage creators who handle the ball frequently face more turnover-generating situations
- The relationship between oTOV and usage is non-linear -- some high-usage players are elite at protecting the ball, others trade turnovers for creation
- "Scoring turnovers" (lost-ball turnovers while trying to score) vs "bad pass turnovers" (passing errors) are distinct skills

### The Stonehands Problem
If a player can't catch the ball, the turnover sometimes gets credited to the passer in the box score. Doesn't matter here -- this is lineup-level data. When you're on the floor and the team turns it over more, that's your oTOV, regardless of whose name ends up in the play-by-play. A player with bad hands will show up as negative oTOV even if the box score blames his teammates.

### Team-Level Effects
- Offensive system complexity affects team turnover rates
- A player who simplifies the offense (fewer reads, cleaner actions) can reduce team turnovers
- Conversely, a player running a complex motion offense might temporarily increase turnovers

## Box-Score Regression Drivers (5Y Model)

| Metric | Ridge R^2 | XGBoost R^2 | Spearman |
|--------|----------|-------------|----------|
| **OFF_TOV** | 0.591 | 0.606 | 0.704 |

**Top predictors** (in order of importance):
1. **scoring_tovs** -- Lost-ball turnovers in scoring situations. The most direct measure of ball security under defensive pressure.
2. **badpass_tovs** -- Passing errors. Distinct from scoring turnovers -- measures passing quality and vision.
3. **Usage** -- Higher-usage players handle the ball more, creating more turnover opportunities (but also more chances to avoid them).
4. **Pot Assists** (Potential Assists) -- Players who create a lot of passing opportunities are exposed to more bad-pass risk.
5. **creation_TSA** -- Shot creation volume. More creation = more turnovers, but elite creators manage the tradeoff.

**What the R^2 = 0.591 means**: Box scores explain ~59% of lineup-level turnover impact. Similar explanatory power to oTS. The distinct turnover types (scoring vs bad pass) are strongly predictive, suggesting that individual ball-handling and passing decisions are a major driver of team turnover rates.

**Notable**: The two turnover type predictors (scoring_tovs, badpass_tovs) as top-2 drivers suggest oTOV is one of the more "individual" factors -- a player's own turnover tendencies strongly predict their team-level turnover impact.

## March 14, 2026: Clean Turnover Split And Burden Findings

These notes supersede the older "top predictors" shorthand when the task is specifically about current prior-building or turnover decomposition work.

### Clean 5Y split-prior harness

Current clean target:

- pure `2022-2026` regular-season targets
- no playoffs
- no time decay
- weighted by `5-year offposs`

Current best large-feature weighted 5-fold CV results:

- `oTOV_bp`: `R^2 = 0.5216`
- `oTOV_sc`: `R^2 = 0.4848`
- `oTOV`: `R^2 = 0.5778`

Important result:

- the split estimate beats the direct total-turnover fit on the common modeled sample
- `pred_oTOV_bp + pred_oTOV_sc`: weighted `R^2 = 0.6162`
- direct `pred_oTOV`: weighted `R^2 = 0.6109`

Interpretation:

- for predictive turnover priors, scoring-turnover and bad-pass-turnover lanes should be modeled separately when possible
- the split carries real signal that gets blurred when everything is collapsed into one total-turnover prior

### Bad-pass RAPM is not just "bad-pass turnovers"

Current `oTOV_bp` findings:

- a bad-pass-only model fails
- `true_badpass_tovs_100` alone was near zero explanatory power against clean `oTOV_bp`
- a lean bad-pass-centered model can still work if it keeps burden and context
- dropping `TOV_100` and `scoring_tovs_100` but keeping playmaking / playtype context still held around `R^2 = 0.516`

Most important additive layer beyond raw turnover counts:

- playtype pressure diagnostics

Current ablation result:

- full `oTOV_bp` model: `R^2 = 0.5216`
- remove the playtype block: `R^2 = 0.4570`

Interpretation:

- bad-pass RAPM needs turnover lane volume plus role context
- playtype pressure is the main thing that rescues the bad-pass model beyond simple bad-pass box rates

### Compact offensive-load framing

For current clean `oTOV` work, the best compact framing is:

`oTOV ~ turnover rate term + load term`

Current preferred interpretation:

- the turnover term captures actual turnover outcomes
- the load term captures offensive burden that creates turnover exposure
- this is the right setup if the downstream goal is a turnover-efficiency concept

Current best simple fixed-`TOV_100` load search:

- `TOV_100 + creation_TSA100`: weighted CV `R^2 = 0.4876`
- `TOV_100 + offensive_load`: `0.4813`

So the current best single "load" term is:

- `creation_TSA100`

Current compact load build path:

- add `PASSING_Potential Assists/100` -> `R^2 = 0.5216`
- add `transition_TSA100` -> `0.5440`
- add `shooting_TSA100` -> `0.5558`
- add `PASSING_on-ball-time%` instead of finishing volume -> `0.5580`

Current compact regression:

`oTOV ≈ -0.170 - 0.550*z(TOV_100) + 0.463*z(creation_TSA100) + 0.310*z(Potential Assists/100) + 0.089*z(shooting_TSA100) + 0.081*z(transition_TSA100) - 0.138*z(on-ball-time%)`

Load-only portion:

`Load_v1_raw = 0.463*z(creation_TSA100) + 0.310*z(Potential Assists/100) + 0.089*z(shooting_TSA100) + 0.081*z(transition_TSA100) - 0.138*z(on-ball-time%)`

Practical reading:

- `creation_TSA100` is the anchor load ingredient
- potential assists are the second key burden axis
- shooting and transition add modest load signal
- once creation and passing burden are accounted for, extra on-ball time tilts negative and behaves more like a style / efficiency tax than pure burden

### Current hub-creator caveat

The compact load formula is still too guard-biased to be treated as final "true offensive load."

Observed issue:

- it ranks heliocreator guards correctly
- but it under-credits hub big creators like Nikola Jokic because it leans too heavily on self-created scoring burden

Current implication:

- any public offensive-load stat should probably add one hub-creation term before it is treated as finished
- best current candidate: `box_creation`
- `PASSING_AST PTSCreated/100` is also useful, but weaker

Current add-one screen on top of the compact base:

- add `box_creation` -> `R^2 = 0.5670`
- add `PASSING_AVG_DRIB_PER_TOUCH` -> `0.5619`
- add `PASSING_AVG_SEC_PER_TOUCH` -> `0.5606`
- add `PASSING_on-ball-time%` -> `0.5580`
- add `PASSING_AST PTSCreated/100` -> `0.5580`
- `PASSING_PotAss/Passes` and `touch_burden` did not help

Interpretation:

- raw touch volume is too blunt
- touch style helps some
- a hub-creation proxy helps the most

## Player Examples (2021-2026 Time-Decayed)

### Elite oTOV (Team Turns It Over Less)
| Player | oTOV | Context |
|--------|------|---------|
| SGA | +2.61 | Elite ball security WHILE being extremely high-usage. Rarely forces bad plays. |
| Jalen Brunson | +1.54 | Controls possessions carefully. High assist load with low turnover rate. |
| Pascal Siakam | +1.51 | Disciplined with the ball for his usage level |
| Tyrese Maxey | +1.46 | Quick decision-maker who doesn't over-dribble |
| Kawhi Leonard | +1.38 | Strong hands, methodical approach, rarely forces |
| Kyrie Irving | +1.35 | Elite handles. Ball almost never leaves his hands involuntarily. |
| Fred VanVleet | +1.33 | Floor general who runs a controlled offense |
| Jamal Murray | +1.29 | Poised ball handler who manages possessions well |
| Donovan Mitchell | +1.27 | Efficient creator who protects the ball |
| Miles McBride | +1.26 | Backup PG who runs a clean offense when he plays |

### Negative oTOV (Team Turns It Over More)
| Player | oTOV | Context |
|--------|------|---------|
| Day'Ron Sharpe | -1.52 | Center handling the ball in post = turnover risk |
| Rudy Gobert | -1.48 | Post touches with limited passing ability. Team turns it over more. |
| Ivica Zubac | -1.48 | Same archetype -- big man post-up turnovers |
| Steven Adams | -1.13 | Despite elite oREB, his offensive possessions generate turnovers |
| Mitchell Robinson | -1.04 | Non-passing center -- turnovers when the ball finds him |
| Kevin Durant | -0.95 | Surprising. High-usage scorer with some ball-handling risk. |

**Pattern**: Non-passing centers who receive the ball in the post dominate the negative end. Their team's offensive possessions that run through them have higher turnover rates. Guards/wings with high usage but elite handles dominate the positive end.

### The SGA Archetype
SGA at +2.61 oTOV is remarkable because he's also one of the highest-usage players in basketball. Most high-usage players trade off some turnover rate for creation volume. SGA manages both -- elite ball security AND elite creation. This is one reason his total RAPM is so high (+7.39).

## Cross-Factor Interactions

- **oTOV and oTS**: Can be additive (SGA: +2.89 oTS, +2.61 oTOV -- efficient AND careful) or trade off (Durant: +4.48 oTS, -0.95 oTOV -- efficient but occasionally careless).
- **oTOV and Usage**: High-usage players have a wider range of oTOV outcomes. Some manage both (SGA, Brunson), others sacrifice turnovers for creation.
- **oTOV and oREB as possession control**: Both are about whether the possession continues or ends. oTOV is about not giving the ball away; oREB is about getting it back after a miss. A player who is strong on both is a possession-control machine.
- **oTOV and dTOV**: Not necessarily correlated. A player can be careful with the ball (high oTOV) without being disruptive defensively, and vice versa.
- **Center archetype**: Non-shooting bigs often show negative oTOV (post turnovers) alongside positive oREB. The ball goes through them in the post, sometimes it's a turnover, sometimes it's a miss that they or a teammate rebounds.

## Caveats

- Lineup-level. A team's offensive system affects turnover rates -- a player inserted into a complex motion offense may show higher team turnover rates regardless of their individual ball security.
- The scoring_tovs vs badpass_tovs distinction is important for understanding the mechanism but only shows up in box-score analysis, not in the RAPM factor itself.
- High oTOV does not mean a player is a good passer. It means their team turns it over less when they play. Could be a function of simplifying the offense, not just passing quality.
