# dTOV: Defensive Turnover Rate RAPM

## Definition

How much a player affects the opponent's turnover rate when they are on the court, isolated via ridge regression controlling for every teammate and opponent.

**Units**: TOV% (native), converted to points via reconstruction beta weight.

**Sign convention**: Positive = opponents turn the ball over MORE when this player plays. More opponent turnovers = more possessions for your team = more scoring opportunities. This is a possession-creation factor on the defensive end.

## What Shows Up Here

dTOV captures everything that causes opponents to lose the ball:

### Active Disruption
- **Steals** -- interceptions, reaching in, poking the ball loose. Active hands in passing lanes.
- Deflections that lead to turnovers (even if someone else recovers the ball)
- Trapping and pressure that force rushed decisions
- Showing on ball screens in a way that forces the ball handler to pick up their dribble
- **Drawing charges** -- taking away a possession by getting in position and drawing an offensive foul. A charge is a turnover.
- **Drawing offensive fouls** -- moving screens, hooks, pushoffs by the opponent. The opponent loses the ball. These are turnovers that don't show up as steals but register in lineup data.

### Team Defensive Pressure
- Help rotations that force rushed decisions from the ball handler
- Defensive schemes that create confusion and miscommunication in the offense
- Positioning that makes the offense feel rushed or uncomfortable
- A player who's always in the right spot makes the team's pressure more effective
- Defensive IQ that anticipates and disrupts passing sequences

### The Gambling Spectrum
This is the most nuanced aspect of dTOV:

- **Controlled disruption**: Some players create turnovers through elite positioning and anticipation without compromising their team's defensive structure. They're in the passing lane but can recover if the pass doesn't come.
- **High-risk gambling**: Other players lunge for steals, leaving themselves out of position. They generate turnovers (high dTOV) but at a cost to defensive positioning (potentially negative dTS).
- **The question**: Does the dTOV gain outweigh the dTS cost? The decomposition lets you see whether the cost has materialized in the data, but doesn't hand you a clean causal answer because other factors (teammates, scheme, sample size) are in play.

## Box-Score Regression Drivers (5Y Model)

| Metric | Ridge R^2 | XGBoost R^2 | Spearman |
|--------|----------|-------------|----------|
| **DEF_TOV** | 0.548 | 0.576 | 0.666 |

**Top predictors** (in order of importance):
1. **rFTOV** -- Rate of forced turnovers (forced turnovers per some possession or minute base). The most direct measure of turnover-creating ability.
2. **FTOVs/100** -- Forced turnovers per 100 possessions. Volume version of the same signal.
3. **Steals** -- Classic box score metric. Strongly correlated but incomplete -- many turnovers happen without a steal being recorded.
4. **pos_PF** -- Whether the player is a power forward. Positional signal -- active PFs who switch and disrupt show up here.
5. **OFFD** -- Offensive fouls drawn. Getting in the way of offensive players, taking charges, and drawing illegal screens. A secondary turnover-creation mechanism.

**What the R^2 = 0.548 means**: Box scores explain ~55% of lineup-level defensive turnover impact. Moderate explanatory power -- the counting stats (steals, forced turnovers) are predictive, but about 45% of the signal comes from team defensive pressure, positioning, and scheme effects that don't show up in any individual stat line.

**Notable**: The top drivers being forced turnover rate metrics (not just raw steals) suggests that the broader forced-turnover concept is more predictive than steals alone. Many disruptions that lead to turnovers aren't recorded as steals -- a deflection that leads to an out-of-bounds, a charge drawn, a 5-second violation forced.

## Player Examples (2021-2026 Time-Decayed)

### Elite dTOV (Opponents Turn It Over More)
| Player | dTOV | dTS | Net Assessment |
|--------|------|-----|----------------|
| Alex Caruso | +3.10 | +1.48 | **The archetype**. Elite disruption WITHOUT dTS cost. His dTS is also positive, suggesting controlled gambling or excellent recovery/teammates. |
| Herbert Jones | +1.76 | +0.84 | Strong disruptor with positive dTS. Controlled defensive activity. |
| Paul Reed | +1.70 | -0.23 | High disruption with slight dTS cost. Possible gambling tradeoff showing up mildly. |
| Dyson Daniels | +1.65 | -0.12 | Active hands, young defender with slight dTS cost |
| Cason Wallace | +1.66 | +0.41 | Disruptive without defensive cost |
| Jalen Suggs | +1.35 | +0.32 | Active perimeter defender |
| Toumani Camara | +1.30 | +0.57 | Versatile disruptive forward |
| Fred VanVleet | +1.27 | +0.60 | Veteran ball hawk with controlled approach |
| Ausar Thompson | +1.25 | +0.71 | Young athletic disruptor with positive dTS |
| Paul George | +1.16 | +0.89 | Active hands, long wingspan |
| Marcus Smart | +1.10 | +0.08 | Famously aggressive but neutral dTS -- gambling risk roughly offset |
| OG Anunoby | +1.06 | +0.57 | Length and activity create turnovers |

### Negative dTOV (Opponents Turn It Over Less)
| Player | dTOV | Context |
|--------|------|---------|
| Steven Adams | -1.21 | Low mobility, doesn't pressure ball handlers or play passing lanes |
| Kevin Durant | -1.08 | Defensive engagement questions -- opponents comfortable with him on court |
| Kevon Looney | -1.03 | Slow feet, limited lateral quickness to force turnovers |
| Giannis Antetokounmpo | -1.01 | Interesting -- elite dTS (+1.74) but negative dTOV. He alters shots (dTS value) but doesn't generate turnovers. Different defensive skill. |
| Chet Holmgren | -0.91 | Great shot blocker (dTS +3.33) but doesn't force turnovers. Rim protector, not ball hawk. |
| Jamal Murray | -0.89 | Opponents comfortable with ball when he's defending |
| Zach Edey | -0.88 | Limited mobility |

**Pattern**: Slow-footed bigs and disengaged defenders dominate the negative end. Active guards and versatile wings dominate the positive end. Notably, some elite defenders by dTS (Giannis, Chet) are negative on dTOV -- they're shot alterers, not turnover creators. Different defensive mechanisms.

### The Caruso Case Study
Alex Caruso at dTOV +3.10 is far and away the highest in the dataset. Combined with dTS +1.48, this means:
- Opponents turn the ball over dramatically more AND shoot worse when he plays
- His disruption is NOT coming at a dTS cost
- His total defensive RAPM is +4.24 -- almost entirely from dTOV + dTS
- On offense he's nearly invisible (oTS +0.24, off +0.64)
- Total net RAPM +4.89 -- a star-level impact driven almost entirely by defensive disruption

This is exactly the kind of player profile that box scores undersell. His steal numbers are good but not mind-blowing. His dTOV-RAPM tells you his team-level defensive disruption is historically elite.

## Cross-Factor Interactions

### dTOV and dTS (The Gambling Tradeoff)
The most important interaction in the framework. See extended discussion in [factor-dTS.md](factor-dTS.md).

Quick summary: Look at both together to see if a player's turnover-creating activity is costing their team on defensive efficiency:
- **Caruso**: dTOV +3.10, dTS +1.48. No cost. Elite in both.
- **Giannis**: dTOV -1.01, dTS +1.74. Opposite profile -- shot alterer, not disruptor.
- **Marcus Smart**: dTOV +1.10, dTS +0.08. High disruption, roughly neutral efficiency. The gambling risk exists but approximately breaks even.
- **Paul Reed**: dTOV +1.70, dTS -0.23. Mild dTS cost. Possibly some gambling showing up.

### dTOV and Position
Guards and active wings dominate elite dTOV. Centers rarely show up positive because their defensive value comes through rim protection (dTS) and rebounding (dREB) rather than turnover creation. The exceptions are switchable bigs who play in passing lanes.

### dTOV and oTOV
Not necessarily correlated. A player can be a ball hawk on defense (high dTOV) while being careful with the ball on offense (high oTOV), or vice versa. These are mechanically unrelated skills. SGA is elite at both (+2.61 oTOV, +1.01 dTOV). Caruso is elite dTOV (+3.10) but modest oTOV (+0.33).

### dTOV as Possession Creation
dTOV and oREB are the two possession-control factors. dTOV creates possessions by forcing opponent turnovers. oREB extends possessions by grabbing offensive rebounds. Both feed back into scoring opportunities. A player elite at both is a possession-control machine. But they're typically different player types -- oREB skews toward bigs, dTOV skews toward guards/wings.

## Caveats

- **Gambling interpretation requires caution**: A high-dTOV, low-dTS player is *consistent with* gambling, but could also reflect teammates, scheme, opponent quality, or sample noise. The framework surfaces patterns to investigate, not causal conclusions.
- **Lineup-level**: Defensive scheme and teammates affect how often opponents turn it over. A player in an aggressive trapping scheme will show higher team dTOV than the same player in a conservative scheme.
- **Steals are incomplete**: Box-score steals capture a subset of turnover-creating events. Many forced turnovers (charges drawn, deflections leading to out-of-bounds, 5-second violations, bad passes forced by pressure) aren't recorded as steals.
- **For the pure series, no box priors**: Multi-season time-decay helps with stability, but read low-minute players' dTOV values with appropriate uncertainty.
