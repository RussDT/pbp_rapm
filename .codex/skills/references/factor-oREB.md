# oREB: Offensive Rebounding Rate RAPM

## Definition

How much a player affects their team's offensive rebounding rate when they are on the court, isolated via ridge regression controlling for every teammate and opponent.

**Units**: OREB% (native), converted to points via reconstruction beta weight.

**Sign convention**: Positive = team grabs more offensive rebounds when this player plays. More OREBs = more second-chance possessions = more scoring opportunities. This is a possession-extension factor.

## What Shows Up Here

oREB captures everything that affects whether your team gets the ball back after a missed shot:

### Current 2026 Prior Framing

For current 2026 hybrid / true-prior work, do not describe the canonical oREB prior as the old `shot_impact_on_oreb` proxy.

The current prior is a **position-adjusted self-OREB burden model** built around:

- `ProbabilityOffRebounded`
- `SelfORebPct`
- position
- shot mix / miss ecology

The key engineered fields are:

- `expected_self_oreb_pct`
- `oreb_burden_resid`
- `reboundable_miss_volume_100`
- `oreb_burden_impact`

Use language around rebound burden, reboundable misses, self-OREB rate, and position-specific role expectations.

### Direct Rebounding
- **Crashing the glass** -- second chances. Individual offensive rebounding ability through positioning, timing, effort, and athleticism.
- **Self-OREB rate**: How often a player rebounds their own miss. Some players are elite at this (putback ability after missing at the rim).
- Glass-crashing frequency and commitment

### Shot-Type Impact on Team OREBs
This is a nuanced and under-discussed mechanism:

- **Rebound burden**: Some players, especially bigs, carry a role-based burden to be near the rim when misses happen. If that player shifts farther from the basket as a shooter or creator, the offense may lose its best rebounder exactly when a miss becomes reboundable.
- **The Center LMR Problem**: When a center like Anthony Davis or Joel Embiid takes a long midrange shot, there is no longer a center near the hoop to OREB the miss. The team's biggest rebounder has abandoned the glass to become a shooter. This is a straightforward causal mechanism -- the center takes on a burden to OREB their own misses (and their teammates' misses) that they abandon by taking shots far from the basket.
- **Transition shots**: A shot taken in transition where there are 3 defenders ready to rebound vs just the shooter matters. If nobody is crashing, the miss is an automatic defensive rebound.
- **Shot-type OREB rates**: Different shots have different OREB rates when missed. Long midrange misses are among the hardest to OREB because they produce long rebounds that go to the defense. Rim-area misses are the easiest to OREB because rebounders are already there.
- **Team scheme**: Some teams scheme around OREBbing certain players' misses. Others don't. This can be noisy -- a team might not bother crashing on a center's long two, making that shot look even worse for oREB.

### Positional Effects
- Centers dominate the positive end of oREB because they're positioned near the basket and have size/positioning advantages
- Guards and wings are typically neutral or negative because they're the first back in transition
- A guard leaking out in transition sacrifices oREB potential for transition defense/offense positioning

## Historical Box-Score Regression Drivers (5Y Model)

| Metric | Ridge R^2 | XGBoost R^2 | Spearman |
|--------|----------|-------------|----------|
| **OFF_REB** | 0.673 | 0.662 | 0.752 |

**Top predictors** (in order of importance):
1. **OffFGRebPct** -- Offensive field goal rebound percentage. The most direct counting measure of individual offensive rebounding.
2. **OffReb/100** -- Offensive rebounds per 100 possessions. Volume measure.
3. **Legacy `shot_impact` proxy** -- OREB rate off a player's own misses relative to league average, scaled by usage. Captures the shot-selection-to-rebounding pipeline, but is not the canonical 2026 prior concept.
4. **pos_C** -- Whether the player is a center. Strong positional signal -- bigs near the basket dominate OREBs.
5. **SelfORebPct** -- How often a player offensive rebounds their own miss. Putback and tip-in ability.

**What the R^2 = 0.673 means**: Box scores explain ~67% of lineup-level offensive rebounding impact. This is the highest R^2 of all six factors -- offensive rebounding is the most "individual" and box-score-visible factor. If a guy crashes the boards hard, it shows up both in counting stats AND in lineup data. The remaining ~33% likely captures team scheme effects, positional abandonment (centers shooting away from the basket), and the subtle ways some players create rebounding opportunities for teammates (e.g., boxing out the opponent's rebounder so a teammate can grab it).

**2026 note**: For current prior work, use the rebound-burden framing above rather than presenting `shot_impact` as the canonical concept.

## Player Examples (2021-2026 Time-Decayed)

### Elite oREB (Team Gets More OREBs)
| Player | oREB | Context |
|--------|------|---------|
| Steven Adams | +4.04 | The archetype. Elite glass crasher. His team gets dramatically more second chances. |
| Moussa Diabate | +3.35 | Pure hustle rebounder |
| Mitchell Robinson | +3.28 | Elite OREB machine. Compensates for negative oTS (-2.28). |
| Day'Ron Sharpe | +2.53 | Young center who crashes hard |
| Tari Eason | +2.50 | Active forward who generates second chances |
| Amen Thompson | +2.11 | Athletic wing/forward who crashes aggressively |
| Kevon Looney | +1.97 | Undersized but elite positioning and effort |
| Zach Edey | +1.92 | Size and positioning advantage |
| Toumani Camara | +1.85 | Hustle forward |
| Luke Kornet | +1.63 | Rim presence creates rebounding opportunities |
| Isaiah Hartenstein | +1.61 | Modern big who still crashes the glass |
| Rudy Gobert | +1.60 | Rim dweller who extends possessions |

### Negative oREB (Team Gets Fewer OREBs)
| Player | oREB | Context |
|--------|------|---------|
| Joel Embiid | -1.45 | **The Center LMR archetype**. Takes the most long midrange shots of any center. When he's pulling up from 15-20 feet, there's no center near the rim to OREB the miss. His shooting ability (oTS +3.23) comes at a rebounding cost. |
| Myles Turner | -1.31 | Stretch big who spaces to the three-point line. Same tradeoff as Embiid but from deeper. |
| Draymond Green | -1.05 | Not positioned near the rim on offense. Facilitates from the perimeter. |
| Al Horford | -0.91 | Stretch big archetype |
| Paul George | -0.88 | Wing who doesn't crash the boards |
| Luka Doncic | -0.80 | Ball handler who's never near the rim for OREBs |

### The Adams vs Embiid Contrast
This is the clearest illustration of the oTS-oREB tradeoff:
- **Steven Adams**: oTS -0.92, oREB +4.04. Hurts team shooting (no spacing) but creates a massive rebounding advantage.
- **Joel Embiid**: oTS +3.23, oREB -1.45. Dramatically improves team shooting but his shot selection (especially LMR) means nobody is near the rim to OREB his misses.

Both are centers. Completely opposite offensive profiles. The decomposition reveals the tradeoff that total ORAPM hides.

## Cross-Factor Interactions

- **oREB vs oTS tradeoff**: The most common and important cross-factor interaction. Non-shooting bigs trade oTS for oREB. Stretch bigs trade oREB for oTS. See Adams vs Embiid above.
- **oREB and transition**: A team that crashes the offensive glass sacrifices transition defense/offense. A player with high oREB might indirectly affect dTS or dREB on the other end because their team can't get back in transition.
- **oREB and oTOV**: Non-passing bigs often show both negative oTOV (post turnovers) and positive oREB. The ball goes through them near the rim -- sometimes it's a turnover, sometimes it's a miss they or a teammate rebounds.
- **oREB and dREB (same player)**: A player who is an elite offensive rebounder is often also positioned to affect defensive rebounding. Steven Adams: oREB +4.04, dREB +1.10. His rim presence works both ways.
- **The rebound-burden / miss-ecology pipeline**: A player's shot selection (which affects oTS) also affects oREB through the reboundability of misses and through who is positioned to recover them. This creates a mechanical link between WHERE you shoot and how many second chances your team gets.

## Caveats

- **Legacy `shot_impact` language is not canonical for 2026 prior work**: If you're describing the current hybrid / true-prior process, use the position-adjusted self-OREB burden framing instead.
- **Lineup-level**: A player's oREB-RAPM includes the effect of who else is on the court. A guard playing next to two glass-crashing bigs might show neutral oREB because the bigs are doing the work.
- **Positional confounding**: Centers dominate both the positive and (via the LMR tradeoff) some negative oREB outcomes. Position is both a real signal and a confound.
- **Scheme dependency**: Some teams prioritize offensive rebounding in their system. A player in a crash-heavy scheme will show higher oREB than the same player in a "get back" scheme. The regression tries to control for this via teammate effects, but imperfectly.
