# dTS: Defensive True Shooting RAPM

## Definition

How much a player affects opponent shooting efficiency (True Shooting %) when they are on the court, isolated via ridge regression controlling for every teammate and opponent.

**Units**: TS% (native), converted to points via reconstruction beta weight.

**Sign convention**: Positive = opponents shoot WORSE when this player plays. This is the defensive efficiency factor -- the defensive counterpart to oTS and typically the second-highest-weighted factor in the reconstruction regression.

## What Shows Up Here

dTS captures everything that affects how efficiently opponents convert possessions into points via shooting. This is arguably the hardest factor to predict from box scores because much of defensive impact is invisible:

### Direct Shot Defense
- **Rim protection** -- contesting and altering shots at the basket. The single most valuable defensive skill in basketball.
- **Rim deterrence** -- making opponents avoid the rim entirely. The shot that never gets taken because the rim protector is lurking. This doesn't show up in any counting stat but registers in lineup data.
- **Contest quality** -- being well-positioned, making shooters uncomfortable. Getting a hand up on the release.
- **Closeout quality** -- arriving on balance without flying past the shooter. Controlled closeouts contest the shot; wild ones leave the defense in rotation.
- **Getting blown by → negative dTS.** When a defender can't stay in front, the result is easy baskets at the rim for the opponent. Perimeter defenders who get beaten off the dribble are contributing negatively to dTS.
- Help defense shot alteration -- rotating to contest shots away from your assignment
- Post defense -- making it hard for bigs to score inside

### Scheme and Positioning (Invisible Defense)
These are the things that make dTS so hard to capture with counting stats:

- **Defensive communication**: Calling out switches, helping teammates get into position, funneling ball handlers into help. A player who makes the whole defense more organized reduces opponent shooting efficiency -- even if his individual defensive stats look ordinary.
- **Help positioning**: Being in the right spot to deter drives without fully committing. The "stunt and recover" skill. Making the rim feel smaller even without a block attempt.
- **Closeout quality**: Closing out under control vs flying by. A controlled closeout contests the shot; a wild one leaves the defense in rotation.
- **Funneling**: Directing ball handlers into help defenders or toward less efficient shots. A perimeter defender who forces his man into the waiting rim protector is contributing to dTS even though the block goes to someone else.
- **Defensive scheme execution**: Running a scheme correctly (ICE, DROP, SWITCH, HEDGE) to force the offense into bad looks.

### ShotQuality Sub-Decomposition (Level 2)

dTS decomposes into the same three ShotQuality components as oTS (see [ts-decomposition.md](ts-decomposition.md)):

| Component | What It Isolates | Player Archetype |
|-----------|-----------------|------------------|
| **dSQ** (Shot Quality suppression) | Forcing opponents to take worse looks. Pre-shot EV reduction. | Kornet (+1.4 dSQ), Gobert (+1.2 dSQ) |
| **dContest** (Shot alteration) | Making opponents miss shots they'd normally make. Actual minus expected. | Wemby (+2.7 dContest), Gobert (+2.6 dContest), D. White (+2.2 dContest) |
| **dFT** (Free throw suppression) | Reducing opponent value from the foul line channel. | Gobert (+0.7 dFT) |

**Key insight on defensive mechanisms**: Gobert and Wemby are Contest-dominant -- they make guys miss shots they'd normally make (shot alteration). Kornet is SQ-dominant -- he forces bad shots in the first place (shot quality suppression). These are mechanically different defensive skills that look similar in total dTS. Derrick White shows up as a guard with rim-protector-level dContest (+2.2) -- elite perimeter contesting from a guard.

## Box-Score Regression Drivers (5Y Model)

| Metric | Ridge R^2 | XGBoost R^2 | Spearman |
|--------|----------|-------------|----------|
| **DEF_TS** | 0.405 | 0.489 | 0.596 |

**Top predictors** (in order of importance):
1. **STOPS** -- A composite defensive metric capturing stock-based defensive events
2. **rSTOPS** -- Rate-adjusted stops (per possession or per minute variant)
3. **dfga/100** -- Defended field goal attempts per 100 possessions. How often a player is in position to contest shots.
4. **rim_dfg%** -- Opponent field goal % at the rim when this player is the nearest defender. The most direct measure of rim protection.
5. **pos_C** -- Whether the player is a center. Centers have outsized defensive shooting impact because rim protection is the single most valuable defensive skill.

**What the R^2 = 0.405 means**: Box scores only explain ~40% of lineup-level defensive shooting impact. **This is the lowest R^2 of the efficiency factors.** The remaining ~60% is invisible defensive work -- communication, positioning, scheme execution, help defense, funneling, deterrence effects. This is why defensive RAPM is so valuable and so different from counting-stat defensive metrics. Box scores miss more than half of what's actually happening defensively.

**Why the gap matters**: When you see a player with ordinary defensive counting stats but elite dTS-RAPM, the invisible stuff is real. The lineup data is capturing defensive value that no box score can see.

## Player Examples (2021-2026 Time-Decayed)

### Elite dTS (Opponents Shoot Worse)
| Player | dTS | Context |
|--------|-----|---------|
| Rudy Gobert | +4.44 | The archetype. Best rim protector in the league. Opponents shoot dramatically worse with him on court. dSQ +1.2, dContest +2.6, dFT +0.7 |
| Chet Holmgren | +3.33 | Elite shot alteration from a young player. Length and timing. |
| Victor Wembanyama | +3.23 | The new Gobert-tier defender. dContest +2.7 -- pure shot alteration machine. |
| Derrick White | +2.57 | **A guard at +2.57 dTS is extraordinary.** dContest +2.2 -- perimeter contesting at rim-protector levels. |
| Jaren Jackson Jr. | +2.41 | Blocks, contests, rim protection |
| Draymond Green | +2.34 | The connective tissue defender. Communication, switching, help -- it all shows up in dTS. |
| Kevon Looney | +2.19 | Positional defense, scheme execution |
| Aaron Gordon | +2.18 | Versatile defender who impacts opponent shooting |
| Luke Kornet | +2.17 | SQ-dominant: dSQ +1.4, dContest +0.6. Forces bad looks rather than altering good ones. |
| Joel Embiid | +2.02 | Rim presence deters and alters shots |
| Steven Adams | +1.84 | Size and positioning suppress opponent efficiency |

### Negative dTS (Opponents Shoot Better)
| Player | dTS | Context |
|--------|-----|---------|
| LaMelo Ball | -1.55 | Opponents shoot significantly better with him on court. Poor defensive positioning/effort. |
| Stephen Curry | -1.25 | Elite offense comes with a defensive cost. Opponents target him. |
| Jalen Brunson | -1.27 | Undersized guard who gets targeted by opponents |
| Jamal Murray | -0.39 | Modest negative -- opponents shoot somewhat better |
| Luka Doncic | -0.14 | Slight negative, but less than reputation suggests |

**Pattern**: Undersized or defensively limited guards dominate the negative end. The best defensive impact comes from rim protectors (Gobert, Wemby, Chet) and elite perimeter defenders (White, Green). Position and size matter, but so do effort and IQ (Draymond is undersized but elite through communication and scheme).

### The Draymond Green Archetype
Green at +2.34 dTS despite modest individual defensive counting stats is the poster child for invisible defense showing up in lineup data. His value is communication, switching, help defense, and making the team's entire defensive scheme work better. Box scores barely see it. dTS-RAPM sees it clearly.

## The Gambling Tradeoff (dTS and dTOV Interaction)

This is one of the most interesting cross-factor interactions:

A player who gambles for steals in passing lanes will boost their **dTOV** -- they're genuinely creating turnovers. But the possessions where they gamble and miss, they've blown their defensive positioning, and that can show up as a **dTS cost** over time.

**Important framing**: The decomposition lets you see whether the dTS cost that you'd expect from a gambling habit has actually materialized over time. If a player's dTOV is elite but their dTS is suffering, that's *consistent with* the gambling hypothesis -- but it's not proof. There could be other things going on, and this is lineup-level, so teammates are part of the picture. The framework surfaces patterns worth investigating. It doesn't hand you conclusions.

**Example**: Alex Caruso has dTOV +3.10 (elite) AND dTS +1.48 (also positive). This suggests his steal/deflection production is NOT coming at a dTS cost -- either his gambling is controlled, or his positioning is good enough to recover, or his teammates (including rim protectors) are covering for the moments he gambles.

## Cross-Factor Interactions

- **dTS and dTOV (gambling tradeoff)**: See above. The most theoretically interesting interaction in the whole framework.
- **dTS and dREB**: Related but distinct. A rim protector like Gobert affects both (dTS +4.44, dREB +0.87), but some players specialize -- Bam Adebayo is +0.99 dTS but +1.37 dREB, more of a defensive rebounder than a shot alterer.
- **dTS and oTS**: The full efficiency picture. A player's Net Efficiency = beta(oTS) + beta(dTS). Gobert (-0.23 oTS, +4.44 dTS) is a one-way efficiency player. SGA (+2.89 oTS, +1.38 dTS) creates efficiency on both ends.
- **pos_C dominance**: Centers have the highest average dTS because rim protection is the most efficient defensive skill. The difference between a good and bad rim protector in dTS is larger than the difference between a good and bad perimeter defender.

## Caveats

- **Hardest to predict from box scores** (R^2 = 0.405). Most defensive impact is invisible. Don't dismiss a player's dTS just because their steal/block numbers are modest.
- **Lineup-level**: A player's dTS includes the rim protector behind them. A perimeter defender who funnels into Gobert will look better than the same player funneling into a mediocre center. The regression tries to control for this, but imperfectly.
- **For the pure series, no box priors**: Defensive RAPM is especially noisy with small samples. Multi-season time-decay helps, but read low-minute players' dTS values with appropriate uncertainty.
- **Scheme effects**: A player in a DROP scheme looks different than the same player in a SWITCH scheme. The regression captures outcomes, not intent.
