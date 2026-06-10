# dREB: Defensive Rebounding Rate RAPM (Opponent OREB Denial)

## Definition

How much a player affects the opponent's offensive rebounding rate when they are on the court, isolated via ridge regression controlling for every teammate and opponent.

**Units**: Opponent OREB% (native), converted to points via reconstruction beta weight.

**Sign convention**: Positive = opponents grab FEWER offensive rebounds when this player plays. Denying opponent OREBs = preventing second-chance points = ending opponent possessions sooner. This is a possession-denial factor.

**Important naming note**: Despite being called "dREB" in the column name, this measures opponent OREB suppression, not the player's own defensive rebounding directly. A positive dREB means the opponent's offensive rebounding rate decreases when this player is on court.

## What Shows Up Here

dREB captures everything that prevents opponents from extending possessions through offensive rebounds:

### Direct Rebounding Denial
- **Boxing out** -- physically preventing opponents from reaching the ball. Denying second chances. A center who boxes out for a guard to grab the board might not get credit in box score rebounding but shows up in team dREB when he plays.
- Defensive rebounding positioning and timing
- Grabbing contested defensive rebounds
- Tip-outs and deflections that prevent opponent second chances

### Scheme and Positioning
- **Box-out vs go-get-it strategies**: Some defensive schemes prioritize boxing out (even if a teammate grabs the board) while others have specific players crash for the rebound. Both affect team-level dREB.
- **A center who boxes out for a guard**: The center might not get the defensive rebound himself, but by boxing out the opponent's center, he enables a guard to grab it. His box score rebounding might look modest while his dREB-RAPM is positive.
- **Transition defense priority**: A team that prioritizes getting back in transition might sacrifice some defensive rebounding because fewer players are in rebounding position. This can show up as lower dREB for players in those schemes.

### Size and Physicality
- Larger players naturally occupy more space near the rim, reducing opponent OREB opportunities
- Physicality in boxing out is harder to measure than the outcome (whether the opponent gets the board)
- Wing players who crash defensively can also contribute to dREB

## Box-Score Regression Drivers (5Y Model)

| Metric | Ridge R^2 | XGBoost R^2 | Spearman |
|--------|----------|-------------|----------|
| **DEF_REB** | 0.175 | 0.258 | 0.382 |

**Top predictors** (in order of importance):
1. **STOPS** -- Composite defensive metric. Appears as top driver for both dTS and dREB, suggesting overall defensive engagement correlates with rebounding denial.
2. **DefFGRebPct** -- Defensive field goal rebound percentage. The most direct counting stat for securing defensive rebounds.
3. **DefReb/100** -- Defensive rebounds per 100 possessions. Volume measure.
4. **Blocks** -- Shot blockers tend to also be near the rim for defensive rebounds, and blocked shots can lead to controlled defensive possessions.
5. **pos_C** -- Whether the player is a center. Bigs near the rim dominate defensive rebounding.

**What the R^2 = 0.175 means**: Box scores explain only ~18% of lineup-level defensive rebounding impact. **This is BY FAR the lowest R^2 of all six factors.** Over 80% of the lineup-level signal is NOT captured by any individual defensive rebounding stat.

**Why this is so low**: Defensive rebounding is deeply team-dependent. Whether your team secures a defensive rebound depends on:
- Who is boxing out vs who is going for the ball
- Team scheme (crash vs outlet vs transition priority)
- Opponent offensive rebounding personnel and scheme
- The interaction between all 10 players on the court

Individual defensive rebound counting stats are heavily influenced by uncontested rebounds (boards that anyone could have grabbed) and don't distinguish between players who are boxing out (enabling rebounds) vs players who are collecting uncontested boards. The lineup regression captures the true team-level impact, which is mostly about positioning, effort, and team coordination -- things that don't show up in any counting stat.

**This makes dREB-RAPM particularly valuable**: Since box scores barely capture it, the lineup signal is telling you something that no other stat can.

## Player Examples (2021-2026 Time-Decayed)

### Elite dREB (Opponents Get Fewer OREBs)
| Player | dREB | Context |
|--------|------|---------|
| Nikola Jokic | +1.79 | Highest in the dataset. His team denies opponent OREBs dramatically. Size, positioning, and awareness. |
| Bam Adebayo | +1.37 | Elite boxing-out ability. Opponents don't get second chances. |
| Ivica Zubac | +1.23 | Big body that denies the offensive glass |
| Giannis Antetokounmpo | +1.20 | Combination of size, length, and effort on the glass |
| Steven Adams | +1.10 | Despite his offensive glass crashing (oREB +4.04), he also denies opponent OREBs |
| Victor Wembanyama | +1.10 | Length and positioning make it hard for opponents to crash |
| Franz Wagner | +1.10 | Versatile forward who contributes to rebounding denial |
| Rudy Gobert | +0.87 | Rim dweller who boxes out and secures the glass |
| Aaron Gordon | +0.84 | Physicality and effort on the defensive glass |
| Kevon Looney | +0.86 | Elite boxing-out specialist -- enables teammates to rebound |
| Isaiah Hartenstein | +0.81 | Modern big with defensive glass impact |

### Negative dREB (Opponents Get More OREBs)
| Player | dREB | Context |
|--------|------|---------|
| Kyrie Irving | -0.95 | Small guard, not boxing out, not contesting the glass |
| Tyrese Haliburton | -0.80 | Small guard archetype -- opponents crash his side |
| Tyrese Maxey | -0.65 | Same pattern -- small, fast guards sacrifice rebounding |
| Cason Wallace | -0.58 | Young guard, not yet a defensive rebounder |
| Paul Reed | -0.55 | Interesting for a big -- potentially out of position due to gambling for steals (dTOV +1.70) |
| Sam Merrill | -0.54 | Undersized wing |
| Jamal Murray | -0.46 | Guard who doesn't box out |

**Pattern**: Small guards consistently show negative dREB because they're physically unable to box out bigger opponents. The elite dREB players are almost all bigs -- centers and power forwards who occupy space near the rim and physically deny the offensive glass. The few exceptions (Franz Wagner at +1.10) are large, physical wings.

### The Jokic Defensive Rebounding Profile
Jokic at +1.79 dREB is the highest in the dataset despite not being known as a rim protector (dTS -0.34). His defensive value splits:
- **dREB +1.79**: Elite glass denial. Opponents don't get second chances.
- **dTS -0.34**: Slightly negative shot defense. Not a rim protector.
- **dTOV -0.49**: Opponents don't turn it over more when he plays.

His defensive impact is almost entirely through rebounding denial. Different mechanism than Gobert (dTS-dominant), Caruso (dTOV-dominant), or Wemby (dTS + dREB).

## Cross-Factor Interactions

### dREB and dTS
Related but distinct. Rim protectors often contribute to both (Gobert: dTS +4.44, dREB +0.87), but the correlation is imperfect:
- **Jokic**: dTS -0.34, dREB +1.79. Glass denial without shot defense.
- **Bam Adebayo**: dTS +0.99, dREB +1.37. Strong at both, but more a rebounder than a shot alterer.
- **Chet Holmgren**: dTS +3.33, dREB +0.21. Great shot blocker, modest rebounding denial.

### dREB and oREB (Both Ends of the Glass)
Some players dominate the glass on both ends:
- **Steven Adams**: oREB +4.04, dREB +1.10. The glass is his domain in every direction.
- **Gobert**: oREB +1.60, dREB +0.87. Similar bidirectional rebounding impact.
- **Hartenstein**: oREB +1.61, dREB +0.81. Modern version of the same archetype.

These players are "possession-flow controllers" on the glass -- they create extra possessions for their team and deny them to opponents.

### dREB and Transition
Defensive rebounding is the first step in transition offense. A team that secures defensive boards cleanly can push pace. A player whose dREB is high may also indirectly help oTS by enabling more transition opportunities (fewer second-chance situations = more clean defensive rebounds = more transition pushes).

### The Nene Case Study: Boxing Out in His Own Words

Nene is the definitive case study for why dREB-RAPM exists and why box-score rebounds are misleading. His personal DREB% was consistently low for a center -- he didn't grab a lot of boards himself. But in 3-year rolling dREB-RAPM, he ranked **#1 in the entire league** in 2013, 2014, 2015, and was top-5 for most of his prime, with dREB values consistently between +1.0 and +1.9.

In a [2015 Grantland interview](https://grantland.com/the-triangle/qa-nene-on-the-wizards-his-health-and-the-future/), he explained exactly what was happening:

> "If I don't box out, if I try to steal the ball from my teammates, I could average 13 rebounds or 14 rebounds per game. But I learned the right way. I learned to box out, respect each side of the hoop. There's a reason we have better rebounding when I play, because I know the fundamentals."

He also pointed out that boxing out isn't just a center responsibility: "Not only on the low block or the paint. The guys outside have to box out, so the little guys don't surprise you down there."

This is the R²=0.175 story in a single player. Box scores said Nene was an average rebounder. dREB-RAPM said he was literally the best in the league at denying opponent offensive rebounds. The mechanism -- fundamentally sound boxing out that enables teammates to grab the board -- is exactly the kind of impact that individual counting stats cannot see but lineup data captures clearly.

| End Szn | dREB +/- | dREB Rank | DRAPM +/- | DRAPM Rank |
|---------|----------|-----------|-----------|------------|
| 2013 | +1.9 | **1** | +4.3 | 5 |
| 2014 | +1.9 | **1** | +3.2 | 11 |
| 2015 | +1.7 | **1** | +2.7 | 18 |
| 2016 | +1.4 | **3** | +2.8 | 18 |
| 2017 | +1.4 | 4 | +3.1 | 9 |
| 2018 | +1.3 | 4 | +2.9 | 13 |

His defensive RAPM was elite throughout (top-20 in the league consistently), and dREB was always the dominant factor in his defensive profile. He wasn't a rim protector (dTS varied), he wasn't a ball hawk (dTOV modest) -- he was a defensive rebounding specialist whose value was almost entirely invisible to traditional stats.

### The Guard Problem
Small guards are almost always negative on dREB. They can't box out bigger players. This is a fundamental physical constraint, not a skill deficiency. When evaluating guard impact, dREB is typically a tax they pay, and their value needs to come from other factors (oTS, oTOV, dTOV, etc.).

## Caveats

- **Lowest box-score predictability** (R^2 = 0.175). Individual defensive rebounding stats are extremely poor predictors of lineup-level rebounding impact. Don't trust counting-stat defensive rebounding to tell you about a player's dREB-RAPM.
- **Heavily team-dependent**: More than any other factor, dREB is about team coordination, scheme, and the interaction between all 10 players. The regression tries to isolate individual contributions, but the signal-to-noise ratio is lower.
- **Boxing out vs grabbing**: A player who boxes out consistently but lets a teammate grab the board won't show up in defensive rebound counting stats but will show up in dREB-RAPM. This is a feature, not a bug -- the lineup data captures the actual team outcome.
- **Scheme confounding**: A team that prioritizes transition over defensive rebounding will show different dREB patterns than a team that prioritizes securing every board. Players in different schemes face different dREB contexts.
- **Uncontested rebounds inflating counting stats**: Many defensive rebounds are uncontested and would have gone to whoever was nearby. Individual DREB counting stats are contaminated by these "free" boards. dREB-RAPM captures contested rebounding impact more cleanly.
